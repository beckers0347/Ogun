"""Ogun's desktop window.

`python main.py --ui` (or Ogun.vbs, which hides the console) starts a small local server and opens the page in an Edge or
Chrome app window (no tabs or address bar). The whole program then runs from that window:
- the chat, with Ogun's tool calls and builder steps shown live;
- plugins and plugin builds;
- GPU memory controls;
- Quit.

Safety:
- The server listens on 127.0.0.1 only.
- Every API call needs a random token made at launch and a localhost Host header. Other web pages therefore can't drive
  Ogun (the token travels in a custom header, which a cross-site request can't send without a CORS preflight that this
  server never grants).

Lifetime: the window polls (a heartbeat) and says goodbye when it closes. When it is gone, Ogun exits. A turn that is
already running is allowed to finish first, so no file is left half written. The GPU is then freed as usual.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ogun_core import __version__

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
DEFAULT_PORT = 5010
HEARTBEAT_GRACE = 90.0          # no heartbeat at all (window crashed or killed): exit after this long
BYE_GRACE = 8.0                 # the window said goodbye and no page came back (a reload comes back at once)
FIRST_CONNECT_GRACE = 180.0     # the window never connected (no browser found and the user never opened the link)
MAX_EVENTS = 4000
_RULE = re.compile(r"^\s*[=\-_*#~.]{3,}\s*$")      # '=====' / '-----' banner lines: decoration, not activity


def state_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".ogun")
    d = Path(base) / "Ogun"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------------
# Shared state: an append-only event list the page polls with ?after=<last id>
# ---------------------------------------------------------------------------------------------------------------------
class UIState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self.next_id = 1
        self.busy = False
        self.job = ""                   # "chat" | "build" | "startup"
        self.status = "Starting..."
        self.started_at = 0.0
        self.ready = False
        self.error = ""
        self.model = ""
        self.last_seen = 0.0
        self.ever_seen = False
        self.bye_at = 0.0

    def add(self, kind: str, text: str, **extra: Any) -> Dict[str, Any]:
        with self.lock:
            ev = {"id": self.next_id, "kind": kind, "text": text, "ts": time.time(), **extra}
            self.next_id += 1
            self.events.append(ev)
            if len(self.events) > MAX_EVENTS:
                del self.events[: len(self.events) - MAX_EVENTS]
            if kind == "activity" and self.busy:
                self.status = text[:160]
            return ev

    def snapshot(self, after: int) -> Dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy, "job": self.job, "status": self.status, "ready": self.ready, "error": self.error,
                "model": self.model, "version": __version__,
                "elapsed": round(time.time() - self.started_at, 1) if self.busy else 0,
                "events": [e for e in self.events if e["id"] > after],
                "last_id": self.next_id - 1,
            }

    def begin(self, job: str, status: str) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy, self.job, self.status, self.started_at = True, job, status, time.time()
            return True

    def end(self, status: str = "Ready") -> None:
        with self.lock:
            self.busy, self.job, self.status = False, "", status


class _ActivityStream(io.TextIOBase):
    """Replaces stdout/stderr in window mode: each printed line becomes an activity event and goes to the log file.
    (Under pythonw there is no console at all, so sys.stdout would otherwise be None.)"""

    def __init__(self, state: UIState, log_path: Path, kind: str = "activity") -> None:
        self.state, self.kind, self.buf = state, kind, ""
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        try:
            self.log.write(s)
        except Exception:
            pass
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip()
            if line.strip() and not _RULE.match(line):
                self.state.add(self.kind, line)
        return len(s)

    def flush(self) -> None:
        try:
            self.log.flush()
        except Exception:
            pass


def _short_args(args: Any, limit: int = 140) -> str:
    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            v = str(v).replace("\n", " ")
            parts.append(f"{k}={v[:60] + '...' if len(v) > 60 else v}")
        text = ", ".join(parts)
    else:
        text = str(args).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def trace_tools(engine: Any, state: UIState) -> None:
    """Show every tool call in the window as it starts (the engine itself is unchanged)."""
    original = engine.dispatch_tool

    def traced(name: str, args: Any, *a: Any, **kw: Any) -> Any:
        state.add("activity", f"Running {name}({_short_args(args)})", tool=name)
        return original(name, args, *a, **kw)

    engine.dispatch_tool = traced


# ---------------------------------------------------------------------------------------------------------------------
# The app: owns the engine and runs one job at a time (single stream, like the GPU lease expects)
# ---------------------------------------------------------------------------------------------------------------------
class OgunApp:
    def __init__(self, args: Any, state: Optional[UIState] = None,
                 setup: Optional[Callable] = None, handle_input: Optional[Callable] = None,
                 build_plugin: Optional[Callable] = None) -> None:
        self.args = args
        self.state = state or UIState()
        self.engine = None
        self.plugins = None
        self._setup, self._handle_input, self._build_plugin = setup, handle_input, build_plugin
        self.shutdown_requested = threading.Event()

    def _entry(self, name: str) -> Callable:
        given = {"setup": self._setup, "handle_input": self._handle_input, "build_plugin": self._build_plugin}[name]
        if given is not None:
            return given
        import main as ogun_main      # the same functions the terminal chat uses
        return getattr(ogun_main, name)

    # -- startup ------------------------------------------------------------------------------------------------------
    def start_engine(self) -> None:
        st = self.state
        st.begin("startup", "Starting: tuning for this machine and connecting to Ollama...")
        st.add("activity", "Starting Ogun...")
        try:
            engine, plugins, error = self._entry("setup")(self.args)
        except Exception as exc:                                    # never leave the window stuck on "Starting"
            engine, plugins, error = None, None, f"{type(exc).__name__}: {exc}"
            st.add("activity", traceback.format_exc()[-1500:])
        if engine is None:
            with st.lock:
                st.error = error
            st.add("error", f"Ogun could not start: {error}")
            st.end("Not connected")
            return
        trace_tools(engine, st)
        self.engine, self.plugins = engine, plugins
        with st.lock:
            st.ready, st.error, st.model = True, "", self.args.model
        n = len(plugins.list_tools()) if plugins else 0
        st.add("system", f"Ready. Model {self.args.model}, {n} plugin{'s' if n != 1 else ''} loaded.")
        st.end("Ready")

    def retry_start(self) -> bool:
        if self.state.ready or self.state.busy:
            return False
        threading.Thread(target=self.start_engine, daemon=True, name="ogun-startup").start()
        return True

    # -- jobs -----------------------------------------------------------------------------------------------------------
    def _run_job(self, job: str, status: str, user_text: str, work: Callable[[], str]) -> bool:
        st = self.state
        if not st.ready:
            return False
        if not st.begin(job, status):
            return False
        st.add("user", user_text, job=job)

        def runner() -> None:
            try:
                reply = work()
                st.add("assistant", str(reply), job=job)
            except Exception as exc:
                st.add("activity", traceback.format_exc()[-2000:])
                st.add("error", f"{type(exc).__name__}: {exc}")
            finally:
                st.end("Ready")

        threading.Thread(target=runner, daemon=True, name=f"ogun-{job}").start()
        return True

    def chat(self, text: str) -> bool:
        handle = self._entry("handle_input")
        return self._run_job("chat", "Thinking...", text,
                             lambda: handle(self.engine, text, self.args.model,
                                            lambda m: self.state.add("activity", m)))

    def build(self, request_text: str) -> bool:
        build = self._entry("build_plugin")

        def work() -> str:
            ok, msg = build(self.engine, self.plugins, request_text)
            return ("Plugin built and installed.\n\n" if ok else "The plugin build did not finish.\n\n") + str(msg)

        return self._run_job("build", "Building plugin...", f"Build a plugin: {request_text}", work)

    def vram(self, arg: str) -> str:
        """GPU controls from Settings. Refused while a job runs (it would reload or evict the model mid-turn)."""
        if self.state.busy:
            return "Ogun is busy; GPU settings can be changed when the current task finishes."
        from ogun_core.engine import handle_vram_command
        reply = handle_vram_command(f"/vram {arg}".strip(), model=self.args.model)
        return reply if reply is not None else "Unknown GPU command."

    def plugin_list(self) -> List[Dict[str, Any]]:
        if not self.plugins:
            return []
        out = []
        for schema in self.plugins.list_tools():
            fn = schema.get("function", schema)
            params = (fn.get("parameters") or {}).get("properties") or {}
            out.append({"name": fn.get("name", "?"), "description": fn.get("description", ""),
                        "parameters": sorted(params)})
        return sorted(out, key=lambda p: p["name"])

    def reload_plugins(self) -> int:
        if self.plugins and not self.state.busy:
            self.plugins.discover_tools()
        return len(self.plugin_list())


# ---------------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------------
def make_handler(app: OgunApp, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "Ogun"

        def log_message(self, fmt: str, *a: Any) -> None:        # no per-request noise in the activity feed
            pass

        # -- helpers --
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts      # blocks DNS-rebinding pages

        def _authorized(self) -> bool:
            return self._host_ok() and secrets.compare_digest(self.headers.get("X-Ogun-Token", ""), token)

        def _body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 2_000_000:
                return {}
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        # -- routes --
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            if url.path in ("/", "/index.html"):
                html = INDEX_HTML.read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if url.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not self._authorized():
                return self._json(403, {"error": "forbidden"})
            if url.path == "/api/state":
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                app.state.last_seen, app.state.ever_seen = time.time(), True
                return self._json(200, app.state.snapshot(after))
            if url.path == "/api/plugins":
                return self._json(200, {"plugins": app.plugin_list()})
            return self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if url.path == "/api/bye" and self._host_ok():
                # sendBeacon can't set headers, so this one carries the token in its (text/plain) body
                if secrets.compare_digest(str(self._body().get("token", "")), token):
                    app.state.bye_at = time.time()
                return self._send(204, b"", "text/plain")
            if not self._authorized():
                return self._json(403, {"error": "forbidden"})
            data = self._body()
            if url.path == "/api/heartbeat":
                app.state.last_seen, app.state.ever_seen = time.time(), True
                return self._json(200, {"ok": True})
            if url.path == "/api/chat":
                text = str(data.get("text", "")).strip()
                if not text:
                    return self._json(400, {"error": "empty message"})
                if not app.state.ready:
                    return self._json(409, {"error": app.state.error or "Ogun is still starting."})
                if not app.chat(text):
                    return self._json(409, {"error": "Ogun is busy with the current task."})
                return self._json(202, {"ok": True})
            if url.path == "/api/build":
                text = str(data.get("request", "")).strip()
                if not text:
                    return self._json(400, {"error": "describe the plugin"})
                if not app.state.ready:
                    return self._json(409, {"error": app.state.error or "Ogun is still starting."})
                if not app.build(text):
                    return self._json(409, {"error": "Ogun is busy with the current task."})
                return self._json(202, {"ok": True})
            if url.path == "/api/vram":
                arg = str(data.get("arg", "status")).strip().lower()
                if arg not in ("status", "auto", "shared", "max", "cpu", "free"):
                    return self._json(400, {"error": "unknown GPU command"})
                return self._json(200, {"reply": app.vram(arg)})
            if url.path == "/api/plugins/reload":
                return self._json(200, {"count": app.reload_plugins(), "plugins": app.plugin_list()})
            if url.path == "/api/retry":
                return self._json(200, {"started": app.retry_start()})
            if url.path == "/api/quit":
                app.shutdown_requested.set()
                return self._json(200, {"ok": True, "busy": app.state.busy})
            return self._json(404, {"error": "not found"})

    return Handler


# ---------------------------------------------------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------------------------------------------------
def _browser_candidates() -> List[str]:
    env = os.environ
    names = [
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    found = [str(p) for p in names if str(p) and p.is_file()]
    for exe in ("msedge", "google-chrome", "chromium", "chrome"):
        w = shutil.which(exe)
        if w:
            found.append(w)
    return found


def open_window(url: str) -> str:
    """An app window (Edge/Chrome --app) with its own profile, so it is a separate window that closes cleanly.
    Falls back to the default browser."""
    profile = state_dir() / "window_profile"
    for exe in _browser_candidates():
        try:
            flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW: no console for the browser
            subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1320,880",
                              "--no-first-run", "--no-default-browser-check",
                              # a minimized window must keep polling, or Ogun would think it was closed
                              "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                              "--disable-backgrounding-occluded-windows"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             creationflags=flags)
            return exe
        except OSError:
            continue
    webbrowser.open(url)
    return "default browser"


def _existing_instance() -> Optional[Dict[str, Any]]:
    """Another Ogun window server already running? Then just show its window instead of starting a second engine."""
    info_file = state_dir() / "ui.json"
    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        req = urllib.request.Request(f"http://127.0.0.1:{info['port']}/api/state?after=999999999",
                                     headers={"X-Ogun-Token": info["token"]})
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.status == 200:
                return info
    except Exception:
        return None
    return None


def _watchdog(app: OgunApp, server: ThreadingHTTPServer) -> None:
    """Exit when the window is gone (no heartbeat), or on Quit. A running job is allowed to finish first."""
    st = app.state
    t0 = time.time()
    while True:
        time.sleep(1.0)
        now = time.time()
        said_bye = st.bye_at > 0 and st.last_seen < st.bye_at and now - st.bye_at > BYE_GRACE
        gone = said_bye or (st.ever_seen and now - st.last_seen > HEARTBEAT_GRACE) or \
               (not st.ever_seen and now - t0 > FIRST_CONNECT_GRACE)
        if (gone or app.shutdown_requested.is_set()) and not st.busy:
            server.shutdown()
            return


def run_ui(args: Any, port: Optional[int] = None, open_browser: bool = True) -> int:
    existing = _existing_instance()
    if existing:
        open_window(f"http://127.0.0.1:{existing['port']}/#t={existing['token']}")
        return 0

    state = UIState()
    log_path = state_dir() / "ogun.log"
    sys.stdout = _ActivityStream(state, log_path)
    sys.stderr = _ActivityStream(state, log_path)

    port = port or int(os.environ.get("OGUN_UI_PORT", DEFAULT_PORT))
    token = secrets.token_urlsafe(24)
    app = OgunApp(args, state)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, token, port))
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), None)       # port taken by something else: any free port
        port = server.server_address[1]
        server.RequestHandlerClass = make_handler(app, token, port)
    server.daemon_threads = True

    info_file = state_dir() / "ui.json"
    info_file.write_text(json.dumps({"port": port, "token": token, "pid": os.getpid()}), encoding="utf-8")

    threading.Thread(target=app.start_engine, daemon=True, name="ogun-startup").start()
    threading.Thread(target=_watchdog, args=(app, server), daemon=True, name="ogun-watchdog").start()
    url = f"http://127.0.0.1:{port}/#t={token}"
    if open_browser:
        state.add("activity", f"Window opened with {open_window(url)}.")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        try:
            if json.loads(info_file.read_text(encoding="utf-8")).get("pid") == os.getpid():
                info_file.unlink()
        except Exception:
            pass
    return 0
