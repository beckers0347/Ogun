"""Ogun's desktop window server: access control, the chat/build job flow, live activity, and window-close handling."""
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.ui import server as ui  # noqa: E402

TOKEN = "test-token-123"


class FakeEngine:
    def __init__(self):
        self.calls = []

    def dispatch_tool(self, name, args, turn_events=None):
        self.calls.append(name)
        return "ok"


class FakePlugins:
    def list_tools(self):
        return [{"type": "function", "function": {"name": "word_count", "description": "Counts words.",
                                                  "parameters": {"properties": {"text": {}}}}}]

    def discover_tools(self):
        return {}


def _start(gate=None, setup_error="", state=None):
    engine = FakeEngine()

    def setup(args):
        print("[*] tuning for this machine")
        return (None, None, setup_error) if setup_error else (engine, FakePlugins(), "")

    def handle_input(eng, text, model, note):
        print("[Ogun] thinking about it")
        eng.dispatch_tool("read_file", {"filepath": "a.py"})
        if gate:
            gate.wait(5)
        return f"echo: {text}"

    def build_plugin(eng, plugins, request_text):
        return True, f"built {request_text}"

    app = ui.OgunApp(SimpleNamespace(model="m:1"), state, setup=setup, handle_input=handle_input, build_plugin=build_plugin)
    srv = ui.ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = ui.make_handler(app, TOKEN, port)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    app.start_engine()
    return app, srv, port, engine


def _req(port, method, path, body=None, token=TOKEN, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if token:
        headers["X-Ogun-Token"] = token
    data = None
    if body is not None:
        data = body if isinstance(body, (bytes, str)) else json.dumps(body)
        headers["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw or b"{}")
    except ValueError:
        return r.status, raw


def _wait_idle(app, timeout=5):
    t0 = time.time()
    while app.state.busy and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert not app.state.busy


def test_page_is_served_but_the_api_needs_the_token_and_a_local_host():
    app, srv, port, _ = _start()
    try:
        status, body = _req(port, "GET", "/", token=None)
        assert status == 200 and b"<title>Ogun</title>" in body
        assert _req(port, "GET", "/api/state", token=None)[0] == 403
        assert _req(port, "GET", "/api/state", token="wrong")[0] == 403
        assert _req(port, "POST", "/api/chat", {"text": "rm everything"}, token=None)[0] == 403
        # a DNS-rebinding page reaches 127.0.0.1 under its own host name
        assert _req(port, "GET", "/", host=f"evil.example:{port}")[0] == 403
        assert _req(port, "GET", "/api/state", host=f"evil.example:{port}")[0] == 403
        assert _req(port, "GET", "/api/state")[0] == 200
    finally:
        srv.shutdown()


def test_chat_runs_one_job_with_live_activity_and_refuses_a_second(monkeypatch, tmp_path):
    gate = threading.Event()
    state = ui.UIState()
    monkeypatch.setattr(sys, "stdout", ui._ActivityStream(state, tmp_path / "ogun.log"))   # as run_ui does
    app, srv, port, engine = _start(gate=gate, state=state)
    try:
        status, s = _req(port, "GET", "/api/state")
        assert s["ready"] and s["model"] == "m:1"
        assert any(e["kind"] == "system" and "1 plugin loaded" in e["text"] for e in s["events"])
        assert any(e["kind"] == "activity" and "tuning" in e["text"] for e in s["events"]), "startup prints are shown"
        last = s["last_id"]

        assert _req(port, "POST", "/api/chat", {"text": "hello"})[0] == 202
        time.sleep(0.15)
        assert _req(port, "POST", "/api/chat", {"text": "again"})[0] == 409        # single stream
        assert _req(port, "POST", "/api/build", {"request": "x"})[0] == 409
        assert "busy" in _req(port, "POST", "/api/vram", {"arg": "free"})[1]["reply"]
        busy = _req(port, "GET", f"/api/state?after={last}")[1]
        assert busy["busy"] and busy["job"] == "chat"
        assert busy["status"].startswith("Running read_file(filepath=a.py)")
        gate.set()
        _wait_idle(app)

        evs = _req(port, "GET", f"/api/state?after={last}")[1]["events"]
        kinds = [e["kind"] for e in evs]
        assert kinds[0] == "user" and kinds[-1] == "assistant" and evs[-1]["text"] == "echo: hello"
        texts = [e["text"] for e in evs if e["kind"] == "activity"]
        assert "[Ogun] thinking about it" in texts and any(t.startswith("Running read_file") for t in texts)
        assert engine.calls == ["read_file"]
        assert _req(port, "POST", "/api/chat", {"text": "  "})[0] == 400
    finally:
        gate.set()
        srv.shutdown()


def test_build_and_plugins():
    app, srv, port, _ = _start()
    try:
        assert _req(port, "GET", "/api/plugins")[1]["plugins"] == [
            {"name": "word_count", "description": "Counts words.", "parameters": ["text"]}]
        assert _req(port, "POST", "/api/build", {"request": "a word counter"})[0] == 202
        _wait_idle(app)
        last = app.state.events[-1]
        assert last["kind"] == "assistant" and "Plugin built and installed" in last["text"] and "a word counter" in last["text"]
        assert _req(port, "POST", "/api/vram", {"arg": "format c:"})[0] == 400
    finally:
        srv.shutdown()


def test_a_failed_start_is_shown_in_the_window_and_can_be_retried():
    app, srv, port, _ = _start(setup_error="Cannot connect to Ollama at http://localhost:11434")
    try:
        s = _req(port, "GET", "/api/state")[1]
        assert not s["ready"] and "Cannot connect" in s["error"]
        assert any(e["kind"] == "error" and "could not start" in e["text"] for e in s["events"])
        assert _req(port, "POST", "/api/chat", {"text": "hi"})[0] == 409
        assert _req(port, "POST", "/api/retry", {})[1]["started"] is True
    finally:
        srv.shutdown()


def test_closing_the_window_says_goodbye_and_a_reload_cancels_it():
    app, srv, port, _ = _start()
    try:
        _req(port, "GET", "/api/state")
        # sendBeacon: no custom header, token in a text/plain body; a wrong token is ignored
        assert _req(port, "POST", "/api/bye", json.dumps({"token": "nope"}), token=None)[0] == 204
        assert app.state.bye_at == 0
        assert _req(port, "POST", "/api/bye", json.dumps({"token": TOKEN}), token=None)[0] == 204
        assert app.state.bye_at > 0 and app.state.last_seen < app.state.bye_at
        time.sleep(0.01)
        _req(port, "GET", "/api/state")                   # a reload polls again at once
        assert app.state.last_seen > app.state.bye_at
    finally:
        srv.shutdown()


def test_watchdog_exits_after_goodbye_but_waits_for_a_running_job(monkeypatch):
    monkeypatch.setattr(ui, "BYE_GRACE", 0.2)
    gate = threading.Event()
    app, srv, port, _ = _start(gate=gate)
    stopped = threading.Event()
    srv_shutdown = srv.shutdown
    srv.shutdown = lambda: (stopped.set(), srv_shutdown())
    try:
        app.state.last_seen, app.state.ever_seen = time.time(), True
        assert app.chat("long task")
        app.state.bye_at = time.time() + 0.01
        threading.Thread(target=ui._watchdog, args=(app, srv), daemon=True).start()
        assert not stopped.wait(1.5), "must not exit while the task is still running"
        gate.set()
        assert stopped.wait(3), "exits once the task finished"
        assert app.state.events[-1]["text"] == "echo: long task"
    finally:
        gate.set()


def test_quit_button_stops_the_server():
    app, srv, port, _ = _start()
    assert _req(port, "POST", "/api/quit", {})[1]["ok"]
    t = threading.Thread(target=ui._watchdog, args=(app, srv), daemon=True)
    t.start()
    t.join(4)
    assert not t.is_alive()


def test_print_capture_survives_a_missing_console(tmp_path):
    """Under pythonw sys.stdout is None; window mode replaces it, so prints never crash and land in the feed + log."""
    st = ui.UIState()
    stream = ui._ActivityStream(st, tmp_path / "ogun.log")
    print("=" * 70, file=stream)                      # banner rules are decoration, not activity
    print("first line\nsecond", file=stream, flush=True)
    print("  ------  ", file=stream)
    stream.write("partial")
    stream.flush()
    assert [e["text"] for e in st.events] == ["first line", "second"]
    assert "partial" in (tmp_path / "ogun.log").read_text(encoding="utf-8")
    stream.log.close()


def test_launcher_and_main_wiring():
    root = Path(__file__).resolve().parents[1]
    vbs = (root / "Ogun.vbs").read_text(encoding="utf-8")
    assert "run_ogun.bat" in vbs and "--ui" in vbs and ", 0, False" in vbs     # hidden console
    import main as ogun_main
    assert ogun_main.parse_args(["--ui"]).ui is True
    assert ogun_main.parse_args([]).ui is False
