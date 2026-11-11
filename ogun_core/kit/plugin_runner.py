"""Run a plugin's execute() in its own process, with a hard timeout.

In-process execution used a worker thread: a timeout returned an error but could not stop the thread, so a hung plugin
kept running inside the agent, and a plugin that crashed the interpreter took the agent down with it. Here each call is
a child process: on timeout the whole process tree is killed (taskkill /T /F on Windows, killpg elsewhere) and an
error result comes back; a crash is just a failed call.

    result = run_isolated(app_root, plugin_file, {"text": "hi"}, timeout=30)
    -> {"status": "success" | "error", "output": ..., "error_trace": ...}

Security is unchanged: the child gets exactly what the plugin had in-process (the app root on sys.path).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_MARK = "<<<PLUGIN_RESULT>>>"

# The child: import the plugin file as plugins.custom_tools.<stem>, call execute(**kwargs), print the result as JSON.
_CHILD = r'''
import importlib.util, inspect, json, sys, traceback
root, plugin_file, mark = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
kwargs = json.loads(sys.stdin.read() or "{}")
def emit(d):
    sys.stdout.write("\n" + mark + json.dumps(d, default=str) + "\n"); sys.stdout.flush()
try:
    import pathlib
    stem = pathlib.Path(plugin_file).stem
    spec = importlib.util.spec_from_file_location("plugins.custom_tools." + stem, plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    classes = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if c.__module__ == mod.__name__ and callable(getattr(c, "execute", None))
               and any(b.__name__ == "BaseTool" for b in c.__mro__[1:])]
    if not classes:
        emit({"status": "error", "output": None, "error_trace": "no BaseTool class in " + plugin_file}); sys.exit(0)
    res = classes[0]().execute(**kwargs)
    if hasattr(res, "status") and not isinstance(res, dict):
        res = {"status": res.status, "output": getattr(res, "output", None), "error_trace": getattr(res, "error_trace", None)}
    elif isinstance(res, dict):
        res = {"status": res.get("status", "success"), "output": res.get("output", res), "error_trace": res.get("error_trace")}
    else:
        res = {"status": "success", "output": res, "error_trace": None}
    emit(res)
except BaseException as exc:
    emit({"status": "error", "output": None, "error_trace": "Execution exception: %s\n%s" % (exc, traceback.format_exc())})
'''


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def run_isolated(app_root: Path, plugin_file: Path, kwargs: Dict[str, Any], timeout: float = 30.0,
                 extra_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    paths = [str(app_root)] + list(extra_paths or []) + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        payload = json.dumps(kwargs, default=str)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "output": None, "error_trace": f"Arguments are not JSON-serializable: {exc}"}
    started = time.time()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, str(app_root), str(plugin_file), _MARK],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(app_root),
                            env=env, text=True, encoding="utf-8", errors="replace",
                            start_new_session=(sys.platform != "win32"))
    try:
        out, err = proc.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return {"status": "error", "output": None,
                "error_trace": f"Execution timed out after {timeout:g}s; the plugin process was terminated."}
    for line in reversed((out or "").splitlines()):
        if line.startswith(_MARK):
            try:
                return json.loads(line[len(_MARK):])
            except ValueError:
                break
    return {"status": "error", "output": None,
            "error_trace": f"The plugin process exited with code {proc.returncode} after {time.time() - started:.1f}s "
                           f"without a result. stderr: {(err or '').strip()[-800:]}"}
