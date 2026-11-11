"""Shared GPU settings and a model lease for every local app that talks to the same Ollama server.

Two problems this solves (measured 2026-11-10 on a 12 GB GPU):
1. Each app auto-tuned its own load options (one picked 47 GPU layers, the other 48; different num_batch/num_thread).
   Ollama reloads the model whenever load options differ, so alternating requests paid a reload each time.
   -> pinned_options(): every app sends the SAME num_gpu / num_ctx / num_batch / num_thread.
2. One app's exit hook or test run unloaded the model (keep_alive=0) in the middle of another app's build.
   -> acquire(): an app holds a lease while a job runs; may_unload() is False while any OTHER live process holds one.

The lease is one small JSON file per holder in a shared directory, so there is no read-modify-write race and nothing
ever blocks: acquiring writes a file, releasing deletes it, readers skip files whose process is gone.

Settings (environment):
  KIT_GPU_LOCK=1          pin load options (default on; 0 = each app keeps its own tuning)
  KIT_NUM_GPU=48          layers on the GPU            KIT_NUM_CTX=8192   context window
  KIT_NUM_BATCH=512       prompt batch size            KIT_NUM_THREAD=12  CPU threads
  KIT_LEASE_DIR=<dir>     lease directory (default %LOCALAPPDATA%/ollama_gpu_lease, or the temp dir)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULTS = {"num_gpu": 48, "num_ctx": 8192, "num_batch": 512, "num_thread": 12}
_ENV = {"num_gpu": "KIT_NUM_GPU", "num_ctx": "KIT_NUM_CTX", "num_batch": "KIT_NUM_BATCH", "num_thread": "KIT_NUM_THREAD"}


def locked() -> bool:
    """True when the shared load options are pinned (the default)."""
    return os.environ.get("KIT_GPU_LOCK", "1").strip().lower() not in ("0", "false", "no", "off")


def pinned_options() -> Dict[str, int]:
    """The load options every app sends to Ollama while locked()."""
    out = {}
    for key, default in DEFAULTS.items():
        try:
            out[key] = int(os.environ.get(_ENV[key], "") or default)
        except ValueError:
            out[key] = default
    return out


def pinned(key: str) -> Optional[int]:
    """One pinned option, or None when not locked (the app then uses its own value)."""
    return pinned_options()[key] if locked() else None


def lease_dir() -> Path:
    base = os.environ.get("KIT_LEASE_DIR") or os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    d = Path(base) if os.environ.get("KIT_LEASE_DIR") else Path(base) / "ollama_gpu_lease"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259                       # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lease:
    """A held lease. Use as a context manager, or call release()."""

    def __init__(self, app: str, purpose: str, model: str = "") -> None:
        self.app, self.purpose, self.model = app, purpose, model
        self.path = lease_dir() / f"{app}-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
        data = {"app": app, "pid": os.getpid(), "purpose": purpose, "model": model, "since": time.time(),
                "options": pinned_options() if locked() else None}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)

    def release(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def acquire(app: str, purpose: str, model: str = "") -> Optional[Lease]:
    """Hold the model for a job. Never raises and never blocks: returns None if the lease could not be written."""
    try:
        return Lease(app, purpose, model)
    except Exception:  # noqa: BLE001 - a lease is a courtesy; a job must not fail over it
        return None


def live_holders(exclude_pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """Leases held by running processes (stale files of dead processes are removed)."""
    out: List[Dict[str, Any]] = []
    try:
        files = list(lease_dir().glob("*.json"))
    except Exception:  # noqa: BLE001
        return out
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except Exception:  # noqa: BLE001
            continue
        if not pid_alive(pid):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        if exclude_pid is not None and pid == exclude_pid:
            continue
        out.append(data)
    return out


def may_unload() -> bool:
    """False while another live process holds a lease: unloading would evict the model from under its job."""
    return not live_holders(exclude_pid=os.getpid())


def unload_blocked_note() -> str:
    holders = live_holders(exclude_pid=os.getpid())
    if not holders:
        return ""
    who = ", ".join(f"{h.get('app')} ({h.get('purpose')})" for h in holders[:3])
    return f"Model kept loaded: another app is using it right now: {who}."
