"""Shared kit (vendored into ogun_core/kit): integrity of the vendored copy, pinned GPU options, and the model lease."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.kit import gpu_lease  # noqa: E402

KIT = Path(__file__).resolve().parents[1] / "ogun_core" / "kit"


def test_vendored_kit_matches_its_manifest():
    """Edit the canonical shared_kit and re-sync; a hand edit here would drift from the other app's copy."""
    m = json.loads((KIT / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in m["files"].items():
        assert hashlib.sha256((KIT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, f"{name} differs from the manifest"
    assert sorted(p.name for p in KIT.glob("*.py")) == sorted(m["files"])


def test_pinned_options_are_shared_and_configurable(monkeypatch):
    monkeypatch.setenv("KIT_GPU_LOCK", "1")
    assert gpu_lease.pinned_options() == {"num_gpu": 48, "num_ctx": 8192, "num_batch": 512, "num_thread": 12}
    monkeypatch.setenv("KIT_NUM_GPU", "49")
    assert gpu_lease.pinned("num_gpu") == 49
    monkeypatch.setenv("KIT_GPU_LOCK", "0")
    assert gpu_lease.pinned("num_gpu") is None


def test_app_sends_the_pinned_options_when_locked(monkeypatch):
    monkeypatch.setenv("KIT_GPU_LOCK", "1")
    from ogun_core import config
    opts = config.llm_options()
    assert (opts["num_gpu"], opts["num_ctx"], opts["num_batch"], opts["num_thread"]) == (48, 8192, 512, 12)
    from ogun_core import autotune
    assert autotune.refresh("http://127.0.0.1:1", "m", force=True) is None   # no re-tuning while pinned


def test_lease_blocks_unload_only_for_other_live_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("KIT_LEASE_DIR", str(tmp_path))
    with gpu_lease.acquire("app_a", "build") as lease:
        assert lease.path.exists()
        assert gpu_lease.may_unload(), "this process's own lease must not block its own unload"
    assert not lease.path.exists()
    # another live process holds a lease -> unload is refused
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "app_b-1.json").write_text(json.dumps({"app": "app_b", "pid": holder.pid, "purpose": "build"}))
        assert not gpu_lease.may_unload()
        assert "app_b (build)" in gpu_lease.unload_blocked_note()
    finally:
        holder.kill()
        holder.wait()
    # its process is gone -> the stale lease is cleaned up and unloading is allowed again
    assert gpu_lease.may_unload()
    assert not (tmp_path / "app_b-1.json").exists()


def test_unload_is_refused_while_another_app_holds_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("KIT_LEASE_DIR", str(tmp_path))
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "other-1.json").write_text(json.dumps({"app": "other", "pid": holder.pid, "purpose": "turn"}))
        from ogun_core.engine import handle_vram_command
        reply = handle_vram_command("/vram free", host="http://127.0.0.1:1")
        assert reply.startswith("Model kept loaded") and "other (turn)" in reply
    finally:
        holder.kill()
        holder.wait()
