"""Append-only run journal: one JSONL file per task, so a turn can be resumed after a timeout, an error or a restart.

    j = RunJournal.start(directory, task)        # new task
    j.log("plan", plan={...}); j.log("tool", receipt="[Tool receipt: ...]"); j.log("verify", done=False, problems=[...])
    j2 = RunJournal.latest(directory, task)      # the most recent journal for this task (or None)
    j2.state()  -> {"task", "plan", "receipts", "verdicts", "status"}

Files older than `keep_days` are removed when a new journal starts.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunJournal:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def start(cls, directory: Path, task: str, keep_days: float = 7.0) -> "RunJournal":
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - keep_days * 86400
        for old in d.glob("*.jsonl"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except OSError:
                pass
        task_id = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        j = cls(d / f"{task_id}.jsonl")
        j.log("task", task=task)
        return j

    @classmethod
    def latest(cls, directory: Path, task: Optional[str] = None) -> Optional["RunJournal"]:
        d = Path(directory)
        if not d.is_dir():
            return None
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            j = cls(f)
            if task is None or j.state().get("task") == task:
                return j
        return None

    def log(self, kind: str, **data: Any) -> None:
        rec = {"t": round(time.time(), 3), "kind": kind, **data}
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass

    def events(self) -> List[Dict[str, Any]]:
        out = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return out

    def state(self) -> Dict[str, Any]:
        st: Dict[str, Any] = {"task": "", "plan": None, "receipts": [], "verdicts": [], "status": "running"}
        for e in self.events():
            k = e.get("kind")
            if k == "task":
                st["task"] = e.get("task", "")
            elif k == "plan":
                st["plan"] = e.get("plan")
            elif k == "tool":
                st["receipts"].append(e.get("receipt", ""))
            elif k == "verify":
                st["verdicts"].append({"done": e.get("done"), "problems": e.get("problems", [])})
            elif k == "end":
                st["status"] = e.get("status", "done")
        return st
