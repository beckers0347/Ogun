"""Context ledger: what goes into the model's context window, under an exact token budget.

Zones (built into the message list in this order):
  ANCHOR   system prompt + plan (one system message) and the current task (one user message). Never evicted or
           truncated, whatever the budget pressure.
  JOURNAL  rolling "state of work" lines (milestones, receipts of evicted steps), appended to the system message.
  HISTORY  earlier conversation turns (evicted first, into one-line journal entries).
  WORKING  this task's tool interactions and follow-ups. An assistant message keeps its tool_calls, and every tool
           result keeps the tool's name, so a call and its result always travel together.

Tool outputs over `tool_output_cap` tokens are saved whole to `spill_dir` and replaced by a preview (head and tail)
plus a footer with size, line count, sha256 and the saved path. Outputs that fall out of the recent window become
one-line receipts: [Tool receipt: read_file('a.py') -> 420 lines, 12034 bytes, sha256:3f1c2a9b].

Compaction order when over budget: older tool outputs -> receipts; oldest history turns -> journal lines; oldest
working step groups -> journal receipts; oldest journal lines; finally even recent tool outputs -> receipts.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _short_args(arguments: Any, limit: int = 80) -> str:
    if isinstance(arguments, dict):
        parts = []
        for k, v in arguments.items():
            s = repr(v) if not isinstance(v, str) else repr(v if len(v) <= 40 else v[:37] + "...")
            parts.append(s if len(arguments) == 1 else f"{k}={s}")
        text = ", ".join(parts)
    else:
        text = repr(arguments)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def receipt_for(tool_name: str, arguments: Any, output: str) -> str:
    data = (output or "").encode("utf-8", errors="replace")
    first = next((ln.strip() for ln in (output or "").splitlines() if ln.strip()), "")
    lead = f" {first[:60]!r}" if first and len(first) < 200 else ""
    return (f"[Tool receipt: {tool_name}({_short_args(arguments)}) -> {len((output or '').splitlines())} lines, "
            f"{len(data)} bytes, sha256:{hashlib.sha256(data).hexdigest()[:8]};{lead}]")


class ContextLedger:
    def __init__(self, count_messages: Callable[[List[Dict[str, Any]]], int], budget_tokens: int,
                 count_text: Optional[Callable[[str], int]] = None, tool_output_cap: int = 1200,
                 keep_recent_results: int = 2, spill_dir: Optional[Path] = None) -> None:
        self.count_messages = count_messages
        self.count_text = count_text or (lambda s: count_messages([{"role": "user", "content": s}]))
        self.budget = int(budget_tokens)
        self.tool_output_cap = int(tool_output_cap)
        self.keep_recent_results = int(keep_recent_results)
        self.spill_dir = Path(spill_dir) if spill_dir else None
        self.system = ""
        self.plan = ""
        self.task = ""
        self.journal: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.working: List[Dict[str, Any]] = []
        self.dropped_journal = 0

    # ---------------- anchors / journal
    def set_system(self, text: str) -> None:
        self.system = text or ""

    def set_task(self, text: str) -> None:
        self.task = text or ""

    def set_plan(self, text: str) -> None:
        self.plan = text or ""

    def note(self, line: str) -> None:
        if line:
            self.journal.append(line.strip())

    # ---------------- messages
    def add_history(self, messages: List[Dict[str, Any]]) -> None:
        self.history.extend(dict(m) for m in messages if m.get("role") != "system")

    def add_user(self, content: str) -> None:
        self.working.append({"role": "user", "content": content or ""})

    def add_assistant(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = [{"function": {"name": (tc.get("function") or tc).get("name"),
                                               "arguments": (tc.get("function") or tc).get("arguments") or {}}}
                                 for tc in tool_calls]
        self.working.append(msg)

    def add_tool_result(self, tool_name: str, arguments: Any, output: str) -> str:
        """Store a tool result (previewed and spilled when large). Returns the text that was stored."""
        output = "" if output is None else str(output)
        stored = output
        if self.count_text(output) > self.tool_output_cap:
            stored = self._preview(tool_name, arguments, output)
        self.working.append({"role": "tool", "tool_name": tool_name, "content": stored,
                             "_receipt": receipt_for(tool_name, arguments, output)})
        return stored

    def _spill(self, output: str) -> str:
        if not self.spill_dir:
            return ""
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            cutoff = time.time() - 86400
            for old in self.spill_dir.glob("*.txt"):
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            path = self.spill_dir / (hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()[:16] + ".txt")
            path.write_text(output, encoding="utf-8", errors="replace")
            return str(path)
        except OSError:
            return ""

    def _preview(self, tool_name: str, arguments: Any, output: str) -> str:
        lines = output.splitlines()
        head: List[str] = []
        budget = int(self.tool_output_cap * 0.6)
        used = 0
        for ln in lines:
            c = self.count_text(ln + "\n")
            if used + c > budget:
                break
            head.append(ln)
            used += c
        tail: List[str] = []
        used = 0
        for ln in reversed(lines[len(head):]):
            c = self.count_text(ln + "\n")
            if used + c > int(self.tool_output_cap * 0.25):
                break
            tail.insert(0, ln)
            used += c
        omitted = len(lines) - len(head) - len(tail)
        path = self._spill(output)
        footer = (f"[Output truncated: {len(lines)} lines, {len(output.encode('utf-8', errors='replace'))} bytes, "
                  f"sha256:{hashlib.sha256(output.encode('utf-8', errors='replace')).hexdigest()[:8]}; "
                  f"{omitted} lines omitted"
                  + (f"; full text saved to {path} - read it with read_file and start_line/end_line" if path else "")
                  + "]")
        return "\n".join(head + ([f"... ({omitted} lines omitted) ..."] if omitted > 0 else []) + tail) + "\n" + footer

    def _system_message(self) -> Dict[str, Any]:
        text = self.system
        if self.plan:
            text += "\n\n## Current plan\n" + self.plan
        if self.journal or self.dropped_journal:
            lines = ([f"({self.dropped_journal} earlier entries omitted)"] if self.dropped_journal else []) + self.journal
            text += "\n\n## State of work so far\n" + "\n".join(f"- {ln}" for ln in lines)
        return {"role": "system", "content": text}

    def messages(self) -> List[Dict[str, Any]]:
        out = [self._system_message()]
        out += [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.history]
        out.append({"role": "user", "content": self.task})
        out += [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.working]
        return out

    def total_tokens(self) -> int:
        return self.count_messages(self.messages())

    # ---------------- compaction
    def _tool_indexes(self) -> List[int]:
        return [i for i, m in enumerate(self.working) if m.get("role") == "tool"]

    def _to_receipt(self, i: int) -> bool:
        m = self.working[i]
        if m.get("_receipt") and m.get("content") != m["_receipt"]:
            m["content"] = m["_receipt"]
            return True
        return False

    def _pop_oldest_working_group(self) -> bool:
        """Remove the oldest step (an assistant message plus the tool results that answer it) into the journal."""
        if len(self.working) <= 1:
            return False
        group = [self.working.pop(0)]
        while self.working and self.working[0].get("role") == "tool":
            group.append(self.working.pop(0))
        for m in group:
            if m.get("role") == "tool":
                self.journal.append(m.get("_receipt") or "[tool result]")
            elif m.get("role") == "assistant" and m.get("content"):
                self.journal.append("Assistant: " + " ".join(m["content"].split())[:160])
            elif m.get("role") == "user":
                self.journal.append("User follow-up: " + " ".join((m.get("content") or "").split())[:160])
        return True

    def compact(self) -> int:
        """Bring the context under budget without touching the anchors. Returns the resulting token count."""
        total = self.total_tokens()
        if total <= self.budget:
            return total
        tools = self._tool_indexes()
        for i in tools[: max(0, len(tools) - self.keep_recent_results)]:
            if self._to_receipt(i):
                total = self.total_tokens()
                if total <= self.budget:
                    return total
        while self.history and total > self.budget:
            m = self.history.pop(0)
            role = "User" if m.get("role") == "user" else "Assistant"
            self.journal.append(f"Earlier {role.lower()}: " + " ".join((m.get("content") or "").split())[:160])
            total = self.total_tokens()
        while total > self.budget and len(self.working) > 2 and self._pop_oldest_working_group():
            total = self.total_tokens()
        while total > self.budget and self.journal:
            self.journal.pop(0)
            self.dropped_journal += 1
            total = self.total_tokens()
        for i in self._tool_indexes():
            if total <= self.budget:
                break
            if self._to_receipt(i):
                total = self.total_tokens()
        while total > self.budget and len(self.working) > 1 and self._pop_oldest_working_group():
            total = self.total_tokens()
        while total > self.budget and self.journal:
            self.journal.pop(0)
            self.dropped_journal += 1
            total = self.total_tokens()
        return total
