"""Turn controller for the general loop: PLAN -> EXECUTE (tool rounds) -> VERIFY, before a turn may end as done.

Before this, a turn ended the moment the model replied without a tool call: a to-do list, an untested edit or a file
with a syntax error all counted as success. Now:

PLAN     one schema-constrained call (sampling "plan") returns concrete steps, each with an acceptance check
         (file_exists / tests_pass / syntax_ok / answer_only and its target). The plan is pinned in the context.
EXECUTE  the usual tool rounds, through the context ledger; every step is written to the run journal.
VERIFY   when the model stops calling tools: every touched .py file must parse, every touched or planned test must
         pass, every planned file must exist, and a schema-constrained completion verdict (sampling "verdict") must
         agree that the request is done (not a to-do list, no placeholders, nothing left to the user). Anything
         missing goes back to the model as NOT DONE, up to MAX_VERIFY_CYCLES times.

Only clients that support structured output (OllamaClient.supports_structured) get the plan/verify stages; simple
test doubles keep the plain loop.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ogun_core.kit.sampling import profile
from ogun_core.kit.schema_check import validate
from ogun_core.kit.structured import parse_json

MAX_VERIFY_CYCLES = 3
PLAN_ROUND_BUDGET = 30

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "acceptance": {"type": "string", "enum": ["file_exists", "tests_pass", "syntax_ok", "answer_only"]},
                    "target": {"type": "string"},
                },
                "required": ["goal", "acceptance", "target"],
            },
        },
    },
    "required": ["steps"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"done": {"type": "boolean"}, "missing": {"type": "array", "items": {"type": "string"}}},
    "required": ["done", "missing"],
}


def _structured(engine: Any, prompt: str, schema: Dict[str, Any], phase: str, num_predict: int) -> Optional[Dict]:
    msgs = [{"role": "user", "content": prompt}]
    for _ in range(2):
        try:
            reply = engine._chat(msgs, tools=None, options={**profile(phase), "num_predict": num_predict}, fmt=schema)
        except Exception:  # noqa: BLE001 - a planning/verdict failure must not end the turn
            return None
        data = parse_json((reply or {}).get("content", ""))
        problems = ["not valid JSON"] if data is None else validate(data, schema)
        if not problems:
            return data
        msgs = msgs + [{"role": "assistant", "content": (reply or {}).get("content", "")},
                       {"role": "user", "content": "Rejected: " + "; ".join(problems[:5]) + ". Reply again."}]
    return None


def make_plan(engine: Any, task: str, tool_names: List[str]) -> Optional[Dict[str, Any]]:
    prompt = (
        "Plan how to complete this request with the available tools. Give 1-8 concrete steps. For each step say how "
        "its completion is checked: 'file_exists' (target: the file path), 'tests_pass' (target: the test file path), "
        "'syntax_ok' (target: the .py file) or 'answer_only' (target: '' - for steps that only answer in words).\n"
        f"Tools: {', '.join(tool_names)}\n"
        "Paths are relative to the project root (new projects go under workspace/<name>/).\n\n"
        f"REQUEST:\n{task}")
    return _structured(engine, prompt, PLAN_SCHEMA, "plan", 500)


def format_plan(plan: Dict[str, Any]) -> str:
    # The checks are written as sentences, not labels: live, a step shown as "answer_only" made the model reply
    # with a fake tool call named answer_only.
    say = {"file_exists": "{t} exists", "tests_pass": "{t} passes", "syntax_ok": "{t} parses",
           "answer_only": "you have answered in plain words"}
    lines = []
    for i, s in enumerate(plan.get("steps", []), 1):
        check = say.get(s["acceptance"], s["acceptance"]).format(t=s.get("target") or "the file")
        lines.append(f"{i}. {s['goal']}  (done when {check})")
    return "\n".join(lines) + ("\nUse tools where a step needs them. Finish only when every check holds; then reply "
                               "in plain words (a short summary, or the answer itself).")


def _resolve(base_dir: Path, target: str) -> Optional[Path]:
    t = (target or "").strip().strip("`'\"")
    if not t:
        return None
    p = Path(t)
    return p if p.is_absolute() else (base_dir / p)


def needs_verdict(plan: Optional[Dict[str, Any]], touched: List[Path]) -> bool:
    """The model verdict is only asked when the mechanical checks cannot establish completion: a step that is answered
    in words, or a plan without any file/test check. Live 2026-11-10: with all tests passing, the verdict demanded a
    commit message and a README the request never asked for, and the model then edited the test file to 'comply'."""
    steps = (plan or {}).get("steps", [])
    if any(s.get("acceptance") == "answer_only" for s in steps):
        return True
    return not any(s.get("acceptance") in ("file_exists", "tests_pass", "syntax_ok") for s in steps)


def mechanical_problems(engine: Any, plan: Optional[Dict[str, Any]], events: Any, run_test
                        ) -> Tuple[List[str], List[Path], Dict[str, Any]]:
    """Checks that need no model: touched .py files parse, touched/planned tests pass, planned files exist."""
    base = Path(engine.base_dir)
    touched = []
    for p in getattr(events, "files_written", []) or []:
        p = Path(p)
        if p.exists() and p not in touched:
            touched.append(p)
    problems: List[str] = []
    details: Dict[str, Any] = {"touched": [str(p) for p in touched], "tests": {}}

    for p in touched:
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as e:
                problems.append(f"{p.name} does not parse: SyntaxError at line {e.lineno}: {e.msg}")

    tests: List[Path] = [p for p in touched if p.suffix == ".py" and p.name.startswith("test_")]
    for p in touched:
        sib = p.with_name(f"test_{p.name}")
        if p.suffix == ".py" and not p.name.startswith("test_") and sib.exists() and sib not in tests:
            tests.append(sib)
    for s in (plan or {}).get("steps", []):
        target = _resolve(base, s.get("target", ""))
        if target is None:
            continue
        if s.get("acceptance") == "file_exists" and not target.exists():
            problems.append(f"{s.get('target')} does not exist yet (step: {s.get('goal')})")
        if s.get("acceptance") == "tests_pass":
            if not target.exists():
                problems.append(f"the test {s.get('target')} does not exist yet (step: {s.get('goal')})")
            elif target not in tests:
                tests.append(target)
    for t in tests:
        out = str(run_test(str(t)))
        passed = out.startswith("TEST PASS")
        details["tests"][str(t)] = passed
        if not passed:
            problems.append(f"{t.name} fails: " + " ".join(out.split())[:600])
    return problems, touched, details


def verify(engine: Any, task: str, plan: Optional[Dict[str, Any]], events: Any, final_reply: str,
           run_test) -> Tuple[bool, List[str], Dict[str, Any]]:
    """(done, problems, details). Mechanical checks first; the completion verdict when they all hold.

    The verdict runs for every planned turn, answer-only ones included: live, an answer-only turn ended with "I will
    read the content of the file." and no answer.
    """
    base = Path(engine.base_dir)
    problems, touched, details = mechanical_problems(engine, plan, events, run_test)
    if problems:
        return False, problems, details
    if not needs_verdict(plan, touched):
        return True, [], details

    files_note = "\n".join(f"- {p.relative_to(base) if p.is_relative_to(base) else p}" for p in touched) or "(none)"
    tests_note = "\n".join(f"- {Path(t).name}: {'PASS' if ok else 'FAIL'}" for t, ok in details["tests"].items()) \
        or "(none run)"
    prompt = (
        "Decide whether this request is FULLY done. Judge ONLY against what the request literally asks: never require "
        "extras it did not ask for (documentation, READMEs, commit messages, explanations of the change, refactoring, "
        "more tests). Say done=false if the final reply is a to-do list or a plan, only announces what it is going "
        "to do instead of doing it, leaves work to the user, contains placeholders or TODOs, or skips a part of the "
        "request; list only what the request asked for and is missing.\n\n"
        f"REQUEST:\n{task}\n\nFILES WRITTEN THIS TURN:\n{files_note}\n\nTESTS RUN:\n{tests_note}\n\n"
        f"FINAL REPLY:\n{(final_reply or '')[-2000:]}")
    verdict = _structured(engine, prompt, VERDICT_SCHEMA, "verdict", 300)
    details["verdict"] = verdict
    if verdict and not verdict.get("done"):
        missing = [m for m in verdict.get("missing", []) if isinstance(m, str) and m.strip()] or \
                  ["the request is not fully done yet"]
        problems += missing
    return (not problems), problems, details


REFUSE_AFTER_REPEATS = 5


def error_lines(result: str, limit: int = 4) -> List[str]:
    """The lines of a tool result that name the actual error (ImportError: ..., AssertionError: ..., File ... line N)."""
    lines = [ln.strip() for ln in str(result).splitlines() if ln.strip()]
    hits = [ln for ln in lines if ("Error" in ln or "error:" in ln or "Exception" in ln) and not ln.startswith("=")]
    return (hits or lines)[-limit:]


def stall_message(call: str, problems: List[str], evidence: Optional[List[str]] = None) -> str:
    """Live 2026-11-10: the model re-ran the same failing test ~20 times and never wrote the module it imported, then
    (after one note) ran it ~20 more times instead of fixing the relative import the error named."""
    note = (f"You have now run {call} with the same result more than once; repeating it will not change anything. "
            "Change approach: edit the file the error points to, then run it again.")
    if evidence:
        note += "\nThe error it reports every time:\n  " + "\n  ".join(evidence)
    if problems:
        note += " The plan checks currently report:\n- " + "\n- ".join(problems[:6]) + "\nDo the missing step now."
    return note


def not_done_message(problems: List[str]) -> str:
    return ("NOT DONE yet. The automatic checks found:\n- " + "\n- ".join(problems[:8])
            + "\nFix these with tools (do not just describe the fix), then reply with a short summary.")
