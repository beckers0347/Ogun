"""Ogun dynamic plugin staged builder and self-correction loop.

Coordinates staged code generation (tool implementation, then unit tests) and iterative self-correction. With
max_attempts <= 0 there is no attempt limit: the loop researches failures that keep coming back and keeps going
until the plugin passes its test or the user presses Ctrl+C.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Callable, List, Optional, Tuple

from ogun_core.engine import CoreEngine
from ogun_core.plugin_manager import PluginManager
from ogun_core.quality import check_requirements_coverage, fit_feedback, research_build_failure
from ogun_core.kit.build_kit import (  # noqa: F401 - shared helpers, re-exported
    BASE_TEMPERATURE,
    MAX_FIXES_PER_DRAFT,
    EXAMPLE_WAIVE_AFTER,
    FRESH_DRAFT_TEMPERATURE,
    FIX_TEMPERATURE,
    _BOOT,
    _tool_class,
    normalize_plugin_test,
    _FAIL_LINE_RE,
    disputed_case,
    DROP_DISPUTED_AFTER,
    case_quoted_in_request,
    drop_case_line,
    arbitrate_case,
    same_failure_streak,
    retry_temperature,
    is_tool_block,
    TEST_SUSPECT_NOTE,
    STUCK_NOTE,
    tool_code_error_context,
)


def parse_code_blocks(text: str) -> List[str]:
    """Extract fenced python code blocks from text."""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    return [b.strip() for b in blocks if b.strip()]


def _research_plugin_failure(engine: CoreEngine, request_text: str, last_error: str, tool_code: str) -> str:
    """Web search + root-cause diagnosis for a plugin failure that keeps coming back ('' when unavailable)."""
    client = getattr(engine, "client", None)
    if client is None:
        return ""
    real_client = type(client).__module__ != "unittest.mock" and not hasattr(client, "responses") \
        and not hasattr(client, "call_history")
    search = None
    if real_client:
        from ogun_core.research import search_web, web_research_enabled
        if web_research_enabled():
            search = lambda q: search_web(q, 3)  # noqa: E731

    def _chat(text: str, max_tokens: int) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            r = client.chat(msgs, tools=None, options={"num_predict": max_tokens, "temperature": 0.3})
        except TypeError:
            r = client.chat(msgs, tools=None)
        return (r or {}).get("content", "")

    try:
        return research_build_failure(_chat, request_text, last_error, tool_code, search)
    except Exception as exc:  # noqa: BLE001
        print(f"[Ogun Plugin Builder] research step failed: {exc}", flush=True)
        return ""


# Live 2026-11-09 (free GPU target: a plugin in under 10 minutes): 5 identical failures before a restart wasted
# attempts that all failed the same way; 3 is enough to tell the code is stuck.
RESTART_AFTER_SAME_FAILURES = 3
STAGE_MAX_TOKENS = 3072

# Short build prompt (the long PLUGIN_BUILDER_PROMPT with its template made replies ~35% longer with no better pass rate
# in the same measurement). Every step's prompt starts with it.
STAGED_PROMPT = """You write a plugin tool for Ogun, a local assistant (Windows, Python 3.12).

TOOL (saved as plugins/custom_tools/<name>.py)
- `from ogun_core.base_tool import BaseTool, ToolResult`; one class inheriting BaseTool with class attributes `name` (snake_case str), `version` = "1.0.0", `description`, `parameters_schema` (JSON Schema dict).
- `def execute(self, **kwargs) -> ToolResult`; read parameters with kwargs.get(). Return `ToolResult.success(output=...)` or `ToolResult.error(error_trace="...")`; never raise.
- Real standard-library logic, no stubs or TODOs. An empty string is a value: test parameters with `is None`, never `if not value`.
- Stream large files in chunks; catch OSError and return it as an error result.

TEST (saved as plugins/tests/test_<name>.py, run as a script, no pytest)
- First lines: `import sys`, `from pathlib import Path`, `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`; only then `from plugins.custom_tools.<name> import <the class the tool defines>`.
- Cases as (kwargs, expected) pairs checked in one loop. For a case that must fail, check only `res.status == "error"`, never the message text. Use only expected values the request states or that you compute with code; never hand-calculate checksums, hashes or other numbers.
- Check every example value the request quotes, valid and invalid, each as its own case.
- Every assertion message names the input and the actual result. Print "PASS" and exit 0; on the first failure print the message to stderr and exit 1.

- Use this loop (fill in the cases; "error" marks a case that must fail):
  ```
  def same(a, b):        # numbers within rounding error count as equal (-459.67 vs -459.66999999999996)
      if isinstance(a, dict) and isinstance(b, dict):
          return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
      if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
          return abs(a - b) <= 1e-6 * max(1.0, abs(b))
      return a == b
  CASES = [({"param": "valid input"}, <expected output>), ({"param": "bad input"}, "error")]
  for kwargs, expected in CASES:
      res = tool.execute(**kwargs)
      ok = res.status == "error" if expected == "error" else (res.status == "success" and same(res.output, expected))
      if not ok:
          print(f"FAIL execute({kwargs}): expected {expected!r}, got {res.status} {res.output!r} {res.error_trace!r}",
                file=sys.stderr)
          sys.exit(1)
  print("PASS")
  ```
  `<expected output>` must be exactly what the tool returns as `output` (if it returns {"result": x}, expect {"result": x}).
  Write expected numbers as Python expressions from the inputs (e.g. {"bmi": 90 / 1.75 ** 2}), never as a number you worked out yourself: hand-worked numbers were wrong in live builds (a BMI of 28.57 for 90 kg, 1.75 m).
"""


def plugin_name_from_code(code: str) -> Optional[str]:
    """`name = "x"` or `name: str = "x"` in the tool class (live: the short prompt led to the unannotated form, which the
    old pattern missed, and the plugin was mounted as 'custom_tool')."""
    m = re.search(r"""^\s*name\s*(?::\s*str\s*)?=\s*["']([A-Za-z0-9_]+)["']""", code or "", re.M)
    return m.group(1) if m else None


def requested_plugin_name(request_text: str) -> Optional[str]:
    """'plugin named roman_numeral' / 'plugin called roman_numeral' -> 'roman_numeral' (the user's own name wins)."""
    m = re.search(r"\b(?:plugin|tool)\s+(?:named|called)\s+[`'\"]?([A-Za-z][A-Za-z0-9_]*)", request_text or "")
    return m.group(1) if m else None


def generate(engine: Any, prompt: str, temperature: float = BASE_TEMPERATURE, fmt: Optional[Any] = None,
             num_predict: Optional[int] = None) -> str:
    """One fresh model call: the prompt alone, no chat history, no tool schemas.

    Live 2026-11-09: every build step went through run_turn (system prompt, every tool schema, the honesty check and
    possible tool rounds) although the builder only needs two code blocks back. Test doubles without a real engine
    still go through run_turn.
    """
    client = getattr(engine, "client", None)
    if isinstance(engine, CoreEngine) and client is not None and hasattr(client, "chat"):
        msgs = [{"role": "user", "content": prompt}]
        from ogun_core.kit.sampling import top_p_for
        opts = {"num_predict": num_predict or STAGE_MAX_TOKENS, "temperature": temperature,
                "top_p": top_p_for(temperature)}
        try:
            r = (client.chat(msgs, tools=None, options=opts, format=fmt) if fmt is not None
                 else client.chat(msgs, tools=None, options=opts))
        except TypeError:
            r = client.chat(msgs, tools=None)
        return (r or {}).get("content", "") or ""
    return engine.run_turn(prompt)


def failing_check_context(test_code: str, error: str, max_spots: int = 3) -> str:
    """Show the lines of the test that produced the failure, so the model sees WHICH input failed.

    Live bug: a roman_numeral build failed 93 times with only "TEST ASSERTION FAILED: Expected error, got success".
    The model never saw which input that was (the tool accepted 'IIII' and 'VX'), so it resubmitted the same code.
    """
    if not test_code or not error:
        return ""
    lines = test_code.splitlines()
    spots: List[int] = []
    # 1. Traceback line numbers inside the test file
    for m in re.finditer(r'File "[^"]*test_[^"]*\.py", line (\d+)', error):
        spots.append(int(m.group(1)) - 1)
    # 2. The assertion message text, matched back to the test source
    for raw in [l.strip() for l in error.splitlines() if l.strip()][-8:]:
        msg = re.sub(r"^.*?(?:AssertionError|ASSERTION FAILED|FAILED|FAIL)\s*:?\s*", "", raw).strip()
        for size in (40, 30, 20, 12):
            frag = msg[:size].strip()
            if len(frag) < 8:
                break
            hits = [i for i, l in enumerate(lines) if frag in l]
            if hits:
                spots.extend(hits)
                break
    seen, blocks = set(), []
    for i in spots:
        if not 0 <= i < len(lines) or i in seen:
            continue
        seen.add(i)
        lo, hi = max(0, i - 8), min(len(lines), i + 2)
        blocks.append("\n".join(f"{n + 1:4d}{'>>' if n == i else '  '} {lines[n]}" for n in range(lo, hi)))
        if len(blocks) >= max_spots:
            break
    if not blocks:
        return ""
    return ("THE FAILING CHECK IN THE TEST (>> marks the line that failed; read the input it uses):\n"
            + "\n...\n".join(blocks))


_QUOTED_RE = re.compile(r"(?<![\w'\"`])(?:'([^'\n]{1,40})'|\"([^\"\n]{1,40})\"|`([^`\n]{1,40})`)(?![\w'\"`])")


_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


def _is_code_name(value: str, tool_code: str) -> bool:
    """A quoted snake_case word that the tool's own code uses is a parameter or option name, not a test input.

    Live: "Parameters: 'value' (string) and 'direction' ('to_roman' or 'to_int')" made the check demand a test for
    'value' and 'direction', and a whole attempt was lost. Upper-case inputs like 'IIII' or 'VX' are never skipped.
    """
    return bool(tool_code) and bool(_NAME_RE.fullmatch(value)) and value in tool_code


def request_examples_missing(request_text: str, test_code: str, tool_code: str = "") -> List[str]:
    """Quoted example values in the request that the test never uses.

    Live bug: the request said invalid numerals "like 'IIII' or 'VX'" must be errors; the model's test only tried
    'IIII', so a tool that turned 'VX' into 5 passed and was mounted. For an update only the requested change counts.
    """
    text = request_text or ""
    if "=== REQUESTED CHANGE ===" in text:
        text = text.split("=== REQUESTED CHANGE ===", 1)[1].split("=== END ===", 1)[0]
    missing: List[str] = []
    for m in _QUOTED_RE.finditer(text):
        value = next(g for g in m.groups() if g is not None)
        if (value.strip() and value not in (test_code or "") and value not in missing
                and not _is_code_name(value, tool_code)):
            missing.append(value)
    return missing


def _note_failure(errors_seen: List[str], attempt: int, of_total: str, error: str) -> None:
    """Record a failed attempt and say why on the console (live: 93 attempts printed nothing but 'Researching...')."""
    errors_seen.append(error)
    first = " ".join((error or "").split())
    print(f"[Ogun Plugin Builder] Attempt {attempt}{of_total} failed: {first[:300]}", flush=True)


def build_plugin_staged(*args: Any, **kwargs: Any) -> Tuple[bool, str, Optional[str]]:
    """Build or update a plugin while holding the shared model lease (see _build_plugin_staged for the loop)."""
    from ogun_core.kit.gpu_lease import acquire
    lease = acquire("ogun", "plugin build")
    try:
        return _build_plugin_staged(*args, **kwargs)
    finally:
        if lease:
            lease.release()


def _build_plugin_staged(
    engine: CoreEngine,
    plugin_manager: PluginManager,
    request_text: str,
    max_attempts: Optional[int] = 3,
    force_name: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Build, test, verify, and register a dynamic plugin in stages with self-correction.

    max_attempts <= 0 (or None) means no limit (see module docstring).

    Returns:
        (success: bool, final_message: str, plugin_name: Optional[str])
    """
    unbounded = not max_attempts or max_attempts <= 0
    of_total = "" if unbounded else f"/{max_attempts}"
    last_error = "Unknown error"
    tool_code = ""
    test_code = ""
    plugin_name = None
    have_stage1 = False
    errors_seen: List[str] = []
    research_notes = ""
    restarted = False
    last_submitted: List[Tuple[str, str]] = []   # the last run's code, as written and as normalized
    skipped_same = False
    fixes_on_draft = 0
    suspect_test = False
    arbitrated: dict = {}
    case_fails: dict = {}
    dropped_cases: List[str] = []
    waived: set = set()
    example_misses: dict = {}
    trace = _Trace()

    if unbounded:
        print("[Ogun Plugin Builder] No attempt limit: fixing until the plugin passes its test. Ctrl+C stops it.",
              flush=True)

    # Every prompt carries what it needs (request, current code, failure), so the chat history is reset before each
    # attempt. Live bug: 93 attempts piled up in the history and the start of each prompt fell out of the context.
    base_messages = list(getattr(engine, "messages", []) or [])

    def _fresh_history() -> None:
        if hasattr(engine, "messages"):
            engine.messages = list(base_messages)

    attempt = 0
    try:
        while unbounded or attempt < max_attempts:
            attempt += 1
            trace.start(attempt)
            # A draft that still fails after MAX_FIXES_PER_DRAFT fixes is replaced by a fresh one (see the constant).
            if have_stage1 and fixes_on_draft >= MAX_FIXES_PER_DRAFT:
                print(f"[Ogun Plugin Builder] {MAX_FIXES_PER_DRAFT} fixes did not make this draft pass: "
                      "starting the plugin over from scratch.", flush=True)
                have_stage1, tool_code, test_code = False, "", ""
                restarted = True
            # No limit: research the first time an error repeats. (Every third attempt as well cost ~70 s per build in
            # the 2026-11-10 timing runs without a measurable gain; fresh drafts now do that job.)
            first_repeat = (len(errors_seen) >= 2 and errors_seen[-1] == errors_seen[-2]
                            and (len(errors_seen) < 3 or errors_seen[-3] != errors_seen[-1]))
            if unbounded and attempt > 1 and first_repeat:
                print(f"[Ogun Plugin Builder] Researching the failure before attempt {attempt}...", flush=True)
                research_notes = _research_plugin_failure(engine, request_text, last_error, tool_code) or research_notes
                trace.mark("research")
            streak = same_failure_streak(errors_seen) if have_stage1 else 0
            temperature = (retry_temperature(streak) if have_stage1
                           else (FRESH_DRAFT_TEMPERATURE if restarted else BASE_TEMPERATURE))
            dropped_now = False
            case = disputed_case(last_error) if (have_stage1 and test_code) else None
            if (case and case_fails.get(case, 0) >= DROP_DISPUTED_AFTER
                    and not case_quoted_in_request(case[0], request_text)):
                slimmer = drop_case_line(test_code, case[0])
                if slimmer != test_code:
                    test_code, dropped_now = slimmer, True
                    dropped_cases.append(case[0])
                    print(f"[Ogun Plugin Builder] The test case execute({case[0][:80]}) failed "
                          f"{case_fails[case]} test runs: dropping that one case "
                          "and testing again.", flush=True)
            try:
                if dropped_now:
                    pass
                elif not have_stage1:
                    fixes_on_draft = 0
                    # Stage 1: the tool and its test in one reply.
                    stage1_prompt = (
                        f"{STAGED_PROMPT}\n"
                        f"Stage 1/2: Author the complete Python tool class AND its unit test for this plugin request:\n"
                        f"{request_text}\n"
                        "Output two fenced ```python code blocks: the tool, then the test."
                    )
                    if research_notes:
                        stage1_prompt += f"\n\n{research_notes}"
                    _fresh_history()
                    reply1 = generate(engine, stage1_prompt, temperature)
                    blocks1 = parse_code_blocks(reply1)

                    if len(blocks1) >= 2:
                        tool_code, test_code = blocks1[0], blocks1[1]
                        have_stage1 = True
                    elif len(blocks1) == 1:
                        tool_code = blocks1[0]
                        have_stage1 = True
                        # Stage 2: the test for the tool just written (the prompt carries the tool; no history).
                        stage2_prompt = (
                            f"{STAGED_PROMPT}\n"
                            f"Stage 2/2: Author the standalone unit test for this tool.\n"
                            f"=== PLUGIN REQUEST ===\n{request_text}\n"
                            f"=== TOOL CODE ===\n```python\n{tool_code}\n```\n"
                            "Requirements: Standalone runnable via `python plugins/tests/test_<name>.py` "
                            "(imports the tool, asserts real behavior, exits 0 on success).\n"
                            "Provide the test code in a single fenced ```python code block."
                        )
                        _fresh_history()
                        blocks2 = parse_code_blocks(generate(engine, stage2_prompt, temperature))
                        if blocks2:
                            test_code = blocks2[0]
                    else:
                        last_error = "Model generated no fenced python code blocks in Stage 1."
                        _note_failure(errors_seen, attempt, of_total, last_error)
                        trace.end("no_blocks", last_error, tool_code, test_code)
                        continue
                else:
                    fixes_on_draft += 1
                    # Self-correction: Feed real failure/error output back into next attempt
                    where = failing_check_context(test_code, last_error)
                    code_where = tool_code_error_context(tool_code, last_error)
                    if code_where:
                        where = (where + "\n" if where else "") + code_where
                    case = disputed_case(last_error) if (streak >= 2 or suspect_test) else None
                    if case and case not in arbitrated:
                        print(f"[Ogun Plugin Builder] Checking the disputed case execute({case[0][:80]}) on its own...",
                              flush=True)
                        arbitrated[case] = arbitrate_case(
                            lambda q, schema=None: generate(engine, q, 0.0, fmt=schema, num_predict=600),
                            request_text, case)
                        if arbitrated[case]:
                            fixes_on_draft = min(fixes_on_draft, MAX_FIXES_PER_DRAFT - 1)   # let this fix use it
                    verdict = arbitrated.get(case, "") if case else ""
                    correction_prompt = (
                        f"{STAGED_PROMPT}\n"
                        f"Attempt {attempt}{of_total} - this plugin FAILED verification.\n"
                        f"=== PLUGIN REQUEST ===\n{request_text}\n"
                        f"=== CURRENT TOOL CODE ===\n```python\n{tool_code}\n```\n"
                        f"=== CURRENT TEST CODE ===\n```python\n{test_code}\n```\n"
                        f"=== FAILURE ===\n{fit_feedback(last_error, 2500)}\n"
                        + (f"{where}\n" if where else "")
                        + (f"{verdict}\n" if verdict else (f"{TEST_SUSPECT_NOTE}\n" if suspect_test
                                                            else (f"{STUCK_NOTE}\n" if streak >= 2 else "")))
                        + (f"{fit_feedback(research_notes, 1500)}\n" if research_notes else "")
                        + "=== HOW TO FIX ===\n"
                        "Find the failing check above and the input it uses. If that check matches the plugin "
                        "request, the TOOL is wrong: change the tool so it handles that input. Only change the test "
                        "if it contradicts the request. Resubmitting the same tool code will fail the same way.\n"
                        "Output the corrected tool in a fenced ```python block. If the test must change too, add the "
                        "complete corrected unit test as a second block (every assertion message names the input and "
                        "the actual result); if the test is right, output only the tool."
                    )
                    _fresh_history()
                    fix_reply = generate(engine, correction_prompt, temperature)
                    blocks = parse_code_blocks(fix_reply)
                    if len(blocks) >= 2:
                        tool_code, test_code = blocks[0], blocks[1]
                    elif len(blocks) == 1:
                        if is_tool_block(blocks[0]) or not is_tool_block(tool_code):
                            tool_code = blocks[0]
                        else:
                            test_code = blocks[0]
            except TimeoutError as exc:
                if not unbounded:
                    raise
                last_error = f"Model request timed out: {exc}"
                _note_failure(errors_seen, attempt, of_total, last_error)
                trace.end("timeout", last_error, tool_code, test_code)
                continue
            trace.mark("generate")

            if not tool_code or not test_code:
                last_error = (
                    f"Expected 2 code blocks (tool and test), got tool={bool(tool_code)}, test={bool(test_code)}."
                )
                _note_failure(errors_seen, attempt, of_total, last_error)
                trace.end("missing_block", last_error, tool_code, test_code)
                continue

            # The same code again cannot pass: count it as the same failure (raising the temperature) without a run.
            # Only once in a row, so a model that keeps repeating itself still gets its code run (and a fake can't loop).
            if (tool_code, test_code) in last_submitted and not skipped_same:
                skipped_same = True
                if suspect_test:                       # already asked to check the test: write a fresh draft
                    fixes_on_draft = MAX_FIXES_PER_DRAFT
                else:                                  # one more fix, told that the test is the likely culprit
                    suspect_test = True
                    fixes_on_draft = min(fixes_on_draft, MAX_FIXES_PER_DRAFT - 1)
                errors_seen.append(last_error)
                print(f"[Ogun Plugin Builder] Attempt {attempt}{of_total} returned the same code again; "
                      "retrying with more variation.", flush=True)
                trace.end("same_code", last_error, tool_code, test_code)
                continue

            # The test must try every example the request quotes (they are the user's own acceptance cases).
            untested = [v for v in request_examples_missing(request_text, test_code, tool_code) if v not in waived]
            if untested:
                key = tuple(untested)
                example_misses[key] = example_misses.get(key, 0) + 1
                if example_misses[key] >= EXAMPLE_WAIVE_AFTER:
                    waived.update(untested)
                    print(f"[Ogun Plugin Builder] The test left out {', '.join(repr(v) for v in untested)} "
                          f"{EXAMPLE_WAIVE_AFTER} times: treating that as wording in the request, not a test value.",
                          flush=True)
                    untested = []
            if untested:
                last_error = ("The unit test never checks these examples from the plugin request: "
                              + ", ".join(repr(v) for v in untested)
                              + ". Add a check for EACH one with the result the request gives for it (a value it "
                              "converts to, or an error result), then make the tool pass those checks.")
                _note_failure(errors_seen, attempt, of_total, last_error)
                trace.end("examples", last_error, tool_code, test_code)
                continue

            # Static requirements-coverage check
            covered, missing = check_requirements_coverage(request_text, tool_code)
            if not covered:
                last_error = f"Static requirements coverage check failed: {'; '.join(missing)}"
                _note_failure(errors_seen, attempt, of_total, last_error)
                trace.end("coverage", last_error, tool_code, test_code)
                continue

            # Plugin name: an update keeps the existing name; otherwise the name the request gives, then the code's.
            plugin_name = (force_name or requested_plugin_name(request_text) or plugin_name_from_code(tool_code)
                           or "custom_tool")

            # Mechanical test mistakes (wrong module or class name in the import, imports above the path setup) cost
            # whole attempts in the measurement above; fix them instead of asking the model.
            if last_submitted and test_code != last_submitted[0][1]:
                suspect_test = False             # the test was changed: judge it afresh
            written = (tool_code, test_code)
            test_code = normalize_plugin_test(test_code, tool_code, plugin_name)

            # Attempt creation and isolated subprocess testing
            last_submitted, skipped_same = [written, (tool_code, test_code)], False
            res = plugin_manager.create_or_update_plugin(plugin_name, tool_code, test_code)
            trace.mark("install_and_test")
            if res.status == "success":
                trace.end("pass", "", tool_code, test_code)
                note = (" Dropped test case(s) the model kept disputing: "
                        + "; ".join(f"execute({c[:80]})" for c in dropped_cases) + ".") if dropped_cases else ""
                return (
                    True,
                    f"[PASS] Plugin '{plugin_name}' successfully built, tested, and mounted on attempt {attempt}.{note}",
                    plugin_name,
                )
            last_error = res.error_trace or "Subprocess unit test execution failed."
            _failed_case = disputed_case(last_error)
            if _failed_case:
                case_fails[_failed_case] = case_fails.get(_failed_case, 0) + 1
            _note_failure(errors_seen, attempt, of_total, last_error)
            trace.end("test_failed", last_error, tool_code, test_code)
    except KeyboardInterrupt:
        return (
            False,
            f"[STOPPED] Plugin build stopped by the user after {attempt} attempt(s). Last error: {last_error}",
            plugin_name,
        )

    return (
        False,
        f"[FAIL] Plugin creation failed after {max_attempts} attempts. Last error: {last_error}",
        plugin_name,
    )


class _Trace:
    """Per-attempt timings and full failures as JSON lines, when OGUN_BUILD_TRACE names a file (off by default)."""

    def __init__(self) -> None:
        import os
        self.path = os.environ.get("OGUN_BUILD_TRACE", "").strip()
        self.rec: dict = {}

    def start(self, attempt: int) -> None:
        import time
        self.t = time.time()
        self.rec = {"attempt": attempt, "phases": {}}

    def mark(self, phase: str) -> None:
        import time
        now = time.time()
        self.rec.setdefault("phases", {})[phase] = round(now - self.t, 2)
        self.t = now

    def end(self, outcome: str, error: str, tool_code: str, test_code: str) -> None:
        if not self.path:
            return
        import json
        self.rec.update(outcome=outcome, error=error, tool_code=tool_code, test_code=test_code)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.rec) + "\n")
        except OSError:
            pass
