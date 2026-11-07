"""Ogun dynamic plugin staged builder and self-correction loop.

Coordinates staged code generation (tool implementation, then unit tests) and iterative self-correction. With
max_attempts <= 0 there is no attempt limit: the loop researches failures that keep coming back and keeps going
until the plugin passes its test or the user presses Ctrl+C.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from ogun_core.config import PLUGIN_BUILDER_PROMPT
from ogun_core.engine import CoreEngine
from ogun_core.plugin_manager import PluginManager
from ogun_core.quality import check_requirements_coverage, fit_feedback, research_build_failure


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


RESTART_AFTER_SAME_FAILURES = 5


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


def request_examples_missing(request_text: str, test_code: str) -> List[str]:
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
        if value.strip() and value not in (test_code or "") and value not in missing:
            missing.append(value)
    return missing


def _note_failure(errors_seen: List[str], attempt: int, of_total: str, error: str) -> None:
    """Record a failed attempt and say why on the console (live: 93 attempts printed nothing but 'Researching...')."""
    errors_seen.append(error)
    first = " ".join((error or "").split())
    print(f"[Ogun Plugin Builder] Attempt {attempt}{of_total} failed: {first[:300]}", flush=True)


def build_plugin_staged(
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
            # Same failure RESTART_AFTER_SAME_FAILURES times in a row: the code is stuck, write it again from scratch.
            tail = errors_seen[-RESTART_AFTER_SAME_FAILURES:]
            if (have_stage1 and len(tail) == RESTART_AFTER_SAME_FAILURES and len(set(tail)) == 1
                    and (attempt - 1) % RESTART_AFTER_SAME_FAILURES == 0):
                print(f"[Ogun Plugin Builder] Same failure {RESTART_AFTER_SAME_FAILURES} times in a row: "
                      "starting the plugin over from scratch.", flush=True)
                have_stage1, tool_code, test_code = False, "", ""
            # No limit: research the first time an error repeats, and every third attempt.
            first_repeat = (len(errors_seen) >= 2 and errors_seen[-1] == errors_seen[-2]
                            and (len(errors_seen) < 3 or errors_seen[-3] != errors_seen[-1]))
            if unbounded and attempt > 1 and (first_repeat or (attempt - 1) % 3 == 0):
                print(f"[Ogun Plugin Builder] Researching the failure before attempt {attempt}...", flush=True)
                research_notes = _research_plugin_failure(engine, request_text, last_error, tool_code) or research_notes
            try:
                if not have_stage1:
                    # Stage 1: Request tool implementation
                    stage1_prompt = (
                        f"{PLUGIN_BUILDER_PROMPT}\n\n"
                        f"Stage 1/2: Author the complete Python tool class for this plugin request:\n{request_text}\n"
                        "Provide the tool code in a fenced ```python code block."
                    )
                    if research_notes:
                        stage1_prompt += f"\n\n{research_notes}"
                    _fresh_history()
                    reply1 = engine.run_turn(stage1_prompt)
                    blocks1 = parse_code_blocks(reply1)

                    if len(blocks1) >= 2:
                        # Model provided both tool and test in stage 1
                        tool_code, test_code = blocks1[0], blocks1[1]
                        have_stage1 = True
                    elif len(blocks1) == 1:
                        tool_code = blocks1[0]
                        have_stage1 = True
                        # Stage 2: Request unit tests
                        stage2_prompt = (
                            "Stage 2/2: Author the standalone unit test for the tool class above.\n"
                            "Requirements: Standalone runnable via `python plugins/tests/test_<name>.py` "
                            "(imports the tool, asserts real behavior, exits 0 on success).\n"
                            "Provide the test code in a single fenced ```python code block."
                        )
                        reply2 = engine.run_turn(stage2_prompt)
                        blocks2 = parse_code_blocks(reply2)
                        if blocks2:
                            test_code = blocks2[0]
                    else:
                        last_error = "Model generated no fenced python code blocks in Stage 1."
                        _note_failure(errors_seen, attempt, of_total, last_error)
                        continue
                else:
                    # Self-correction: Feed real failure/error output back into next attempt
                    where = failing_check_context(test_code, last_error)
                    correction_prompt = (
                        f"{PLUGIN_BUILDER_PROMPT}\n\n"
                        f"Attempt {attempt}{of_total} - this plugin FAILED verification.\n"
                        f"=== PLUGIN REQUEST ===\n{request_text}\n"
                        f"=== CURRENT TOOL CODE ===\n```python\n{tool_code}\n```\n"
                        f"=== CURRENT TEST CODE ===\n```python\n{test_code}\n```\n"
                        f"=== FAILURE ===\n{fit_feedback(last_error, 2500)}\n"
                        + (f"{where}\n" if where else "")
                        + (f"{fit_feedback(research_notes, 1500)}\n" if research_notes else "")
                        + "=== HOW TO FIX ===\n"
                        "Find the failing check above and the input it uses. If that check matches the plugin "
                        "request, the TOOL is wrong: change the tool so it handles that input. Only change the test "
                        "if it contradicts the request. Resubmitting the same tool code will fail the same way.\n"
                        "Output exactly two fenced ```python code blocks:\n"
                        "1. The corrected tool implementation.\n"
                        "2. The corrected unit test (every assertion message names the input and the actual result)."
                    )
                    _fresh_history()
                    fix_reply = engine.run_turn(correction_prompt)
                    blocks = parse_code_blocks(fix_reply)
                    if len(blocks) >= 2:
                        tool_code, test_code = blocks[0], blocks[1]
                    elif len(blocks) == 1:
                        tool_code = blocks[0]
            except TimeoutError as exc:
                if not unbounded:
                    raise
                last_error = f"Model request timed out: {exc}"
                _note_failure(errors_seen, attempt, of_total, last_error)
                continue

            if not tool_code or not test_code:
                last_error = (
                    f"Expected 2 code blocks (tool and test), got tool={bool(tool_code)}, test={bool(test_code)}."
                )
                _note_failure(errors_seen, attempt, of_total, last_error)
                continue

            # The test must try every example the request quotes (they are the user's own acceptance cases).
            untested = request_examples_missing(request_text, test_code)
            if untested:
                last_error = ("The unit test never checks these examples from the plugin request: "
                              + ", ".join(repr(v) for v in untested)
                              + ". Add a check for EACH one with the result the request gives for it (a value it "
                              "converts to, or an error result), then make the tool pass those checks.")
                _note_failure(errors_seen, attempt, of_total, last_error)
                continue

            # Static requirements-coverage check
            covered, missing = check_requirements_coverage(request_text, tool_code)
            if not covered:
                last_error = f"Static requirements coverage check failed: {'; '.join(missing)}"
                _note_failure(errors_seen, attempt, of_total, last_error)
                continue

            # Extract plugin name
            name_match = re.search(r'name\s*:\s*str\s*=\s*["\']([^"\']+)["\']', tool_code)
            plugin_name = name_match.group(1) if name_match else "custom_tool"
            if force_name:
                plugin_name = force_name   # an update must replace the existing plugin, never create a differently named one

            # Attempt creation and isolated subprocess testing
            res = plugin_manager.create_or_update_plugin(plugin_name, tool_code, test_code)
            if res.status == "success":
                return (
                    True,
                    f"[PASS] Plugin '{plugin_name}' successfully built, tested, and mounted on attempt {attempt}.",
                    plugin_name,
                )
            last_error = res.error_trace or "Subprocess unit test execution failed."
            _note_failure(errors_seen, attempt, of_total, last_error)
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
