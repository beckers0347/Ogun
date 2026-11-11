"""Plugin-builder helpers shared by every app that builds plugins with a local model (vendored copy; edit the
canonical shared_kit and re-sync, never the vendored files).

Each helper exists because of a measured live failure; the notes stay with the code.
"""
from __future__ import annotations

import ast
import re
from typing import Callable, List, Optional, Tuple

from .sampling import PROFILES, profile
from .structured import parse_json

BASE_TEMPERATURE = PROFILES["code"]["temperature"]        # first draft (sampling.code)

FIX_TEMPERATURE = PROFILES["code_fix"]["temperature"]

FRESH_DRAFT_TEMPERATURE = PROFILES["code_redraft"]["temperature"]

# Measured 2026-11-09 on the Roman-numeral request (27 sampled fix replies, 25 sampled first drafts): fixes of a draft
# whose logic is wrong almost never pass (0/27), while fresh drafts pass about one time in five. So a draft gets at most
# MAX_FIXES_PER_DRAFT fixes, then a fresh draft is written at a higher temperature so it differs from the last one.
MAX_FIXES_PER_DRAFT = 2

# Quoted words the model leaves out of its test this many times are taken as wording, not test values. Live
# 2026-11-10: a request saying `a list of "bad" patterns` failed 29 attempts in a row on "never checks 'bad'".
EXAMPLE_WAIVE_AFTER = 3

# A model-written test case that fails this many test runs is dropped (never a value quoted in the
# request). Live 2026-11-10: every draft's test expected the mistyped 'mmmcmmxcix' to be 3999; the tool rightly
# rejected it, and the update ran into the 15-minute cap on that one case.
DROP_DISPUTED_AFTER = 4

# Live 2026-11-10: a lowercase update failed 10 attempts on its own typo ('mmmcmmxcix' expected to be 3999)
# while every other fix returned the same, correct tool.
STUCK_NOTE = ("The previous fix produced this SAME failure again, so the current approach to this case does not work. "
              "Do not patch it: rewrite the logic that decides this case with a different method.")

TEST_SUSPECT_NOTE = ("Your last fix returned the tool unchanged, so you consider the tool correct. Then the failing "
                     "TEST CASE is probably wrong: re-check its input and its expected value against the request (a "
                     "mistyped value or a miscalculated result) and output the corrected test as the second block.")

_BOOT = ("import sys as _plugin_test_sys\nfrom pathlib import Path as _PluginTestPath\n"
         "_plugin_test_sys.path.insert(0, str(_PluginTestPath(__file__).resolve().parents[2]))\n")

_FAIL_LINE_RE = re.compile(r"FAIL execute\((.{0,300}?)\): expected (.{0,300}?), got (success|error) ?(.{0,300})")

def _tool_class(tool_code: str):
    try:
        tree = ast.parse(tool_code)
    except SyntaxError:
        m = re.search(r"^class\s+(\w+)\s*\([^)]*BaseTool", tool_code or "", re.M)
        return m.group(1) if m else None, set()
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    names |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and any("BaseTool" in ast.unparse(b) for b in n.bases):
            return n.name, names
    return None, names

def normalize_plugin_test(test_code: str, tool_code: str, module_name: str) -> str:
    if not test_code or not module_name:
        return test_code
    cls, defined = _tool_class(tool_code)

    def fix_from(m):
        names = m.group(2)
        if cls:
            parts = [p.strip() for p in names.split(",")]
            fixed = []
            for p in parts:
                base, _, alias = p.partition(" as ")
                base = base.strip()
                if base and base not in defined and base[:1].isupper():
                    alias = alias.strip() or base      # keep the name the test uses
                    p = f"{cls} as {alias}" if alias != cls else cls
                fixed.append(p)
            names = ", ".join(fixed)
        return f"from plugins.custom_tools.{module_name} import {names}"

    # Live: `from plugins.tools.roman_numeral import ...` failed 13 attempts in a row. Any plugins.* / custom_tools.*
    # path, or the bare module name, means the tool module.
    out = re.sub(r"from\s+(plugins(?:\.\w+)*|custom_tools(?:\.\w+)*|" + re.escape(module_name)
                 + r")\s+import\s+([^\n(]+)", fix_from, test_code)
    out = re.sub(r"import\s+plugins\.custom_tools\.(\w+)", f"import plugins.custom_tools.{module_name}", out)
    out = re.sub(r"plugins\.custom_tools\.(?!" + re.escape(module_name) + r"\b)(\w+)\.(\w+)",
                 lambda m: f"plugins.custom_tools.{module_name}.{m.group(2)}", out)
    if "_plugin_test_sys" not in out:
        lines = out.splitlines(keepends=True)
        at = 0
        # keep a module docstring and `from __future__` imports first
        try:
            tree = ast.parse(out)
            for n in tree.body:
                if (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)
                        and isinstance(n.value.value, str)) or (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
                    at = n.end_lineno
                else:
                    break
        except SyntaxError:
            pass
        out = "".join(lines[:at]) + _BOOT + "".join(lines[at:])
    return out

def same_failure_streak(errors: List[str]) -> int:
    """How many of the latest failures in a row are the same failure (0 when there is none)."""
    n = 0
    for e in reversed(errors):
        if e != errors[-1]:
            break
        n += 1
    return n

def retry_temperature(streak: int) -> float:
    """Low temperature first; each repeat of the same failure samples more widely so the next fix differs."""
    return profile("code_fix", streak)["temperature"]

def is_tool_block(code: str) -> bool:
    """A fix reply with one block: is it the tool (defines the BaseTool class) or the test?"""
    return bool(re.search(r"^\s*class\s+\w+\s*\([^)]*BaseTool", code or "", re.M))

def tool_code_error_context(tool_code: str, error: str) -> str:
    """The tool-code lines a syntax error or traceback points at (>> marks the line).

    Live: "SyntaxError at line 17: invalid syntax" came back four attempts running; the fix prompt showed the test's
    lines but never line 17 of the tool, so the model could not see what to fix.
    """
    if not tool_code or not error:
        return ""
    lines = tool_code.splitlines()
    spots: List[int] = []
    if "test_" not in error.split("SyntaxError at line")[0][-200:]:
        spots += [int(n) - 1 for n in re.findall(r"SyntaxError at line (\d+)", error)]
    spots += [int(n) - 1 for n in re.findall(r'File "[^"]*custom_tools[^"]*\.py", line (\d+)', error)]
    # A NameError the tool caught itself has no line number ("Execution failed: name 'x' is not defined"). Live: four
    # attempts in a row could not find it; show where the name is used.
    undefined = re.findall(r"name '([A-Za-z_]\w*)' is not defined", error)
    for nm in undefined[:1]:
        spots += [i for i, ln in enumerate(lines) if re.search(rf"\b{re.escape(nm)}\b", ln)][:3]
    marked = sorted({i for i in spots if 0 <= i < len(lines)})[:3]
    windows: List[List[int]] = []          # overlapping windows merge into one listing with several >> marks
    for i in marked:
        lo, hi = max(0, i - 4), min(len(lines), i + 3)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    blocks = ["\n".join(f"{n + 1:4d}{'>>' if n in marked else '  '} {lines[n]}" for n in range(lo, hi))
              for lo, hi in windows]
    if not blocks:
        return ""
    head = "THE FAILING LINE IN THE TOOL CODE (>> marks it):\n"
    if undefined:
        head = (f"`{undefined[0]}` is used where it is not defined (a name defined inside one function or method is "
                f"not visible in another). Where `{undefined[0]}` appears in the tool code (>> marks it):\n")
    return head + "\n...\n".join(blocks)

def disputed_case(error: str) -> Optional[Tuple[str, str, str]]:
    """(input, expected by the test, what the tool gave) from the test loop's FAIL line, or None."""
    m = _FAIL_LINE_RE.search(error or "")
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), f"{m.group(3)} {m.group(4).strip()}".strip()

def case_quoted_in_request(given: str, request_text: str) -> bool:
    """True when a string value of the failing call is quoted in the request (the user's own example)."""
    try:
        kwargs = ast.literal_eval(given)
    except Exception:  # noqa: BLE001
        return True                          # cannot tell: keep the case
    values = kwargs.values() if isinstance(kwargs, dict) else []
    for v in values:
        if re.fullmatch(r"[a-z][a-z0-9]*_[a-z0-9_]+", str(v)):
            continue                         # an option name such as 'to_int', not an example value
        if isinstance(v, str) and v and re.search(r"[\'\"`]" + re.escape(v) + r"[\'\"`]", request_text or ""):
            return True
    return False

def drop_case_line(test_code: str, given: str) -> str:
    """The test without the one single-line case whose input is `given` (unchanged if it is not exactly one line)."""
    try:
        kwargs = ast.literal_eval(given)
    except Exception:  # noqa: BLE001
        return test_code
    if not isinstance(kwargs, dict) or not kwargs:
        return test_code

    def lits(v):
        return [repr(v), '"' + v + '"'] if isinstance(v, str) else [repr(v)]

    lines = test_code.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines)
            if ln.lstrip().startswith(("(", "[", "{")) and all(any(x in ln for x in lits(v)) for v in kwargs.values())]
    if len(hits) != 1:
        return test_code
    candidate = "".join(lines[:hits[0]] + lines[hits[0] + 1:])
    try:
        ast.parse(candidate)
    except SyntaxError:
        return test_code
    return candidate

ARBITRATION_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string", "enum": ["A", "B"]}},
    "required": ["reasoning", "answer"],
}


def _ask(ask: Callable, question: str) -> str:
    try:
        return ask(question, ARBITRATION_SCHEMA) or ""
    except TypeError:                     # a one-argument ask (no structured output available)
        return ask(question) or ""


def _answer(reply: str) -> str:
    data = parse_json(reply)
    if isinstance(data, dict) and data.get("answer") in ("A", "B"):
        return data["answer"]
    m = re.findall(r"ANSWER:\s*([AB])\b", (reply or "").upper())
    return m[-1] if m else "?"


def arbitrate_case(ask: Callable, request_text: str, case: Tuple[str, str, str]) -> str:
    """Ask the model ONE focused question: for this input, is the test's expectation or the tool's result right?

    `ask(prompt, schema)` returns the reply text; the schema makes the answer {"reasoning": ..., "answer": "A"|"B"}
    (Ollama `format`). Live 2026-11-10: every draft's test expected a BMI of 24.69 to be 'overweight' (it is 'normal');
    the tool was right and 21 attempts failed on the test's own mistake. A short question about just that case gets
    it right where the full fix prompt did not.

    The question is asked twice with the two options SWAPPED (verdict sampling is deterministic, so asking the same
    question twice would always agree); only a verdict that survives the swap counts. Only "the test is wrong" is
    acted on: a "the tool is wrong" verdict adds nothing (every fix prompt says so) and, live, a mistaken one made an
    update drop the very change the user asked for. Returns a note for the next fix prompt ('' otherwise).
    """
    given, expected, got = case
    if "=== REQUESTED CHANGE ===" in request_text:
        # An update: the change decides. Live 2026-11-10: with the old code in the question, both answers sided with
        # the old behaviour ('y' not a vowel) and the update shipped without the change the user asked for.
        change = request_text.split("=== REQUESTED CHANGE ===", 1)[1].split("=== END ===", 1)[0].strip()
        context = ("An existing plugin is being UPDATED. The user asked for this change, and the new behaviour must "
                   f"follow it wherever the old behaviour differs:\n{change[:1500]}\n\n")
    else:
        context = "A plugin was built for this request:\n" + request_text[-2500:] + "\n\n"

    def question(first: str, second: str) -> str:
        return (context + f"For the call execute({given}):\n"
                f"A) {first}\nB) {second}\n"
                "Which one does the request call for? Work it out step by step with the actual numbers or rules in "
                'a few short lines in "reasoning", then give "answer": "A" or "B".')

    test_side, tool_side = f"the unit test expects: {expected}", f"the tool returned: {got}"
    first = _answer(_ask(ask, question(test_side, tool_side)))            # B = the tool is right
    swapped = _answer(_ask(ask, question(tool_side, test_side)))          # A = the tool is right
    if not (first == "B" and swapped == "A"):
        return ""
    return (f"CHECKED SEPARATELY: for execute({given}) the TOOL's result ({got}) is correct and the test's expected "
            f"value ({expected}) is WRONG. Keep the tool. Fix that case in the test and output the complete "
            "corrected test as the second block.")

def plugin_shape_problems(code: str) -> list:
    """What a plugin lacks to be callable: a BaseTool subclass with a `name` string, a `parameters_schema` dict and an
    `execute` method that returns ToolResult. Empty list when the shape is right.

    Live 2026-11-10: chat-model tool calls installed plugins with none of these (a bare `def execute(self, miles)`
    returning a float), and the reply said the plugin was "successfully created, tested". Every install path now needs
    the shape; the staged builder's prompt already asks for exactly this.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code or "")
    except SyntaxError:
        return []                              # the syntax check reports this
    classes = [n for n in tree.body if isinstance(n, _ast.ClassDef)
               and any("BaseTool" in _ast.unparse(b) for b in n.bases)]
    if not classes:
        return ["no class that inherits from BaseTool"]
    cls = classes[0]
    assigned = {}
    for n in cls.body:
        if isinstance(n, _ast.Assign):
            for t in n.targets:
                if isinstance(t, _ast.Name):
                    assigned[t.id] = n.value
        elif isinstance(n, _ast.AnnAssign) and isinstance(n.target, _ast.Name) and n.value is not None:
            assigned[n.target.id] = n.value
    problems = []
    if not (isinstance(assigned.get("name"), _ast.Constant) and isinstance(assigned["name"].value, str)):
        problems.append("class attribute `name` (a snake_case string)")
    execute = next((n for n in cls.body if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                    and n.name == "execute"), None)
    # BaseTool supplies an empty schema, so only a tool that takes parameters must describe them.
    takes_params = execute is not None and (
        len(execute.args.args) > 1 or execute.args.kwonlyargs
        or re.search(r"\bkwargs\s*(?:\.get\(|\[)", _ast.unparse(execute)) is not None)
    if takes_params and not isinstance(assigned.get("parameters_schema"), (_ast.Dict, _ast.Call)):
        problems.append("class attribute `parameters_schema` (a JSON Schema dict describing the parameters)")
    if execute is None:
        problems.append("an `execute(self, **kwargs)` method")
    if "ToolResult" not in (code or ""):
        problems.append("results returned as ToolResult.success(output=...) / ToolResult.error(error_trace=...)")
    return problems
