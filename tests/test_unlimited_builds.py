"""Tool builds without an attempt limit: they research failures that keep coming back, keep going past the old limit
of 5 until they pass, and stop on Ctrl+C."""
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ogun_core.standalone_builder as sb  # noqa: E402
from ogun_core import plugin_builder as plb  # noqa: E402
from ogun_core import quality as q  # noqa: E402
from ogun_core.research import parse_results  # noqa: E402

PROMPT = (
    "Build me a command-line tool called `dupefinder`. Point it at a folder and it finds duplicate files by content "
    "(not by name), groups them, and reports how much disk space the duplicates waste. It needs a `--json` output "
    "option and a `--min-size` option to ignore small files. Write tests for it, run them, and make sure they pass."
)

TOOL = """
import argparse
import os
import sys
import hashlib
import json

def hash_file(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(folder, min_size=0):
    groups = {}
    return groups

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Find duplicate files.")
    parser.add_argument('folder', help="Target folder")
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--min-size', type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.folder):
        print(f"Error: {args.folder} does not exist", file=sys.stderr)
        sys.exit(1)
    wasted_space = 0
    print(f"Total wasted space: {wasted_space} bytes")

if __name__ == '__main__':
    main()
"""

FAILING_TEST = """
from _scaffold import ToolTestCase
import unittest
class TestDupe(ToolTestCase):
    def test_min_size(self):
        res = self.run_tool("--min-size", "0")
        self.assertEqual(res.returncode, 999)
if __name__ == '__main__':
    unittest.main()
"""

PASSING_TEST = """
from _scaffold import ToolTestCase
import unittest
class TestDupe(ToolTestCase):
    def test_help(self):
        res = self.run_tool("--help")
        self.assertEqual(res.returncode, 0)
if __name__ == '__main__':
    unittest.main()
"""


class FakeClient:
    """Tool + failing tests; the first `fix_failures` fixes repeat the failure, the next one fixes it.
    `call_history` marks it as a test double (no web search, no model auto-selection)."""

    def __init__(self, fix_failures, on_fix=None):
        self.fix_failures, self.on_fix = fix_failures, on_fix
        self.call_history = []
        self.fixes = 0
        self.research = 0
        self.fix_prompts = []

    def chat(self, messages, tools=None, options=None):
        text = messages[-1]["content"]
        self.call_history.append(text)
        if text.startswith(("A tool build keeps failing", "Write ONE short web search query")):
            self.research += 1
            return {"content": "ROOT CAUSE: test_min_size expects exit 999. CHANGES: assert --help exits 0."}
        if "Write comprehensive unit tests in" in text:
            return {"content": f"```python\n{FAILING_TEST}\n```"}
        if "CURRENT TOOL CODE" in text or "FAILING TEST TRACEBACKS" in text:
            self.fixes += 1
            self.fix_prompts.append(text)
            if self.on_fix:
                self.on_fix(self.fixes)
            good = self.fixes > self.fix_failures
            return {"content": f"```python\n{TOOL}\n```\n```python\n{PASSING_TEST if good else FAILING_TEST}\n```"}
        if "CORE ENGINEERING CONTRACT" in text:
            return {"content": f"```python\n{TOOL}\n```"}
        return {"content": "{}"}


def _in_temp_workspace(fn):
    td = Path(tempfile.mkdtemp())
    try:
        with patch.object(sb, "WORKSPACE_DIR", td / "workspace"):
            return fn(td / "workspace" / "dupefinder")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_search_results_are_parsed():
    page = ('<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2F'
            'docs.python.org%2F3%2Flibrary%2Fargparse.html&amp;rut=x">argparse &mdash; <b>Parser</b></a>'
            '<a class="result__snippet" href="#">Use <b>nargs</b> for lists.</a></div>')
    out = parse_results(page)
    assert "argparse \u2014 Parser" in out and "https://docs.python.org/3/library/argparse.html" in out, out
    assert "Use nargs for lists." in out


def test_research_uses_real_errors_and_survives_search_failures():
    calls = []

    def chat(text, n):
        calls.append(text)
        return "ROOT CAUSE: x" if "ROOT CAUSE" in text else "unused query"

    def search(query):
        if "KeyError" in query:
            raise OSError("offline")
        return "- result about " + query

    fb = "Traceback ...\nValueError: could not convert string to float: 'abc'\nKeyError: 'mean'"
    notes = q.research_build_failure(chat, "req", fb, "code", search)
    assert "ROOT CAUSE: x" in notes and notes.startswith("RESEARCH NOTES")
    assert len(calls) == 1 and "python ValueError" in calls[0]


def test_unlimited_build_keeps_going_past_five_attempts_and_researches():
    def run(project_dir):
        client = FakeClient(fix_failures=6)
        ok, msg, details = sb.build_standalone_project_staged(SimpleNamespace(client=client), PROMPT, max_attempts=0)
        assert ok, msg[:500]
        assert details.get("attempts", 0) >= 7, details.get("attempts")
        assert client.research >= 1, "a failure that kept coming back was never researched"
        assert any("RESEARCH NOTES" in p for p in client.fix_prompts), "research never reached the fix prompt"
        for _p in client.fix_prompts:
            # live bug: feedback piled up across attempts until the prompt overflowed the 8192-token context
            assert _p.count("RESEARCH NOTES (") <= 1 and _p.count("FAILING TESTS EXPLAINED") <= 1, "feedback piled up"
    _in_temp_workspace(run)


def test_bounded_build_still_stops_at_its_limit():
    def run(project_dir):
        client = FakeClient(fix_failures=99)
        ok, _, details = sb.build_standalone_project_staged(SimpleNamespace(client=client), PROMPT, max_attempts=3)
        assert not ok and details.get("attempts", 0) <= 3 and client.research == 0
    _in_temp_workspace(run)


def test_ctrl_c_stops_an_unlimited_build():
    def stop(n):
        if n == 2:
            raise KeyboardInterrupt

    def run(project_dir):
        client = FakeClient(fix_failures=99, on_fix=stop)
        ok, msg, details = sb.build_standalone_project_staged(SimpleNamespace(client=client), PROMPT, max_attempts=0)
        assert not ok and details.get("stopped") and "Stopped building" in msg, msg
        assert (project_dir / "dupefinder.py").exists(), "files stay on disk so a later request resumes"
    _in_temp_workspace(run)


def test_stopped_update_restores_the_previous_tool():
    td = Path(tempfile.mkdtemp())
    try:
        with patch.object(sb, "WORKSPACE_DIR", td):
            proj = td / "t1"
            proj.mkdir()
            (proj / "t1.py").write_text("ORIGINAL", encoding="utf-8")

            def stopped_build(engine, request, **kw):
                (proj / "t1.py").write_text("HALF-DONE", encoding="utf-8")
                return False, "Stopped building the tool", {"status": "stopped", "stopped": True}

            ok, msg, _ = sb.update_standalone_project(None, "t1", "Add a --top N option", build_fn=stopped_build,
                                                      max_attempts=0)
            assert not ok and "stopped, as you asked" in msg and "not an option" not in msg, msg
            assert (proj / "t1.py").read_text(encoding="utf-8") == "ORIGINAL"
    finally:
        shutil.rmtree(td, ignore_errors=True)


class _FakeEngine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.client = SimpleNamespace(call_history=[], chat=lambda m, tools=None, options=None: {"content": "ROOT CAUSE: y"})

    def run_turn(self, prompt):
        self.prompts.append(prompt)
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, BaseException):
            raise item
        return item


class _FakePM:
    def __init__(self, succeed_on):
        self.calls = 0
        self.succeed_on = succeed_on

    def create_or_update_plugin(self, name, tool, test):
        self.calls += 1
        ok = self.succeed_on and self.calls >= self.succeed_on
        return SimpleNamespace(status="success" if ok else "error", error_trace="" if ok else "AssertionError: 2 != 3")


PLUGIN_REPLY = "```python\nclass T:\n    name: str = \"adder\"\n```\n```python\nassert True\n```"


def test_unlimited_plugin_build_keeps_going_through_timeouts_and_researches():
    eng = _FakeEngine([PLUGIN_REPLY, TimeoutError("read timeout"), PLUGIN_REPLY])
    pm = _FakePM(succeed_on=5)
    ok, msg, name = plb.build_plugin_staged(eng, pm, "Build adder plugin", max_attempts=0)
    assert ok and name == "adder" and pm.calls == 5, (ok, msg, pm.calls)
    assert any("ROOT CAUSE: y" in p for p in eng.prompts), "the research notes never reached a fix prompt"


def test_ctrl_c_stops_an_unlimited_plugin_build():
    eng = _FakeEngine([PLUGIN_REPLY, PLUGIN_REPLY, KeyboardInterrupt()])
    ok, msg, _ = plb.build_plugin_staged(eng, _FakePM(succeed_on=0), "Build adder plugin", max_attempts=0)
    assert not ok and msg.startswith("[STOPPED]"), msg


LOGSUM = ("Build me a command-line tool called `logsum`. Give it a log file where each line starts with a level word "
          "such as INFO, WARNING or ERROR, and it prints how many lines there are of each level as an aligned "
          "plain-text table with a header row, sorted by count descending. It needs a `--json` option and a "
          "`--level NAME` option to show only that level.")


def test_log_file_tool_is_not_classified_as_csv():
    """Live bug: 'a table with a header row' made a log-level counter a CSV statistics tool, and the wrong CSV draft
    then kept it in CSV mode for 270 attempts."""
    assert q.infer_tool_input_kind(LOGSUM, {}, "", "logsum") != "csv"
    assert q.infer_tool_input_kind(LOGSUM, {}, "import csv\nreader = csv.reader(f)", "logsum") != "csv"
    assert "CSV STATISTICS" not in sb.build_tool_generation_prompt(LOGSUM, "logsum", ["--json", "--level"])
    csv_req = "Build me a tool called `colx`. Give it a CSV file with a header row; print the mean of each column."
    assert q.infer_tool_input_kind(csv_req, {}, "", "colx") == "csv"


def test_unlimited_build_starts_over_after_five_identical_failures():
    class Restarting(FakeClient):
        tool_calls = 0

        def chat(self, messages, tools=None, options=None):
            text = messages[-1]["content"]
            if "CORE ENGINEERING CONTRACT" in text and "CURRENT TOOL CODE" not in text \
                    and "Write comprehensive unit tests in" not in text:
                self.tool_calls += 1
                if self.tool_calls == 2:
                    raise KeyboardInterrupt
            return super().chat(messages, tools, options)

    def run(project_dir):
        client = Restarting(fix_failures=10 ** 6)
        ok, msg, details = sb.build_standalone_project_staged(SimpleNamespace(client=client), PROMPT, max_attempts=0)
        assert client.tool_calls == 2, "the tool was never rewritten from scratch"
        assert details.get("stopped")
    _in_temp_workspace(run)


MOD = q

CRASH_OUTPUT = (
    "EE\n" + "=" * 70 + "\nERROR: test_json (__main__.TestLogsum.test_json)\n" + "-" * 70 + "\n"
    "Traceback (most recent call last):\n"
    '  File "D:\\ws\\logsum\\test_logsum.py", line 12, in test_json\n'
    "    res = self.run_tool(self.text_file(data))\n"
    "TypeError: ToolTestCase.text_file() missing 1 required positional argument: 'text'\n\n"
    + "=" * 70 + "\nERROR: test_level (__main__.TestLogsum.test_level)\n" + "-" * 70 + "\n"
    "Traceback (most recent call last):\n"
    '  File "D:\\ws\\logsum\\test_logsum.py", line 20, in test_level\n'
    "TypeError: ToolTestCase.text_file() missing 1 required positional argument: 'text'\n\n"
    + "-" * 70 + "\nRan 2 tests in 0.010s\n\nFAILED (errors=2)\n"
)


def test_tests_that_crash_in_their_own_code_are_blamed_on_the_tests():
    """Live bug: TypeErrors inside the generated tests were classified as TOOL defects, so the tool kept being
    rewritten while the broken tests stayed."""
    assert MOD.test_code_crashed(CRASH_OUTPUT, "logsum")
    tool_frame = CRASH_OUTPUT.replace("test_logsum.py", "logsum.py")
    assert not MOD.test_code_crashed(tool_frame, "logsum"), "a crash inside the tool is the tool's fault"
    hung = CRASH_OUTPUT.replace("TypeError", "subprocess.TimeoutExpired")
    assert not MOD.test_code_crashed(hung, "logsum"), "a timeout means the tool hung"
    failed = CRASH_OUTPUT.replace("ERROR: test_level", "FAIL: test_level")
    assert not MOD.test_code_crashed(failed, "logsum"), "an assertion failure may be the tool's fault"
    res = MOD.arbitrate_test_failure(request=LOGSUM, tool_code="x", test_code="x", test_output=CRASH_OUTPUT,
                                     failing_tests=["test_json", "test_level"], smoke_passed=True,
                                     coverage_passed=True, memory_passed=True)
    assert res["classification"] == "TEST" and res["lock_tool"], res


def test_scaffold_helpers_accept_content_only():
    """Live bug: `self.text_file(content)` (one argument) crashed every logsum test."""
    import subprocess
    d = Path(tempfile.mkdtemp())
    try:
        (d / "x.py").write_text("import sys\nprint(open(sys.argv[1]).read())\n", encoding="utf-8")
        sb.write_scaffold_file(d, "x")
        (d / "test_x.py").write_text(
            "from _scaffold import ToolTestCase\nimport unittest\n"
            "class T(ToolTestCase):\n"
            "    def test_one_arg(self):\n"
            "        p = self.text_file('INFO a' + chr(10) + 'ERROR b')\n"
            "        self.assertIn('ERROR b', self.run_tool(str(p)).stdout)\n"
            "    def test_name_only(self):\n"
            "        self.assertEqual(self.text_file('empty.txt').read_text(), '')\n"
            "    def test_two_args(self):\n"
            "        self.assertEqual(self.text_file('a.log', 'x').name, 'a.log')\n"
            "    def test_csv_json(self):\n"
            "        self.assertTrue(self.csv_file([['a'], ['1']]).name.endswith('.csv'))\n"
            "        self.assertTrue(self.json_file({'k': 1}).name.endswith('.json'))\n"
            "if __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
        r = subprocess.run([sys.executable, "test_x.py"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_failing_tests_are_explained_with_their_input():
    """Live bug: the model saw raw unittest output and returned the identical tool five times; each failing test is
    now paired with its own input and expectation, tagged HOW TO FIX so it is emphasised in the fix prompt."""
    out = ("F\n" + "=" * 70 + "\nFAIL: test_single (__main__.T.test_single)\n" + "-" * 70 + "\n"
           "Traceback (most recent call last):\n  File \"test_logsum.py\", line 9, in test_single\n"
           "AssertionError: 'INFO' not found in 'Level | Count\\nOTHER | 1'\n\n" + "-" * 70 + "\nRan 1 test\n")
    code = ("class T(ToolTestCase):\n    def test_single(self):\n        data = 'INFO: hello'\n"
            "        res = self.run_tool(self.text_file(data))\n        self.assertIn('INFO', res.stdout)\n\n"
            "    def test_other(self):\n        pass\n")
    text = MOD.explain_test_failures(out, code)
    assert "test_single" in text and "INFO: hello" in text and "OTHER | 1" in text and "test_other" not in text, text
    fix = MOD.emphasize_required_fixes("STAGE: FIX\nbody", text)
    assert "REQUIRED CHANGES" in fix and "INFO: hello" in fix.split("body")[0], fix


def test_feedback_is_trimmed_keeping_head_and_tail():
    text = "RESEARCH NOTES head " + "x" * 20000 + " UNIT TEST OUTPUT tail"
    out = MOD.fit_feedback(text, 6000)
    assert len(out) < 6200 and out.startswith("RESEARCH NOTES head") and out.endswith("UNIT TEST OUTPUT tail")
    assert MOD.fit_feedback("short", 6000) == "short"


def test_near_miss_between_expected_text_and_test_input_is_spelled_out():
    """Live bug: tests fed 'INFO: msg'; the tool compared 'INFO:' to 'INFO' and the model never noticed the colon."""
    body = "data = 'INFO: one\\nWARNING: two'\nres = self.run_tool(self.text_file(data))\nself.assertIn('INFO', res.stdout)"
    hint = MOD._near_miss_hint("AssertionError: 'INFO' not found in 'OTHER | 2'", body)
    assert "'INFO:'" in hint, hint
    assert MOD._near_miss_hint("AssertionError: 'INFO' not found in 'x'", "data = 'INFO one'") == ""
    assert MOD._near_miss_hint("AssertionError: 1 != 2", body) == ""


def test_fixture_files_are_byte_exact():
    """Live bug: text-mode writes on Windows turned \\n into \\r\\n, so a test expecting 10 bytes saw 16."""
    import subprocess
    d = Path(tempfile.mkdtemp())
    try:
        (d / "x.py").write_text("print(1)", encoding="utf-8")
        sb.write_scaffold_file(d, "x")
        (d / "test_x.py").write_text(
            "from _scaffold import ToolTestCase\nimport unittest\n"
            "class T(ToolTestCase):\n"
            "    def test_bytes(self):\n"
            "        self.assertEqual(self.text_file('a.txt', 'ab' + chr(10) + 'cd' + chr(10)).stat().st_size, 6)\n"
            "        self.assertEqual(self.write_file('b.txt', 'x' + chr(10)).stat().st_size, 2)\n"
            "        self.assertEqual(self.folder('f', {'c.txt': 'y' + chr(10)}).joinpath('c.txt').stat().st_size, 2)\n"
            "if __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
        r = subprocess.run([sys.executable, "test_x.py"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_file_counter_is_not_treated_as_a_duplicate_finder():
    """Live bug: 'group' alone demanded duplicate-grouping code in any tool, e.g. 'group results by extension'."""
    ext = ("Build me a command-line tool called `extcount`. Point it at a folder and it counts files by extension "
           "and prints a table of extension, file count and total bytes. Group results by extension.")
    ok, missing = q.check_requirements_coverage(ext, "import argparse\nimport os\nprint(1)\n")
    assert not any("duplicate" in m.lower() for m in missing), missing


def test_total_column_does_not_add_a_total_row():
    """Live bug: 'total bytes' (a column) switched on the Total-row listing rules; tests then demanded a Total row."""
    ext = ("Build me a command-line tool called `extcount`. Point it at a folder and it counts files by extension and "
           "prints an aligned table of extension, file count and total bytes. It needs a `--json` option.")
    assert "TOTAL LINE: FILTER ROWS FIRST" not in (lambda r: sb.build_tool_generation_prompt(r, sb.extract_project_name(r), ["--json"]))(ext).upper()
    lines = ("Build me a command-line tool called `textcount` that prints lines, words and characters for one or more "
             "text files, plus a total line when there is more than one file. It needs a `--json` option.")
    assert "TOTAL LINE: FILTER ROWS FIRST" in (lambda r: sb.build_tool_generation_prompt(r, sb.extract_project_name(r), ["--json"]))(lines).upper()


def test_tool_result_accepts_the_forms_models_write():
    """Live bug: plugins wrote ToolResult(result=...) and every build failed on the missing 'status' argument."""
    from ogun_core.base_tool import ToolResult as TR
    a = TR(result={"n": 1})
    assert a.status == "success" and a.output == {"n": 1} and "result" not in a
    assert TR(output=2).status == "success"
    c = TR(error="boom")
    assert c.status == "error" and c.error_trace == "boom"
    assert TR("success", 3).output == 3 and TR.error("x").status == "error"
    # live bug: generated plugin tests asserted `result.success is True` (a bound method) and always failed
    ok, bad = TR.success(output=5), TR.error("boom")
    assert ok.success is True and ok.error is None and bad.success is False and bad.error == "boom"
    f = TR(success=False, output="No text provided")
    assert f.status == "error" and f.error == "No text provided" and TR(success=True, output=1).status == "success"
    try:
        TR("weird")
        raise AssertionError("an invalid status must still be rejected")
    except ValueError:
        pass


def test_path_lister_is_not_given_the_lookup_oracle():
    """Live bug: 'prints every key path in dotted form' matched 'dotted' and the lookup oracle demanded
    `keypaths file server.host` print 'localhost'."""
    lister = ("Build me a command-line tool called `keypaths`. Give it a JSON file and it prints every key path in "
              "dotted form (like server.ports.0) together with the type of its value. It needs a `--json` option.")
    getter = ("Build me a command-line tool called tomlget. Give it a TOML file and a dotted key path like a.b and it "
              "prints the value. It needs a --json option.")
    assert not MOD.is_value_lookup_request(lister.lower()) and MOD.is_value_lookup_request(getter.lower())
    assert "dotted_path_lookup" not in str(MOD.OracleRegistry.applicable_oracles(lister))
    assert "dotted_path_lookup" in str(MOD.OracleRegistry.applicable_oracles(getter))


def test_requested_output_words_are_required_in_the_tool():
    """Live bug: keypaths printed bool/NoneType/int although the request listed (object, array, string, number,
    boolean or null), and its own tests expected the same wrong names."""
    req = ("Build me a command-line tool called `keypaths`. Give it a JSON file and it prints every key path together "
           "with the type of its value (object, array, string, number, boolean or null), one per line.")
    assert MOD.requested_output_values(req) == ["object", "array", "string", "number", "boolean", "null"]
    wrong = "import json\ndef t(v):\n    return type(v).__name__\n"
    right = ("def t(v):\n    return {dict: 'object', list: 'array', str: 'string', bool: 'boolean', "
             "type(None): 'null'}.get(type(v), 'number')\n")
    assert any("output value" in m for m in MOD.check_requirements_coverage(req, wrong)[1])
    assert not any("output value" in m for m in MOD.check_requirements_coverage(req, right)[1])
    # fixing some of the words is progress, not a regression
    part = "def t(v):\n    return {dict: 'object', list: 'array'}.get(type(v), 'x')\n"
    before = [m for m in MOD.check_requirements_coverage(req, wrong)[1] if "output value" in m]
    after = [m for m in MOD.check_requirements_coverage(req, part)[1] if "output value" in m]
    assert len(after) == 4 and set(after) <= set(before), (before, after)
    assert MOD.requested_output_values("A tool (case-insensitive, ignoring punctuation) for text.") == []


def test_tests_expecting_python_type_names_are_blamed_on_the_tests():
    """Live: keypaths tests expected 'config.dict' although the request listed (object, array, ... or null)."""
    req = ("Build me a command-line tool called `keypaths`. Give it a JSON file and it prints every key path with the "
           "type of its value (object, array, string, number, boolean or null). It needs a `--json` option.")
    code = ("class T(ToolTestCase):\n    def test_types(self):\n        res = self.run_tool('x.json')\n"
            "        self.assertIn('config.dict', res.stdout)\n")
    res = MOD.arbitrate_test_failure(request=req, tool_code="x", test_code=code,
                                     test_output="FAIL: test_types\nAssertionError: 'config.dict' not found in 'config object'",
                                     failing_tests=["test_types"], smoke_passed=True, coverage_passed=True,
                                     memory_passed=True)
    assert res["classification"] == "TEST", res


def test_python_type_names_in_tests_are_mapped_to_the_requested_words():
    """Live: keypaths tests expected 'config.dict' for seven attempts although the request listed JSON type names."""
    req = ("Build me a command-line tool called `keypaths`. Give it a JSON file and it prints every key path with the "
           "type of its value (object, array, string, number, boolean or null).")
    code = ("data = self.assertJsonOutput(res)\nself.assertTrue(isinstance(data, dict))\n"
            "self.assertIn('config.dict', res.stdout)\nself.assertEqual(data, {'a': 'int', 'b': 'NoneType', 'c': 'list'})\n")
    out = MOD.map_python_type_names(code, req)
    assert "isinstance(data, dict)" in out, out
    assert "'config.object'" in out and "'number'" in out and "'null'" in out and "'array'" in out, out
    assert MOD.map_python_type_names(code, "Build me a tool called `x` that counts words.") == code


def test_separator_only_mismatch_is_a_test_defect_and_blocks_are_per_test():
    """Live: a test expected 'config.object' while the tool printed 'config object'; and another test's empty output
    ("not found in ''") was read as THIS test's failure, so the tool was blamed."""
    out = ("FF\n" + "=" * 70 + "\nFAIL: test_depth (__main__.T.test_depth)\n" + "-" * 70 + "\n"
           "AssertionError: 'config.object' not found in 'config object\\nusers array\\n'\n\n"
           + "=" * 70 + "\nFAIL: test_empty (__main__.T.test_empty)\n" + "-" * 70 + "\n"
           "AssertionError: 'No paths' not found in ''\n\n" + "-" * 70 + "\nRan 2 tests\n")
    assert MOD.separator_only_mismatch(MOD._test_block(out, "test_depth"))[0] == "config.object"
    assert "not found in ''" not in MOD._test_block(out, "test_depth")
    assert MOD.separator_only_mismatch("AssertionError: 'INFO' not found in 'OTHER 1'") is None
    req = "Build me a command-line tool called `keypaths` that lists key paths with their types. It needs a `--depth N` option."
    code = "class T(ToolTestCase):\n    def test_depth(self):\n        res = self.run_tool('a.json', '--depth', '1')\n"
    res = MOD.arbitrate_test_failure(request=req, tool_code="x", test_code=code, test_output=out,
                                     failing_tests=["test_depth", "test_empty"], smoke_passed=True,
                                     coverage_passed=True, memory_passed=True)
    assert res["classification"] == "TEST", res


def test_json_value_type_oracle_catches_wrong_types_and_root_line():
    """Live: keypaths passed its own tests while printing `debug number` for `true` and an empty-path root line."""
    import os
    req = ("Build me a command-line tool called `keypaths`. Give it a JSON file and it prints every key path in dotted "
           "form together with the type of its value (object, array, string, number, boolean or null). It needs a "
           "`--json` option and a `--depth N` option.")
    assert "json_value_types" in MOD.OracleRegistry.applicable_oracles(req)
    tool = Path(__file__).resolve().parent / "fixtures" / "options" / "types_tool.py"
    old = os.environ.get("BUG")
    try:
        for bug, ok_expected in (("", True), ("boolnum", False), ("root", False), ("dupes", False)):
            os.environ["BUG"] = bug
            ok, err = MOD.OracleRegistry.oracle_json_value_types(tool, req, tool.parent)
            assert ok is ok_expected, (bug, err)
            if not ok:
                assert "HOW TO FIX" in err, err
    finally:
        if old is None:
            os.environ.pop("BUG", None)
        else:
            os.environ["BUG"] = old


def test_web_research_is_off_unless_opted_in():
    """Ogun promises to stay local, so build research must not touch the web by default."""
    import os
    from ogun_core import research
    old = os.environ.pop("OGUN_WEB_RESEARCH", None)
    try:
        assert research.web_research_enabled() is False
        with patch("urllib.request.urlopen", side_effect=AssertionError("network used while research is off")):
            out = research.search_web("python TypeError")
        assert out.startswith("Error:") and "OGUN_WEB_RESEARCH=1" in out, out
        for val, want in (("1", True), ("true", True), ("0", False), ("off", False), ("", False)):
            os.environ["OGUN_WEB_RESEARCH"] = val
            assert research.web_research_enabled() is want, (val, want)
    finally:
        if old is None:
            os.environ.pop("OGUN_WEB_RESEARCH", None)
        else:
            os.environ["OGUN_WEB_RESEARCH"] = old


def test_command_line_plugin_build_uses_no_limit_setting():
    """--build-plugin called build_plugin_staged without max_attempts, so it still stopped after 3 attempts."""
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    call = src[src.index("build_plugin_staged(\n"):]
    call = call[:call.index(")")]
    assert "max_attempts=BUILD_MAX_ATTEMPTS" in call, call


ROMAN_TEST = """import sys
from plugins.custom_tools.roman_numeral import RomanNumeral

def main():
    t = RomanNumeral()
    for bad in ["IIII", "VX", "0"]:
        res = t.execute(value=bad, direction="to_int")
        if res.status != "error":
            print("TEST ASSERTION FAILED: Expected error, got success", file=sys.stderr)
            sys.exit(1)
    print("PASS")

main()
"""


def test_fix_prompt_shows_failing_check_and_current_code():
    """Live: 93 attempts saw only 'Expected error, got success' and never which input failed."""
    err = "Plugin unit tests failed.\nSTDERR:\nTEST ASSERTION FAILED: Expected error, got success\n"
    ctx = plb.failing_check_context(ROMAN_TEST, err)
    assert '"IIII", "VX"' in ctx and ">>" in ctx, ctx
    tb = 'Traceback:\n  File "C:/x/plugins/tests/test_roman_numeral.py", line 7, in main\nAssertionError'
    assert "7>>" in plb.failing_check_context(ROMAN_TEST, tb).replace(" ", "")

    tool = "```python\nclass RomanNumeral:\n    name: str = \"roman_numeral\"\n```\n"
    reply = tool + "```python\n" + ROMAN_TEST + "```"

    class PM:
        def create_or_update_plugin(self, name, tool_code, test_code):
            return SimpleNamespace(status="error", error_trace=err)

    eng = _FakeEngine([reply])
    eng.messages = [{"role": "system", "content": "sys"}]
    calls = []
    real = eng.run_turn

    def run_turn(prompt):
        calls.append(len(eng.messages))
        eng.messages.append({"role": "user", "content": prompt})
        return real(prompt)

    eng.run_turn = run_turn
    ok, _, _ = plb.build_plugin_staged(eng, PM(), "roman numeral plugin", max_attempts=3)
    assert not ok
    fix = eng.prompts[1]
    assert "=== CURRENT TOOL CODE ===" in fix and "class RomanNumeral" in fix, fix[:400]
    assert "=== CURRENT TEST CODE ===" in fix and '"IIII", "VX"' in fix
    assert "THE FAILING CHECK" in fix and "roman numeral plugin" in fix
    assert calls == [1, 1, 1], f"history was not reset before each attempt: {calls}"


def test_plugin_build_restarts_after_three_identical_failures():
    """Free-GPU target (a plugin in under 10 minutes): restart after 3 identical failures, not 5."""
    eng = _FakeEngine([PLUGIN_REPLY])
    ok, _, _ = plb.build_plugin_staged(eng, _FakePM(succeed_on=3), "Build adder plugin", max_attempts=0)
    assert ok
    stage1 = [p for p in eng.prompts if "Stage 1/2" in p]
    assert len(stage1) == 2, f"expected one fresh restart, got {len(stage1)} stage-1 prompts"


def test_repeated_failures_raise_temperature_and_skip_identical_code():
    assert plb.same_failure_streak(["a", "b", "b"]) == 2 and plb.same_failure_streak([]) == 0
    assert plb.retry_temperature(1) == plb.FIX_TEMPERATURE < plb.retry_temperature(2) < plb.retry_temperature(3) <= 0.9
    eng = _FakeEngine([PLUGIN_REPLY])
    pm = _FakePM(succeed_on=0)
    plb.build_plugin_staged(eng, pm, "Build adder plugin", max_attempts=4)
    assert pm.calls == 2, f"identical code was re-run back to back: {pm.calls} runs in 4 attempts"


def test_fix_reply_with_only_the_tool_keeps_the_test():
    assert plb.is_tool_block("from x import y\nclass Adder(BaseTool):\n    pass")
    assert not plb.is_tool_block("import sys\nfrom plugins.custom_tools.adder import Adder\nassert True")


def test_real_engine_build_steps_skip_run_turn_and_tool_schemas():
    from ogun_core.engine import CoreEngine
    seen = []

    class Client:
        def chat(self, messages, tools=None, options=None):
            seen.append((len(messages), tools, (options or {}).get("temperature")))
            return {"content": "ok"}

    eng = CoreEngine.__new__(CoreEngine)
    eng.client = Client()
    eng.run_turn = lambda p: (_ for _ in ()).throw(AssertionError("run_turn used"))
    assert plb.generate(eng, "hi", 0.45) == "ok" and seen == [(1, None, 0.45)]


def test_failed_plugin_test_is_kept_in_quarantine():
    from ogun_core.plugin_manager import PluginManager
    td = Path(tempfile.mkdtemp())
    try:
        pm = PluginManager(base_dir=td)
        code = ("from typing import Any, Dict\nfrom ogun_core.base_tool import BaseTool, ToolResult\n\n"
                "class Qq(BaseTool):\n    name: str = \"qq_tool\"\n    version: str = \"1.0.0\"\n"
                "    description: str = \"q\"\n    parameters_schema: Dict[str, Any] = {\"type\": \"object\"}\n\n"
                "    def execute(self, **kwargs: Any) -> ToolResult:\n        return ToolResult.success(output=1)\n")
        res = pm.create_or_update_plugin("qq_tool", code, "import sys\nprint('boom', file=sys.stderr)\nsys.exit(1)\n")
        assert res.status == "error"
        assert (td / "plugins" / "quarantine" / "test_qq_tool.py").exists(), list((td / "plugins").rglob("*"))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_plugin_test_must_cover_every_quoted_example_in_the_request():
    """Live: the test skipped 'VX' from "invalid numerals like 'IIII' or 'VX'", so a tool that returned 5 for it passed."""
    req = ("A plugin named roman_numeral. to_roman: '1994' -> 'MCMXCIV'. Invalid numerals like 'IIII' or 'VX' must "
           "return an error. It's simple.")
    test = 'tool.execute(value="1994"); assert out == "MCMXCIV"; tool.execute(value="IIII")  # roman_numeral'
    assert plb.request_examples_missing(req, test) == ["VX"]
    assert plb.request_examples_missing(req, test + ' tool.execute(value="VX")') == []
    upd = ("UPDATE the existing plugin `roman_numeral`.\n=== CURRENT plugins/custom_tools/x.py ===\nname = 'zzz'\n"
           "=== REQUESTED CHANGE ===\nAlso accept lowercase like 'xiv'\n=== END ===")
    assert plb.request_examples_missing(upd, "x") == ["xiv"]

    reply = "```python\nclass T:\n    name: str = \"roman_numeral\"\n```\n```python\nassert 'IIII'\n```"
    eng = _FakeEngine([reply])
    pm = _FakePM(succeed_on=1)
    ok, msg, _ = plb.build_plugin_staged(eng, pm, "Reject 'IIII' and 'VX'", max_attempts=2)
    assert not ok and pm.calls == 0 and "'VX'" in msg, msg


def test_quoted_parameter_names_are_not_required_test_inputs():
    """Live: 'value' and 'direction' (parameter names in the request) were demanded as test inputs; an attempt was lost."""
    from ogun_core import plugin_builder as pf
    req = ("Parameters: 'value' (string) and 'direction' ('to_roman' or 'to_int'). Invalid numerals like 'IIII' "
           "or 'VX' must return an error.")
    code = "def execute(self, value=None, direction=None):  # to_roman / to_int"
    test = "execute(value='IIII', direction='to_int')"
    assert pf.request_examples_missing(req, test, code) == ["VX"]
    assert pf.request_examples_missing(req, test) == ["to_roman", "VX"]  # without the code, names count


def test_tool_syntax_error_shows_the_tool_line():
    """Live: 'SyntaxError at line 17' repeated four attempts; the fix prompt never showed line 17 of the tool."""
    from ogun_core import plugin_builder as pf
    code = "\n".join(f"x{i} = {i}" for i in range(1, 30))
    ctx = pf.tool_code_error_context(code, "Pre-flight code validation failed: SyntaxError at line 17: invalid syntax")
    assert "17>> x17 = 17" in ctx and "x14 = 14" in ctx, ctx
    tb = 'Traceback:\n  File "C:/p/plugins/custom_tools/roman.py", line 3, in execute\nKeyError'
    assert "3>> x3 = 3" in pf.tool_code_error_context(code, tb)
    assert pf.tool_code_error_context(code, 'File "C:/p/plugins/tests/test_roman.py", line 5') == ""


if __name__ == "__main__":
    fails = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {_name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)


def test_caught_name_error_shows_where_the_name_is_used():
    """Live: the tool caught "name 'roman_to_value' is not defined" itself (no line number); four fixes missed it."""
    from ogun_core import plugin_builder as pf
    code = "def a():\n    m = {1: 2}\n    return m\n\ndef b(x):\n    return m[x]\n"
    ctx = pf.tool_code_error_context(code, "Execution failed: name 'm' is not defined")
    assert "`m` is used where it is not defined" in ctx and "6>>     return m[x]" in ctx, ctx
    assert ctx.count("def b(x)") == 1, "overlapping windows must merge into one listing"


def test_plugin_test_imports_are_normalized():
    """Measured: a correct tool failed because its test imported plugins.custom_tools.roman_numeral_tool."""
    tool = "from ogun_core.base_tool import BaseTool\nclass RomanNumeral(BaseTool):\n    name: str = 'roman_numeral'\n"
    test = ('"""doc"""\nfrom __future__ import annotations\n'
            "from plugins.custom_tools.roman_numeral_tool import RomanNumeralTool\nimport sys\n"
            "t = RomanNumeralTool()\n")
    out = plb.normalize_plugin_test(test, tool, "roman_numeral")
    assert "from plugins.custom_tools.roman_numeral import RomanNumeral as RomanNumeralTool" in out, out
    lines = out.splitlines()
    assert lines[0] == '"""doc"""' and lines[1].startswith("from __future__")
    assert out.index("parents[2]") < out.index("from plugins.custom_tools")
    assert plb.normalize_plugin_test(out, tool, "roman_numeral") == out      # idempotent


def test_a_draft_gets_two_fixes_then_a_fresh_hotter_draft():
    eng = _FakeEngine([PLUGIN_REPLY.replace("adder", f"adder{i}") for i in range(6)] + [PLUGIN_REPLY])
    plb.build_plugin_staged(eng, _FakePM(succeed_on=0), "Build adder plugin", max_attempts=4)
    kinds = ["draft" if "Stage 1/2" in p else "fix" for p in eng.prompts]
    assert kinds == ["draft", "fix", "fix", "draft"], kinds
