"""Gate 3: the general loop's PLAN -> EXECUTE -> VERIFY controller (ogun_core/turn_controller.py + CoreEngine)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.engine import CoreEngine  # noqa: E402
from ogun_core.kit.run_journal import RunJournal  # noqa: E402

DEMO = "def double(x):\n    return 2 * x\n"
DEMO_TEST = ("import sys, unittest\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n"
             "from demo import double\n\n\nclass T(unittest.TestCase):\n    def test_double(self):\n"
             "        self.assertEqual(double(4), 8)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def say(text):
    return {"content": text, "tool_calls": []}


class Scripted:
    """Answers plan / verdict requests (recognized by their JSON schema) and plays a script for the tool rounds."""
    supports_structured = True
    model = "qwen2.5-coder:14b"

    def __init__(self, plan, script, verdicts=None):
        self.plan, self.script, self.verdicts = plan, list(script), list(verdicts or [])
        self.seen, self.plan_calls, self.verdict_calls = [], 0, 0

    def chat(self, messages, tools=None, options=None, format=None):
        props = (format or {}).get("properties", {}) if isinstance(format, dict) else {}
        if "steps" in props:
            self.plan_calls += 1
            return {"content": json.dumps(self.plan)}
        if "done" in props:
            self.verdict_calls += 1
            v = self.verdicts.pop(0) if self.verdicts else {"done": True, "missing": []}
            return {"content": json.dumps(v)}
        self.seen.append(messages)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _engine(tmp_path, client):
    eng = CoreEngine(client=client, base_dir=tmp_path)
    eng.retry_backoff = (0.0, 0.0)
    return eng


PLAN = {"steps": [{"goal": "write demo.py", "acceptance": "file_exists", "target": "workspace/demo/demo.py"},
                  {"goal": "test it", "acceptance": "tests_pass", "target": "workspace/demo/test_demo.py"}]}


def test_todo_list_is_not_accepted_as_done(tmp_path):
    client = Scripted(PLAN, [
        say("Here is the plan:\n1. Write demo.py\n2. Write test_demo.py\nYou can run it afterwards."),
        call("write_file", filepath="workspace/demo/demo.py", content=DEMO),
        call("write_file", filepath="workspace/demo/test_demo.py", content=DEMO_TEST),
        say("Done: demo.py with double() and a passing test."),
    ])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("create workspace/demo/demo.py with double(x) and a passing unittest")
    assert "Done: demo.py" in out and eng.last_turn_status["status"] == "success", out
    assert (tmp_path / "workspace" / "demo" / "demo.py").exists()
    nudges = [m["content"] for msgs in client.seen for m in msgs if m["role"] == "user" and m["content"].startswith("NOT DONE")]
    assert nudges and "does not exist yet" in nudges[0]
    assert client.plan_calls == 1 and client.verdict_calls == 0   # file/test checks hold: no verdict needed
    # the plan is pinned in the context for every round
    assert all("## Current plan" in msgs[0]["content"] for msgs in client.seen)
    st = RunJournal.latest(tmp_path / "workspace" / ".journal").state()
    assert st["plan"] == PLAN and st["status"] == "done" and st["verdicts"][0]["done"] is False


def test_touched_file_must_parse_before_the_turn_ends(tmp_path):
    plan = {"steps": [{"goal": "write calc.py", "acceptance": "syntax_ok", "target": "workspace/c/calc.py"}]}
    client = Scripted(plan, [
        call("write_file", filepath="workspace/c/calc.py", content="def f(:\n    return 1\n"),
        say("Finished."),
        call("replace_in_file", filepath="workspace/c/calc.py", old_text="def f(:", new_text="def f():"),
        say("Fixed and finished."),
    ])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("write workspace/c/calc.py with a function f returning 1")
    assert out.startswith("Fixed and finished.")
    nudge = [m["content"] for m in client.seen[2] if m["role"] == "user" and m["content"].startswith("NOT DONE")]
    assert nudge and "SyntaxError at line 1" in nudge[0]


def test_failing_touched_test_blocks_completion(tmp_path):
    bad = DEMO.replace("2 * x", "3 * x")
    client = Scripted(PLAN, [
        call("write_file", filepath="workspace/demo/demo.py", content=bad),
        call("write_file", filepath="workspace/demo/test_demo.py", content=DEMO_TEST),
        say("All done."),
        call("replace_in_file", filepath="workspace/demo/demo.py", old_text="3 * x", new_text="2 * x"),
        say("Fixed: the test passes now."),
    ])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("create workspace/demo/demo.py with double(x) and a passing unittest")
    assert out.startswith("Fixed: the test passes now.")
    nudge = [m["content"] for m in client.seen[3] if m["role"] == "user" and m["content"].startswith("NOT DONE")]
    assert nudge and "test_demo.py fails" in nudge[0]


def test_transient_timeout_is_retried_without_asking_the_user(tmp_path):
    plan = {"steps": [{"goal": "answer", "acceptance": "answer_only", "target": ""}]}
    client = Scripted(plan, [TimeoutError("read timed out"), say("mul multiplies its two arguments.")])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("what does mul do")
    assert out == "mul multiplies its two arguments." and "continue" not in out
    assert client.verdict_calls == 1          # every planned turn gets the completion verdict


def test_unfinished_work_is_reported_honestly_after_three_checks(tmp_path):
    plan = {"steps": [{"goal": "write demo.py", "acceptance": "syntax_ok", "target": "workspace/demo/demo.py"},
                      {"goal": "explain it", "acceptance": "answer_only", "target": ""}]}
    never = {"done": False, "missing": ["the docstring the user asked for"]}
    client = Scripted(plan, [call("write_file", filepath="workspace/demo/demo.py", content=DEMO)]
                      + [say("Done.")] * 4, verdicts=[never] * 4)
    eng = _engine(tmp_path, client)
    out = eng.run_turn("write workspace/demo/demo.py with a docstring")
    assert "Not finished: the automatic checks still report" in out and "docstring" in out
    assert eng.last_turn_status["status"] == "failed" and client.verdict_calls == 4


def test_continue_resumes_from_the_journal_without_replanning(tmp_path):
    plan = {"steps": [{"goal": "write demo.py", "acceptance": "file_exists", "target": "workspace/demo/demo.py"}]}
    client = Scripted(plan, [call("write_file", filepath="workspace/demo/demo.py", content=DEMO)]
                      + [TimeoutError("down")] * 3 + [say("demo.py is written.")])
    eng = _engine(tmp_path, client)
    first = eng.run_turn("write workspace/demo/demo.py")
    assert "timed out" in first and eng.last_turn_status["status"] == "timeout"
    out = eng.run_turn("continue")
    assert out.startswith("demo.py is written.") and client.plan_calls == 1
    resumed_ctx = client.seen[-1][0]["content"]
    assert "Resumed after an interruption" in resumed_ctx and "write_file" in resumed_ctx


def test_library_module_fixes_go_to_the_general_loop(tmp_path):
    """Live: a bug fix in a plain module went to the CLI updater, whose command-line smoke tests can never pass."""
    from ogun_core.engine import _is_cli_project
    lib = tmp_path / "workspace" / "calc"
    lib.mkdir(parents=True)
    (lib / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    cli = tmp_path / "workspace" / "tally"
    cli.mkdir(parents=True)
    (cli / "tally.py").write_text("import argparse\n\ndef main():\n    argparse.ArgumentParser()\n", encoding="utf-8")
    assert not _is_cli_project(tmp_path, "calc")
    assert _is_cli_project(tmp_path, "tally")


def test_repeating_the_same_failing_call_gets_a_stall_note(tmp_path):
    """Live: ~20 identical failing test runs, the imported module never written."""
    plan = {"steps": [{"goal": "write demo.py", "acceptance": "file_exists", "target": "workspace/demo/demo.py"},
                      {"goal": "test it", "acceptance": "tests_pass", "target": "workspace/demo/test_demo.py"}]}
    client = Scripted(plan, [
        call("write_file", filepath="workspace/demo/test_demo.py", content=DEMO_TEST),
        call("run_python_test", test_path="workspace/demo/test_demo.py"),
        call("run_python_test", test_path="workspace/demo/test_demo.py"),
        call("write_file", filepath="workspace/demo/demo.py", content=DEMO),
        say("Done."),
    ])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("create workspace/demo/demo.py with double(x) and a passing unittest")
    assert out.startswith("Done.")
    notes = [m["content"] for m in client.seen[3] if m["role"] == "user" and "more than once" in m["content"]]
    assert notes and "workspace/demo/demo.py does not exist yet" in notes[0]
    assert "You have now run run_python_test('workspace/demo/test_demo.py') with the same result" in notes[0]


def test_unknown_builtin_arguments_are_rejected(tmp_path):
    eng = CoreEngine(client=object(), base_dir=tmp_path)
    out = json.loads(eng.dispatch_tool("run_python_test", {"test_path": "t.py", "working_directory": "x"}))
    assert out["error"] == "invalid_arguments" and any("working_directory" in p for p in out["problems"])


def test_relative_paths_are_found_under_workspace(tmp_path):
    d = tmp_path / "workspace" / "greet"
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("found me", encoding="utf-8")
    eng = CoreEngine(client=object(), base_dir=tmp_path)
    assert "found me" in eng.dispatch_tool("read_file", {"filepath": "greet/notes.txt"})


def test_verdict_is_scoped_to_the_request():
    from ogun_core import turn_controller as tc
    assert not tc.needs_verdict({"steps": [{"acceptance": "tests_pass", "target": "t.py", "goal": "g"}]}, [])
    assert tc.needs_verdict({"steps": [{"acceptance": "answer_only", "target": "", "goal": "g"}]}, [])


def test_an_identical_failing_call_is_refused_after_five_repeats(tmp_path):
    """Live: after the module existed, the same failing test (a relative import) was re-run ~20 times unchanged."""
    plan = {"steps": [{"goal": "test it", "acceptance": "tests_pass", "target": "workspace/demo/test_demo.py"}]}
    bad_test = DEMO_TEST.replace("from demo import double", "from .demo import double")
    run = call("run_python_test", test_path="workspace/demo/test_demo.py")
    client = Scripted(plan, [call("write_file", filepath="workspace/demo/demo.py", content=DEMO),
                             call("write_file", filepath="workspace/demo/test_demo.py", content=bad_test)]
                      + [run] * 7
                      + [call("replace_in_file", filepath="workspace/demo/test_demo.py",
                              old_text="from .demo import double", new_text="from demo import double"),
                         run, say("Fixed the import; the test passes.")])
    eng = _engine(tmp_path, client)
    out = eng.run_turn("create workspace/demo/demo.py with double(x) and a passing unittest")
    assert out.startswith("Fixed the import")
    tool_texts = [m["content"] for m in eng.last_ledger.messages() if m["role"] == "tool"] + \
                 [m["content"] for msgs in client.seen for m in msgs if m["role"] == "tool"]
    assert any(t.startswith("Not run: this exact call returned the same result 5 times") for t in tool_texts)
    notes = [m["content"] for msgs in client.seen for m in msgs if m["role"] == "user" and "more than once" in m["content"]]
    assert notes and any("ImportError" in n for n in notes), "the stall note must quote the actual error"
