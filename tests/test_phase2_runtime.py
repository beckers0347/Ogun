"""Gate 2 (shared kit, vendored into ogun_core/kit): sampling profiles, schema checks, structured output, isolated plugins."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.kit import plugin_runner, sampling  # noqa: E402
from ogun_core.kit.gpu_lease import pid_alive  # noqa: E402
from ogun_core.kit.schema_check import validate  # noqa: E402
from ogun_core.kit.structured import ask_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = """from ogun_core.base_tool import BaseTool, ToolResult
import os, sys, time
from pathlib import Path


class Probe(BaseTool):
    name = "probe"
    description = "test plugin"
    parameters_schema = {"type": "object", "properties": {"mode": {"type": "string"}, "n": {"type": "integer"},
                         "marker": {"type": "string"}}, "required": ["mode"]}

    def execute(self, **kwargs):
        Path(kwargs.get("marker") or os.devnull).write_text(str(os.getpid()))
        mode = kwargs["mode"]
        if mode == "hang":
            time.sleep(60)
        if mode == "crash":
            os._exit(3)
        return ToolResult.success(output={"double": kwargs.get("n", 0) * 2})
"""


def _plugin(tmp_path):
    f = tmp_path / "probe.py"
    f.write_text(PLUGIN, encoding="utf-8")
    return f


def test_sampling_profiles_by_phase():
    assert sampling.profile("plan") == {"temperature": 0.2, "top_p": 0.9}
    assert sampling.profile("code") == {"temperature": 0.05, "top_p": 0.85}
    assert sampling.profile("verdict") == {"temperature": 0.0, "top_p": 0.7}
    assert sampling.profile("code_fix", streak=3)["temperature"] == 0.8
    assert sampling.top_p_for(0.05) == 0.85 and sampling.top_p_for(0.0) == 0.7


def test_schema_check_names_the_bad_field():
    schema = {"type": "object", "properties": {"filepath": {"type": "string"}, "count": {"type": "integer"}},
              "required": ["filepath"], "additionalProperties": False}
    assert validate({"filepath": "a.py", "count": 2}, schema) == []
    problems = validate({"count": True, "extra": 1}, schema)
    assert "filepath: required but missing" in problems
    assert any(p.startswith("count: expected integer, got boolean") for p in problems)
    assert any(p.startswith("extra: not an allowed argument") for p in problems)


def test_structured_output_is_validated_and_retried_once():
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B"]}}, "required": ["answer"]}
    replies, prompts = iter(['{"answer": "C"}', '{"answer": "B"}']), []

    def ask(prompt, fmt, sampling_opts, num_predict):
        prompts.append((prompt, fmt, sampling_opts))
        return next(replies)

    assert ask_json(ask, "pick", schema) == {"answer": "B"}
    assert prompts[0][1] == schema and prompts[0][2] == {"temperature": 0.0, "top_p": 0.7}
    assert "rejected" in prompts[1][0] and "'C' is not one of" in prompts[1][0]
    assert ask_json(lambda *a: "not json", "pick", schema) is None


def test_isolated_plugin_runs_and_returns_its_result(tmp_path):
    res = plugin_runner.run_isolated(ROOT, _plugin(tmp_path), {"mode": "ok", "n": 21}, timeout=30)
    assert res["status"] == "success" and res["output"] == {"double": 42}, res


def test_hung_plugin_is_killed_at_its_timeout(tmp_path):
    marker = tmp_path / "pid.txt"
    t0 = time.time()
    res = plugin_runner.run_isolated(ROOT, _plugin(tmp_path), {"mode": "hang", "marker": str(marker)}, timeout=3)
    elapsed = time.time() - t0
    assert res["status"] == "error" and "timed out after 3s" in res["error_trace"], res
    assert elapsed < 3 + 8, f"took {elapsed:.1f}s"
    pid = int(marker.read_text())
    deadline = time.time() + 5
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    assert not pid_alive(pid), "the plugin process is still running"


def test_crashing_plugin_is_a_failed_call(tmp_path):
    res = plugin_runner.run_isolated(ROOT, _plugin(tmp_path), {"mode": "crash"}, timeout=30)
    assert res["status"] == "error" and "exited with code 3" in res["error_trace"], res


def test_invalid_plugin_arguments_are_rejected_before_anything_runs(tmp_path):
    from ogun_core.plugin_manager import PluginManager
    tools = tmp_path / "plugins" / "custom_tools"
    tools.mkdir(parents=True)
    (tools / "probe.py").write_text(PLUGIN, encoding="utf-8")
    pm = PluginManager(base_dir=tmp_path)
    pm.discover_tools()
    marker = tmp_path / "ran.txt"
    bad = pm.execute_plugin("probe", n="lots", marker=str(marker))
    assert bad.status == "error" and "nothing was run" in bad.error_trace
    assert "mode: required but missing" in bad.error_trace and "n: expected integer" in bad.error_trace
    assert not marker.exists(), "the plugin ran despite invalid arguments"
    good = pm.execute_plugin("probe", mode="ok", n=4, marker=str(marker))
    assert good.status == "success" and good.output == {"double": 8} and marker.exists()
    child_pid = int(marker.read_text())
    import os
    assert child_pid != os.getpid(), "the plugin ran inside Ogun instead of its own process"
    hung = pm.execute_plugin("probe", mode="hang", timeout=2)
    assert hung.status == "error" and "terminated" in hung.error_trace


def test_builtin_tool_arguments_are_validated(tmp_path):
    from ogun_core.engine import CoreEngine
    eng = CoreEngine(client=object(), base_dir=tmp_path)
    out = json.loads(eng.dispatch_tool("read_file", {"filepath": 5}))
    assert out["error"] == "invalid_arguments" and out["problems"] == ["filepath: expected string, got integer"]
    out = json.loads(eng.dispatch_tool("replace_in_file", {"filepath": "a.py"}))
    assert "old_text: required but missing" in out["problems"]


def test_text_tool_call_is_reasked_with_a_schema(tmp_path):
    """A tool call written as text is not regex-extracted: it is asked for again as schema-constrained JSON."""
    from ogun_core.engine import CoreEngine
    (tmp_path / "note.txt").write_text("hello from note", encoding="utf-8")
    calls = []

    class Client:
        supports_structured = True
        model = "qwen2.5-coder:14b"

        def chat(self, messages, tools=None, options=None, format=None):
            if isinstance(format, dict) and "steps" in format.get("properties", {}):     # the turn's plan request
                return {"content": json.dumps({"steps": [{"goal": "answer", "acceptance": "answer_only",
                                                          "target": ""}]})}
            calls.append({"format": format, "options": options})
            if len(calls) == 1:
                return {"content": '```json\n{"name": "read_file", "arguments": "note.txt"}\n```', "tool_calls": []}
            if len(calls) == 2:
                return {"content": json.dumps({"tool": "read_file", "arguments": {"filepath": "note.txt"}})}
            return {"content": "The note says hello.", "tool_calls": []}

    eng = CoreEngine(client=Client(), base_dir=tmp_path)
    out = eng.run_turn("what does note.txt say")
    assert "hello" in out.lower()
    assert calls[0]["options"]["temperature"] == 0.05 and calls[0]["format"] is None
    fmt = calls[1]["format"]
    assert fmt["required"] == ["tool", "arguments"] and "read_file" in fmt["properties"]["tool"]["enum"]
    tool_msgs = [m for m in eng.last_ledger.messages() if m["role"] == "tool"]
    assert tool_msgs and "hello from note" in tool_msgs[0]["content"]
