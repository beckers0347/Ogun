"""Gate 1: exact token accounting and the context ledger (vendored kit: ogun_core/kit)."""
import json
import sys
import urllib.request
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.kit.context_ledger import ContextLedger  # noqa: E402
from ogun_core.kit.tokenizer import get_tokenizer  # noqa: E402

MODEL = "qwen2.5-coder:14b"
TOK = get_tokenizer(MODEL)
TASK = "Fix the failing test in workspace/calc/test_calc.py without changing the test."


def _ollama_prompt_tokens(messages):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"num_predict": 1, "num_gpu": 48, "num_ctx": 8192, "num_batch": 512, "num_thread": 12}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["prompt_eval_count"]


def _ollama_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2).read()
        return True
    except Exception:
        return False


def _ledger(budget, tmp_path=None):
    led = ContextLedger(TOK.count_messages, budget, count_text=TOK.count, tool_output_cap=300,
                        keep_recent_results=2, spill_dir=tmp_path)
    led.set_system("You are a coding agent.")
    led.set_task(TASK)
    return led


def _big(n):
    return "\n".join(f"line {i}: " + "x" * 60 for i in range(n))


@pytest.mark.skipif(not TOK.exact, reason="model file not available for exact counting")
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_token_counts_match_the_model_within_one_percent():
    code = Path(__file__).read_text(encoding="utf-8")
    for content in ("Write a function that reverses a string.", code[:5000],
                    "Ünïcödé ✓ 日本語 🚀 tabs\tand  spaces 3.14159 CamelCase_snake"):
        msgs = [{"role": "system", "content": f"nonce {uuid.uuid4()}"}, {"role": "user", "content": content}]
        ours = TOK.count_messages(msgs) + 3                      # + the "<|im_start|>assistant\n" generation prompt
        theirs = _ollama_prompt_tokens(msgs)
        assert abs(ours - theirs) <= max(1, theirs * 0.01), (ours, theirs)


def test_ledger_token_count_is_the_tokenizer_count():
    led = _ledger(5000)
    led.add_assistant("reading", tool_calls=[{"function": {"name": "read_file", "arguments": {"filepath": "a.py"}}}])
    led.add_tool_result("read_file", {"filepath": "a.py"}, "print(1)")
    assert led.total_tokens() == TOK.count_messages(led.messages())


def test_task_and_system_survive_forced_starvation(tmp_path):
    """50 large tool results into a 600-token budget: the task is never dropped or truncated."""
    led = _ledger(600, tmp_path)
    for i in range(50):
        args = {"filepath": f"src/mod_{i}.py"}
        led.add_assistant(f"step {i}", tool_calls=[{"function": {"name": "read_file", "arguments": args}}])
        led.add_tool_result("read_file", args, _big(400))
        led.compact()
        msgs = led.messages()
        assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You are a coding agent.")
        assert {"role": "user", "content": TASK} in msgs, "the task was dropped"
    assert led.total_tokens() <= 600


def test_tool_calls_stay_paired_with_their_results(tmp_path):
    led = _ledger(900, tmp_path)
    for i in range(12):
        args = {"filepath": f"f{i}.py"}
        led.add_assistant("", tool_calls=[{"function": {"name": "read_file", "arguments": args}}])
        led.add_tool_result("read_file", args, _big(80))
        led.compact()
        msgs = led.messages()
        for j, m in enumerate(msgs):
            if m["role"] == "tool":
                prev = [x for x in msgs[:j] if x["role"] == "assistant"]
                assert prev and prev[-1].get("tool_calls"), "a tool result without its call"
                names = [tc["function"]["name"] for tc in prev[-1]["tool_calls"]]
                assert m["tool_name"] in names
    # evicted steps survive as one-line receipts in the journal
    system = led.messages()[0]["content"]
    assert "[Tool receipt: read_file(" in system and "sha256:" in system


def test_large_output_is_spilled_with_a_preview(tmp_path):
    led = _ledger(5000, tmp_path)
    out = _big(2000)
    stored = led.add_tool_result("read_file", {"filepath": "huge.log"}, out)
    assert "[Output truncated: 2000 lines" in stored and "full text saved to" in stored
    saved = Path(stored.split("full text saved to ")[1].split(" - ")[0])
    assert saved.read_text(encoding="utf-8") == out
    assert TOK.count(stored) < TOK.count(out) / 10


def test_engine_turn_keeps_task_and_tool_calls_under_pressure(tmp_path):
    """A real turn: the model reads a 3,000-line file three times. Every request still carries the task, each tool
    result follows the call that produced it, and the requests stay within the context window."""
    from ogun_core.engine import CoreEngine
    from ogun_core.kit.tokenizer import get_tokenizer
    from ogun_core import config
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"row {i} " + "y" * 70 for i in range(3000)), encoding="utf-8")
    task = "Summarize big.txt in one sentence."
    sent = []

    class Client:
        model = "qwen2.5-coder:14b"

        def chat(self, messages, tools=None, options=None):
            sent.append([dict(m) for m in messages])
            if len(sent) <= 3:
                return {"content": "", "tool_calls": [{"function": {"name": "read_file",
                                                                   "arguments": {"filepath": str(big)}}}]}
            return {"content": "It is a list of 3000 rows.", "tool_calls": []}

    eng = CoreEngine(client=Client(), base_dir=tmp_path)
    eng.run_turn(task)
    tok = get_tokenizer("qwen2.5-coder:14b")
    assert len(sent) == 4
    for msgs in sent:
        assert {"role": "user", "content": task} in msgs
        assert tok.count_messages(msgs) < config.get_num_ctx()
        for j, m in enumerate(msgs):
            if m["role"] == "tool":
                assert msgs[j - 1]["role"] in ("assistant", "tool")
                prev = [x for x in msgs[:j] if x["role"] == "assistant"][-1]
                assert prev["tool_calls"][0]["function"]["name"] == m["tool_name"] == "read_file"
