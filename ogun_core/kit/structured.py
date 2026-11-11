"""Schema-constrained model output: ask for JSON that matches a schema, validate it, retry once on rejection.

Ollama turns a JSON Schema passed as `format` into a decoding grammar, so a compliant model cannot emit malformed JSON.
The result is still validated (enum / required / types), because the grammar alone does not enforce every keyword.

    ask = lambda prompt, fmt, sampling, num_predict: <model call returning text>
    data = ask_json(ask, prompt, schema)           # dict, or None after a rejected retry
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from .sampling import profile
from .schema_check import validate

AskFn = Callable[[str, Dict[str, Any], Dict[str, float], int], str]


def parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads((text or "").strip())
    except ValueError:
        return None


def ask_json(ask: AskFn, prompt: str, schema: Dict[str, Any], phase: str = "verdict", num_predict: int = 600,
             retries: int = 1) -> Optional[Dict[str, Any]]:
    """The validated object, or None when every try was rejected. A rejection is fed back once, precisely."""
    attempt_prompt = prompt
    for _ in range(retries + 1):
        reply = ask(attempt_prompt, schema, profile(phase), num_predict)
        data = parse_json(reply)
        problems = ["the reply is not valid JSON"] if data is None else validate(data, schema)
        if not problems:
            return data
        attempt_prompt = (prompt + "\n\nYour previous reply was rejected: " + "; ".join(problems[:5])
                          + ". Reply again with JSON that matches the schema exactly.")
    return None
