"""A small JSON-Schema validator (no dependencies) for tool arguments and structured model output.

Covers what tool and output schemas use: type (incl. lists of types and 'integer' vs bool), required, properties,
additionalProperties (false), enum, items, minItems/maxItems, minimum/maximum, minLength. Unknown keywords are ignored.

    problems = validate({"filepath": 3}, schema)  ->  ["filepath: expected string, got integer"]
"""
from __future__ import annotations

from typing import Any, Dict, List

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _type_name(v: Any) -> str:
    for name in ("boolean", "integer", "number", "string", "array", "object", "null"):
        if _TYPES[name](v):
            return name
    return type(v).__name__


def validate(value: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
    """Problems with `value` against `schema` ([] when valid). Paths name the offending field."""
    if not isinstance(schema, dict) or not schema:
        return []
    where = path or "value"
    problems: List[str] = []
    t = schema.get("type")
    if t:
        allowed = t if isinstance(t, list) else [t]
        if not any(_TYPES.get(a, lambda _v: True)(value) for a in allowed):
            return [f"{where}: expected {' or '.join(allowed)}, got {_type_name(value)}"]
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{where}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{(path + '.') if path else ''}{key}: required but missing")
        for key, sub in props.items():
            if key in value:
                problems += validate(value[key], sub, f"{path + '.' if path else ''}{key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    problems.append(f"{(path + '.') if path else ''}{key}: not an allowed argument "
                                    f"(allowed: {', '.join(props) or 'none'})")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{where}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{where}: at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                problems += validate(item, schema["items"], f"{where}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{where}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{where}: must be <= {schema['maximum']}")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        problems.append(f"{where}: must not be empty")
    return problems
