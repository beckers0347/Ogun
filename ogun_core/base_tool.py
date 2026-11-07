"""Ogun BaseTool interface and contract for all dynamic tools and plugins.

Every dynamic tool authored by the AI or user must inherit from BaseTool,
define required metadata, and implement execute(**kwargs) returning a ToolResult:
{
    "status": "success" | "error",
    "output": Any,
    "error_trace": str | None
}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class _ClassOrValue:
    """`ToolResult.success(...)` builds a result; `result.success` on a result is True/False (`result.error` is the
    error text or None). Live bug: generated plugin tests asserted `result.success is True`, which compared a bound
    method with True, so every correct plugin failed its own test and was rolled back."""

    def __init__(self, factory, value):
        self._factory, self._value = factory, value

    def __get__(self, obj, owner):
        if obj is None:
            return self._factory.__get__(owner, owner)
        return self._value(obj)


class ToolResult(dict):
    """Standardized result dictionary returned by all tool executions."""

    def __init__(
        self,
        status: Optional[str] = None,
        output: Any = None,
        error_trace: Optional[str] = None,
        **extra: Any,
    ) -> None:
        # Models naturally write ToolResult(result=...) or ToolResult(output=...) without a status (live: a plugin
        # build failed every attempt with "missing 1 required positional argument: 'status'"). Accept those forms:
        # `result` is `output`, `error` is `error_trace`, and a missing status follows from whether there is an error.
        if output is None and "result" in extra:
            output = extra.pop("result")
        if error_trace is None and extra.get("error") is not None:
            error_trace = str(extra.pop("error"))
        if status is None and "success" in extra:
            # ToolResult(success=False, output="No text provided") - also written by generated plugins
            ok = bool(extra.pop("success"))
            status = "success" if ok else "error"
            if not ok and error_trace is None:
                error_trace = str(output) if output is not None else "failed"
        if status is None:
            status = "error" if error_trace else "success"
        if status not in ("success", "error"):
            raise ValueError(f"ToolResult status must be 'success' or 'error', got: {status!r}")
        super().__init__(
            status=status,
            output=output,
            error_trace=error_trace,
            **extra,
        )

    @property
    def status(self) -> str:
        return self["status"]

    @property
    def output(self) -> Any:
        return self["output"]

    @property
    def error_trace(self) -> Optional[str]:
        return self["error_trace"]

    def _make_success(cls, output: Any = None, **extra: Any) -> "ToolResult":
        return cls(status="success", output=output, error_trace=None, **extra)

    def _make_error(cls, error_trace: str, output: Any = None, **extra: Any) -> "ToolResult":
        return cls(status="error", output=output, error_trace=str(error_trace), **extra)

    success = _ClassOrValue(classmethod(_make_success), lambda r: r.get("status") == "success")
    error = _ClassOrValue(classmethod(_make_error), lambda r: r.get("error_trace") if r.get("status") == "error" else None)


class BaseTool(ABC):
    """Abstract base class that all dynamic tools must implement."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    parameters_schema: Union[Dict[str, Any], Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("name"):
            self.name = kwargs["name"]
        elif not getattr(self, "name", None):
            self.name = self.__class__.__name__
        if kwargs.get("version"):
            self.version = kwargs["version"]
        if kwargs.get("description"):
            self.description = kwargs["description"]
        if kwargs.get("parameters_schema"):
            self.parameters_schema = kwargs["parameters_schema"]

    def execute(self, **kwargs: Any) -> Union[ToolResult, Dict[str, Any]]:
        """Execute the tool with keyword arguments.

        Must return a dict or ToolResult with 'status', 'output', and 'error_trace'.
        """
        # Action-based dispatch if an 'action' keyword is passed
        action = kwargs.get("action")
        if action and hasattr(self, action) and callable(getattr(self, action)):
            method_args = {k: v for k, v in kwargs.items() if k != "action"}
            try:
                res = getattr(self, action)(**method_args)
                return ToolResult.success(output=res)
            except Exception as exc:
                return ToolResult.error(error_trace=f"Error executing {action}: {exc}")

        # Candidate action method fallback
        for candidate in ("run", "handle", "perform", "process", "post", "monitor"):
            if hasattr(self, candidate) and callable(getattr(self, candidate)) and candidate != "execute":
                try:
                    res = getattr(self, candidate)(**kwargs)
                    return ToolResult.success(output=res)
                except Exception as exc:
                    return ToolResult.error(error_trace=f"Error executing {candidate}: {exc}")

        return ToolResult.success(output=f"{self.name} executed successfully with arguments: {kwargs}")

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Normalize parameters_schema from either a dict or Pydantic model."""
        schema = getattr(self, "parameters_schema", None)
        if schema is None:
            return {"type": "object", "properties": {}, "required": []}
        # Support Pydantic v2 model_json_schema()
        if hasattr(schema, "model_json_schema") and callable(schema.model_json_schema):
            return schema.model_json_schema()
        # Support Pydantic v1 schema()
        if hasattr(schema, "schema") and callable(schema.schema):
            return schema.schema()
        if isinstance(schema, dict):
            res = dict(schema)
            res.setdefault("type", "object")
            res.setdefault("properties", {})
            return res
        return {"type": "object", "properties": {}, "required": []}

    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert the tool definition to the OpenAI/Ollama function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} version={self.version!r}>"
