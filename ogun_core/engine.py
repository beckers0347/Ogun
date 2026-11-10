"""Ogun Core Engine.

Coordinates the headless chat loop, Ollama API interactions, bounded context window,
native and text-based tool execution, and the dynamic plugin registry.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib.error
import urllib.request

from ogun_core.base_tool import BaseTool, ToolResult
from ogun_core.codetools import (
    check_syntax,
    create_plugin,
    list_plugins,
    read_file,
    replace_in_file,
    run_python_test,
    write_file,
)
from ogun_core.config import (
    KEEP_ALIVE,
    MODEL_NAME,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_HOST,
    REQUEST_TIMEOUT,
    STOP_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    VRAM_PROFILES,
    apply_hardware_profile,
    get_hardware_profile,
    get_keep_alive,
    get_num_ctx,
    get_num_gpu,
    get_num_thread,
    get_request_timeout,
    get_vram_profile,
    llm_options,
    resolve_vram_profile,
    set_vram_profile,
)
from ogun_core.honesty import TurnEvents, sanitize_reply
from ogun_core.paths import OGUN_ROOT
from ogun_core.plugin_manager import PluginManager
from ogun_core.security import PathSecurityGuard
from ogun_core.toolcalls import parse_tool_calls

BUILTIN_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file, optionally bounded by line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to read."},
                    "start_line": {"type": "integer", "description": "Optional starting line (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Optional ending line (1-indexed)."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create and write a NEW file. Refuses to overwrite existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the new file."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace an exact block of text in an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the existing file."},
                    "old_text": {"type": "string", "description": "Exact text block to replace."},
                    "new_text": {"type": "string", "description": "New replacement text block."},
                    "count": {"type": "integer", "description": "Expected number of occurrences (default 1)."},
                },
                "required": ["filepath", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": "Verify Python syntax via AST parser on a file or code string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_or_code": {"type": "string", "description": "File path or Python source code string."},
                },
                "required": ["path_or_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_test",
            "description": "Execute a Python unit test in an isolated child process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {"type": "string", "description": "Path to the test file to run."},
                },
                "required": ["test_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plugin",
            "description": "Create, test, hot-load, and mount a new dynamic plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the plugin (snake_case)."},
                    "code": {"type": "string", "description": "Complete Python source code for the tool class."},
                    "test_code": {"type": "string", "description": "Complete Python source code for the unit test."},
                },
                "required": ["name", "code", "test_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_plugins",
            "description": "List all currently active and mounted dynamic plugins.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class OllamaClient:
    """Client for Ollama API with fallback to standard library urllib."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = MODEL_NAME,
        timeout: Optional[float] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self._custom_timeout = timeout is not None
        self.timeout = timeout if self._custom_timeout else REQUEST_TIMEOUT

    def get_effective_timeout(self) -> float:
        """Return the active timeout: explicitly passed value, or live adaptive timeout."""
        return self.timeout if self._custom_timeout else get_request_timeout()

    def check_connection(self) -> Tuple[bool, str]:
        """Verify Ollama daemon is reachable."""
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return True, "Connected to Ollama."
                return False, f"Ollama returned HTTP status {resp.status}."
        except Exception as exc:
            return (
                False,
                f"Cannot connect to Ollama daemon at {self.host} ({exc}). "
                "Ensure Ollama is running ('ollama serve').",
            )

    def unload(self) -> Dict[str, Any]:
        """Tell Ollama to drop the model from memory NOW (`{"model": ..., "keep_alive": 0}`), freeing its GPU VRAM.

        Never raises: Ollama being down or the model not being loaded is not an error here.
        """
        payload = {"model": self.model, "prompt": "", "keep_alive": 0, "stream": False}
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp.read()
            return {"unloaded": True, "model": self.model, "detail": "Model unloaded; GPU memory released."}
        except Exception as exc:
            return {"unloaded": False, "model": self.model, "detail": f"Could not unload ({exc}). Ollama may not be running."}

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Send chat request to Ollama using live adaptive timeout and optional token streaming."""
        overrides = options or {}
        options = llm_options(**overrides)
        eff_timeout = self.get_effective_timeout()
        use_stream = bool(stream or on_token is not None)

        # Try official ollama package first
        eff_keep_alive = get_keep_alive()
        try:
            import ollama

            client = ollama.Client(host=self.host, timeout=eff_timeout)
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "keep_alive": eff_keep_alive,
            }
            if tools:
                kwargs["tools"] = tools
            if use_stream:
                kwargs["stream"] = True
                stream_resp = client.chat(**kwargs)
                parts: List[str] = []
                tcalls: List[Any] = []
                for chunk in stream_resp:
                    msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
                    piece = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    if piece:
                        parts.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    tc = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                    if tc:
                        tcalls.extend(tc)
                return {
                    "content": "".join(parts),
                    "tool_calls": tcalls,
                }

            resp = client.chat(**kwargs)
            msg = resp.get("message", {})
            return {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
            }
        except ImportError:
            pass
        except Exception as exc:
            # Re-raise timeout specifically
            if "timeout" in str(exc).lower():
                raise TimeoutError(str(exc))
            raise

        # Fallback to stdlib urllib
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "keep_alive": eff_keep_alive,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data.get("message", {})
                return {
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                }
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise


class ContextManager:
    """Manages conversational history within context token boundaries."""

    def __init__(self, max_tokens: Optional[int] = None, budget_ratio: float = 0.85) -> None:
        if max_tokens is None:
            max_tokens = get_num_ctx()      # live: follows the auto-tuned context window
        self.max_tokens = int(max_tokens * budget_ratio)

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total_chars = 0
        for m in messages:
            content = m.get("content") or ""
            total_chars += len(content)
            for tc in m.get("tool_calls", []):
                total_chars += len(json.dumps(tc))
        return total_chars // 4

    def prune_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep system message and drop oldest non-system turns if over budget."""
        if not messages:
            return messages

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        turns = messages[1:] if system_msg else messages[:]

        while turns and self.estimate_tokens(([system_msg] if system_msg else []) + turns) > self.max_tokens:
            turns.pop(0)

        return ([system_msg] if system_msg else []) + turns


class CoreEngine:
    """Ogun Core Engine handling execution turns and tool dispatches."""

    def __init__(
        self,
        client: Optional[Any] = None,
        plugin_manager: Optional[PluginManager] = None,
        security_guard: Optional[PathSecurityGuard] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = (base_dir or OGUN_ROOT).resolve()
        self.client = client or OllamaClient()
        self.plugin_manager = plugin_manager or PluginManager(base_dir=self.base_dir)
        self.security_guard = security_guard or PathSecurityGuard(base_dir=self.base_dir)
        self.context_manager = ContextManager()

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.last_turn_status: Dict[str, Any] = {
            "status": "success",
            "task": "",
            "error": "",
        }

    def _get_active_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = list(BUILTIN_TOOL_SCHEMAS)
        if self.plugin_manager:
            schemas.extend(self.plugin_manager.list_tools())
        return schemas

    def _get_registered_tool_names(self) -> Set[str]:
        names = {s["function"]["name"] for s in BUILTIN_TOOL_SCHEMAS}
        if self.plugin_manager:
            names.update(self.plugin_manager._active_tools.keys())
        return names

    def dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        turn_events: Optional[TurnEvents] = None,
    ) -> str:
        """Dispatch a tool call to built-in code tools or dynamic plugins."""
        try:
            if name == "read_file":
                return read_file(
                    filepath=args.get("filepath", ""),
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                    base_dir=self.base_dir,
                )
            elif name == "write_file":
                res = write_file(
                    filepath=args.get("filepath", ""),
                    content=args.get("content", ""),
                    security_guard=self.security_guard,
                )
                if turn_events is not None and "Successfully created" in res:
                    try:
                        target = self.security_guard.validate_write(args.get("filepath", ""))
                        if target.exists():
                            turn_events.add_file_written(target)
                    except Exception:
                        pass
                return res
            elif name == "replace_in_file":
                res = replace_in_file(
                    filepath=args.get("filepath", ""),
                    old_text=args.get("old_text", ""),
                    new_text=args.get("new_text", ""),
                    count=int(args.get("count", 1)),
                    security_guard=self.security_guard,
                )
                if turn_events is not None and "Successfully updated" in res:
                    try:
                        target = self.security_guard.validate_write(args.get("filepath", ""))
                        if target.exists():
                            turn_events.add_file_written(target)
                    except Exception:
                        pass
                return res
            elif name == "check_syntax":
                return check_syntax(
                    path_or_code=args.get("path_or_code", ""),
                    base_dir=self.base_dir,
                )
            elif name == "run_python_test":
                res = run_python_test(
                    test_path=args.get("test_path", ""),
                    base_dir=self.base_dir,
                )
                if turn_events is not None:
                    passed = "TEST PASS" in res
                    turn_events.add_test_execution(
                        test_path=args.get("test_path", ""),
                        passed=passed,
                        output=res,
                    )
                return res
            elif name == "create_plugin":
                res = create_plugin(
                    name=args.get("name", ""),
                    code=args.get("code", ""),
                    test_code=args.get("test_code", ""),
                    plugin_manager=self.plugin_manager,
                )
                res_str = json.dumps(res) if not isinstance(res, str) else res
                if turn_events is not None:
                    try:
                        data = json.loads(res_str)
                        if isinstance(data, dict) and data.get("status") == "success":
                            norm = args.get("name", "").strip().replace("-", "_")
                            tool_p = self.base_dir / "plugins" / "custom_tools" / f"{norm}.py"
                            test_p = self.base_dir / "plugins" / "tests" / f"test_{norm}.py"
                            if tool_p.exists():
                                turn_events.add_file_written(tool_p)
                            if test_p.exists():
                                turn_events.add_file_written(test_p)
                            turn_events.add_test_execution(str(test_p), passed=True, output="PASS")
                    except Exception:
                        pass
                return res_str
            elif name == "list_plugins":
                return json.dumps(list_plugins(self.plugin_manager))
            else:
                # Dynamic plugin execution
                res = self.plugin_manager.execute_plugin(name=name, **args)
                return json.dumps(res)
        except Exception as exc:
            return f"Error executing tool '{name}': {exc}"

    def run_turn(self, user_text: str, max_rounds: int = 10) -> str:
        """Run an agentic conversational turn with tool execution loop and honesty verification."""
        turn_events = TurnEvents()

        # Handle short follow-up resumption after failure/timeout
        continuation_words = {"continue", "go on", "try again", "keep going", "retry", "proceed", "resume", "carry on"}
        clean_user = user_text.strip().lower().rstrip(".?!")
        effective_user_text = user_text

        if clean_user in continuation_words and self.last_turn_status.get("status") in ("timeout", "error", "failed"):
            prev_task = self.last_turn_status.get("task", "")
            prev_error = self.last_turn_status.get("error", "")
            effective_user_text = (
                f"Resume and continue the task: '{prev_task}'.\n"
                f"Previous status or failure context: {prev_error}.\n"
                "Please execute real tool calls (write_file, run_python_test) to complete and verify the implementation. "
                "Do not output plain prose claims without executing the tools."
            )

        # A request to CHANGE an existing workspace tool or plugin updates it instead of starting something new.
        from ogun_core.standalone_builder import detect_change_target
        # (No routing while a plugin build is running: its own prompts mention the plugin and must reach the model.)
        _target = None if (clean_user in continuation_words or getattr(self, "_plugin_build_active", False)) \
            else detect_change_target(effective_user_text)
        if _target and _target[0] == "workspace_tool":
            from ogun_core.standalone_builder import update_standalone_project
            try:
                from ogun_core.config import BUILD_MAX_ATTEMPTS
                success, final_msg, details = update_standalone_project(self, _target[1], effective_user_text,
                                                                        max_attempts=BUILD_MAX_ATTEMPTS)
            except TimeoutError:
                success, final_msg, details = False, "Model request timed out while updating the tool. Say 'continue' to retry.", {"status": "timeout"}
            self.last_turn_status = {
                "status": "success" if success else ("timeout" if details.get("status") == "timeout" else "failed"),
                "task": user_text,
                "error": "" if success else details.get("error", "Update failed"),
            }
            self.messages.append({"role": "assistant", "content": final_msg})
            return final_msg
        if _target and _target[0] == "plugin":
            from ogun_core.paths import CUSTOM_TOOLS_DIR, PLUGINS_TESTS_DIR
            _pname = _target[1]
            try:
                _pcode = (CUSTOM_TOOLS_DIR / f"{_pname}.py").read_text(encoding="utf-8")
            except Exception:
                _pcode = "(could not read the current code)"
            try:
                _ptest = (PLUGINS_TESTS_DIR / f"test_{_pname}.py").read_text(encoding="utf-8")
            except Exception:
                _ptest = "(no existing test file)"
            # Live bug: asking the chat model to call create_plugin itself ran 10 rounds without ever calling it.
            # The staged plugin builder (write -> test -> mount, with failure feedback) does the update instead.
            update_request = (
                f"UPDATE the existing plugin `{_pname}` (keep `name: str = \"{_pname}\"`). Start from its current code "
                "below, keep everything that works and apply only the requested change. Output the COMPLETE updated "
                "tool code and the COMPLETE updated test.\n\n"
                f"=== CURRENT plugins/custom_tools/{_pname}.py ===\n{_pcode}\n"
                f"=== CURRENT plugins/tests/test_{_pname}.py ===\n{_ptest}\n"
                f"=== REQUESTED CHANGE ===\n{user_text}\n=== END ==="
            )
            from ogun_core.plugin_builder import build_plugin_staged
            self._plugin_build_active = True
            saved_messages = list(self.messages)
            try:
                from ogun_core.config import BUILD_MAX_ATTEMPTS
                ok, final_msg, _ = build_plugin_staged(self, self.plugin_manager, update_request, force_name=_pname,
                                                       max_attempts=BUILD_MAX_ATTEMPTS)
            except TimeoutError:
                ok, final_msg = False, f"Model request timed out while updating the '{_pname}' plugin; it was not changed."
            finally:
                self._plugin_build_active = False
                self.messages = saved_messages
            if not ok:
                final_msg = (f"The '{_pname}' plugin was NOT updated; its previous working version is unchanged.\n\n"
                             f"{final_msg}")
            self.last_turn_status = {"status": "success" if ok else "failed", "task": user_text,
                                     "error": "" if ok else final_msg[:300]}
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": final_msg})
            return final_msg

        # Check if standalone CLI project request
        from ogun_core.quality import classify_project_request
        if classify_project_request(effective_user_text) == "standalone" and clean_user not in continuation_words:
            from ogun_core.standalone_builder import build_standalone_project_staged
            try:
                from ogun_core.config import BUILD_MAX_ATTEMPTS
                success, final_msg, details = build_standalone_project_staged(self, effective_user_text,
                                                                              max_attempts=BUILD_MAX_ATTEMPTS)
                is_timeout = "timed out" in final_msg.lower() or details.get("status") == "timeout"
                self.last_turn_status = {
                    "status": "success" if success else ("timeout" if is_timeout else "failed"),
                    "task": user_text,
                    "error": "" if success else details.get("error", "Tests failed"),
                }
                self.messages.append({"role": "assistant", "content": final_msg})
                return final_msg
            except TimeoutError:
                eff_timeout = (
                    getattr(self.client, "get_effective_timeout", lambda: 300.0)()
                    if hasattr(self.client, "get_effective_timeout")
                    else getattr(self.client, "timeout", 300.0)
                )
                err_msg = (
                    f"Model request timed out after {eff_timeout:.0f} seconds. Ollama may be loading weights or overloaded. "
                    "Your progress has been preserved; you can say 'continue' to resume."
                )
                self.messages.append({"role": "assistant", "content": err_msg})
                self.last_turn_status = {
                    "status": "timeout",
                    "task": user_text,
                    "error": f"Request timed out after {eff_timeout:.0f}s",
                }
                return err_msg

        self.messages.append({"role": "user", "content": effective_user_text})
        self.messages = self.context_manager.prune_history(self.messages)

        tool_schemas = self._get_active_tool_schemas()
        registered_names = self._get_registered_tool_names()

        for round_idx in range(max_rounds):
            try:
                response = self.client.chat(self.messages, tools=tool_schemas)
            except TimeoutError:
                eff_timeout = (
                    getattr(self.client, "get_effective_timeout", lambda: 300.0)()
                    if hasattr(self.client, "get_effective_timeout")
                    else getattr(self.client, "timeout", 300.0)
                )
                err_msg = (
                    f"Model request timed out after {eff_timeout:.0f} seconds. Ollama may be loading weights or overloaded. "
                    "Your progress has been preserved; you can say 'continue' to resume."
                )
                self.messages.append({"role": "assistant", "content": err_msg})
                self.last_turn_status = {
                    "status": "timeout",
                    "task": user_text if clean_user not in continuation_words else self.last_turn_status.get("task", user_text),
                    "error": f"Request timed out after {eff_timeout:.0f}s",
                }
                return err_msg
            except Exception as exc:
                err_msg = f"Inference error: {exc}"
                self.messages.append({"role": "assistant", "content": err_msg})
                self.last_turn_status = {
                    "status": "error",
                    "task": user_text if clean_user not in continuation_words else self.last_turn_status.get("task", user_text),
                    "error": str(exc),
                }
                return err_msg

            content = response.get("content", "")
            raw_tool_calls = response.get("tool_calls", [])

            # Extract tool calls: native first, then text parser
            tool_calls: List[Dict[str, Any]] = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    tname = fn.get("name")
                    if tname and tname in registered_names:
                        targs = fn.get("arguments", {})
                        if isinstance(targs, str):
                            try:
                                targs = json.loads(targs)
                            except Exception:
                                targs = {"raw": targs}
                        tool_calls.append({"name": tname, "arguments": targs})

            if not tool_calls and content:
                tool_calls = parse_tool_calls(content, registered_names)

            # If no tool calls, verify honesty and return visible text
            if not tool_calls:
                raw_visible = content.strip() if content else "(No response generated)"
                sanitized, intercepted = sanitize_reply(raw_visible, turn_events, self.base_dir)
                if intercepted:
                    self.last_turn_status = {
                        "status": "failed",
                        "task": user_text if clean_user not in continuation_words else self.last_turn_status.get("task", user_text),
                        "error": "Model produced unverified claims without tool execution",
                    }
                else:
                    self.last_turn_status = {
                        "status": "success",
                        "task": user_text if clean_user not in continuation_words else self.last_turn_status.get("task", user_text),
                        "error": "",
                    }
                self.messages.append({"role": "assistant", "content": sanitized})
                return sanitized

            # Execute tool calls
            self.messages.append({"role": "assistant", "content": content or ""})
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                result = self.dispatch_tool(name, args, turn_events=turn_events)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                    }
                )

            # Prune context before next round
            self.messages = self.context_manager.prune_history(self.messages)

        files_str = ", ".join(f"`{p.name}`" for p in turn_events.files_written if p.exists()) if turn_events.files_written else "None"
        last_test = turn_events.tests_executed[-1] if turn_events.tests_executed else None
        test_info = ""
        if last_test:
            status = "PASSED" if last_test.get("passed") else "FAILED"
            test_info = f"\n- Last test ({last_test.get('test_path', '')}): {status}\n```text\n{last_test.get('output', '').strip()[:400]}\n```"
        final_msg = (
            f"Reached maximum tool execution rounds ({max_rounds}).\n\n"
            f"### Turn Status Summary:\n"
            f"- Files written: {files_str}\n"
            f"- Tests executed: {len(turn_events.tests_executed)}{test_info}\n\n"
            "Task is incomplete. Review the files and errors above before proceeding."
        )
        sanitized, _ = sanitize_reply(final_msg, turn_events, self.base_dir)
        self.last_turn_status = {
            "status": "failed",
            "task": user_text if clean_user not in continuation_words else self.last_turn_status.get("task", user_text),
            "error": "Max execution rounds reached",
        }
        self.messages.append({"role": "assistant", "content": sanitized})
        return sanitized


def unload_model(model: str = MODEL_NAME, host: str = OLLAMA_HOST) -> Dict[str, Any]:
    """Free the GPU: send {"model": MODEL_NAME, "keep_alive": 0} to Ollama. Safe to call any time, any number of times."""
    return OllamaClient(host=host, model=model).unload()


# ---------------------------------------------------------------------------------------------------------------------
# /vram commands (interactive chat)
# ---------------------------------------------------------------------------------------------------------------------
import re as _re

_VRAM_CMD = _re.compile(r"^\s*/vram(?:\s+(\S+))?\s*$", _re.IGNORECASE)
VRAM_HELP = (
    "Usage: /vram shared (half the layers on the GPU, leaves ~5-6 GB VRAM for Unreal / Fusion 360) | "
    "/vram max (all layers on the GPU, fastest) | /vram cpu (no GPU) | /vram free (unload the model now) | /vram status"
)


def is_vram_command(text: str) -> bool:
    """True only for '/vram' optionally followed by ONE word, so ordinary sentences are never hijacked."""
    return bool(_VRAM_CMD.match(text or ""))


def ollama_ps(host: str = OLLAMA_HOST) -> List[Dict[str, Any]]:
    """Models Ollama currently holds in memory (GET /api/ps). Never raises: returns [] when Ollama is unreachable."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/ps", timeout=5.0) as resp:
            return list(json.loads(resp.read().decode("utf-8")).get("models", []))
    except Exception:
        return []


def _describe_loaded(models: List[Dict[str, Any]]) -> str:
    if not models:
        return "No model is loaded right now (GPU memory is free)."
    parts = []
    for m in models:
        size, in_vram = float(m.get("size") or 0), float(m.get("size_vram") or 0)
        gpu = round(100 * in_vram / size) if size else 0
        parts.append(f"{m.get('name', '?')}: {gpu}% on the GPU ({in_vram / 2**30:.1f} of {size / 2**30:.1f} GB in VRAM)")
    return "Loaded: " + "; ".join(parts) + "."


def _vram_auto(model: str, host: str) -> str:
    """/vram auto: probe this machine again, tune against what is free RIGHT NOW (memory held by the loaded model counts as free), apply."""
    from ogun_core import hardware
    try:
        hw = hardware.probe_hardware()
        oll = hardware.query_ollama(host, family=str(model).split(":")[0])
        hw = hardware.adjust_for_resident_models(hw, oll["resident_vram_gb"], oll["resident_ram_gb"])
        prof = hardware.auto_configure(hw, installed=oll["installed"] or None, overrides={"configured_model": model})
    except Exception as exc:  # never break the chat loop over a probe problem
        return f"Hardware auto-tune failed ({exc}); settings unchanged."
    before = (get_vram_profile(), get_num_gpu(), get_num_ctx())
    set_active = apply_hardware_profile(hardware.profile_to_dict(prof))
    if not set_active:
        return "Hardware auto-tune produced no usable profile; settings unchanged."
    after = (get_vram_profile(), get_num_gpu(), get_num_ctx())
    notes = " ".join(prof.warnings[:2])
    if before == after:
        return f"Already using the auto-tuned settings: {prof.model_tag}, mode {prof.mode}, num_gpu={prof.num_gpu}, num_ctx={prof.num_ctx}. {notes}".strip()
    result = unload_model(model=model, host=host)
    note = "The model was unloaded" if result["unloaded"] else "Ollama could not be reached to unload it"
    return (f"Auto-tuned: {prof.model_tag} [{prof.quant}], mode {prof.mode}, {prof.gpu_units}/{prof.total_units} layers on GPU "
            f"(num_gpu={prof.num_gpu}), num_ctx={prof.num_ctx}, num_thread={prof.num_thread}. {note}; the next reply reloads "
            f"the model with these settings. {notes}").strip()


def handle_vram_command(text: str, model: str = MODEL_NAME, host: str = OLLAMA_HOST) -> Optional[str]:
    """Run a /vram command. Returns the reply to print, or None when `text` is not a /vram command (never raises)."""
    match = _VRAM_CMD.match(text or "")
    if not match:
        return None
    arg = (match.group(1) or "status").strip().lower()

    if arg == "free":
        result = unload_model(model=model, host=host)
        if result["unloaded"]:
            return "Model unloaded: its GPU memory is free now. The next reply reloads it (about 40 seconds)."
        return f"Could not unload the model: {result['detail']}"

    if arg == "auto":
        return _vram_auto(model, host)

    if arg == "status":
        profile, num_gpu = get_vram_profile(), get_num_gpu()
        shown = "automatic" if num_gpu is None else f"num_gpu={num_gpu}"
        info = VRAM_PROFILES.get(profile, {}).get("summary", "")
        hw_prof = get_hardware_profile() if profile == "AUTO" else None
        if hw_prof:
            info = (f"auto-tuned: {hw_prof.get('model_tag')} [{hw_prof.get('mode')}], "
                    f"{hw_prof.get('gpu_units')}/{hw_prof.get('total_units')} layers on GPU, "
                    f"num_ctx={hw_prof.get('num_ctx')}, num_thread={hw_prof.get('num_thread')}")
        return (f"VRAM profile: {profile} ({shown}){' - ' + info if info else ''}. keep_alive={KEEP_ALIVE}. "
                f"{_describe_loaded(ollama_ps(host))}")

    key = resolve_vram_profile(arg)
    if key is None:
        return f"Unknown /vram option '{arg}'. {VRAM_HELP}"
    ok, changed, profile, num_gpu = set_vram_profile(key)
    summary = VRAM_PROFILES[profile]["summary"]
    if not changed:
        return f"Already using {profile}: {summary}."
    result = unload_model(model=model, host=host)       # a changed split forces a reload anyway: free the VRAM right now
    note = "The model was unloaded" if result["unloaded"] else "Ollama could not be reached to unload it"
    return (f"Switched to {profile} (num_gpu={num_gpu}): {summary}. {note}; your next reply takes about 40 seconds "
            "while the model reloads with the new split.")
