"""Ogun dynamic plugin manager.

Handles plugin discovery, AST pre-flight validation, isolated subprocess unit testing,
hot-reloading, execution sandboxing, quarantine, and safe registry persistence.
"""
from __future__ import annotations

import ast
import concurrent.futures
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ogun_core.base_tool import BaseTool, ToolResult
from ogun_core.paths import (
    CUSTOM_TOOLS_DIR,
    OGUN_ROOT,
    PLUGINS_TESTS_DIR,
    QUARANTINE_DIR,
    REGISTRY_PATH,
)
from ogun_core.security import PathSecurityGuard

_PLACEHOLDER_COMMENT_RE = re.compile(
    r"#\s*(?:placeholder\b|todo\b|mock\b|dummy\b|stub\b|to be implemented|replace with actual|in a real implementation)",
    re.I,
)
_PLACEHOLDER_TEXT_RE = re.compile(
    r"\b(?:placeholder for|mock implementation|dummy implementation|simulated implementation|simulated posting|fake implementation|not actually implemented)\b",
    re.I,
)
_ACTION_NAME_RE = re.compile(r"^(?:post|send|publish|tweet|monitor|scrape|fetch|upload|download|delete|execute|run|create)", re.I)
_ACTION_SUCCESS_CLAIM_RE = re.compile(r"\b(?:posted to|monitoring\s+[a-z]+|published to|tweeted|uploaded to|deleted from)\b", re.I)


def normalize_plugin_name(name: str) -> str:
    """Normalize a plugin/tool name by stripping whitespace and converting hyphens to underscores."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().replace("-", "_")


_UNITTEST_RUNNER = (
    "import importlib.util, sys, unittest\n"
    "spec = importlib.util.spec_from_file_location('plugin_test_module', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "suite = unittest.defaultTestLoader.loadTestsFromModule(mod)\n"
    "result = unittest.TextTestRunner(verbosity=2).run(suite)\n"
    "if result.testsRun == 0:\n"
    "    sys.stderr.write('NO TESTS RAN: the test file defines no runnable tests.\\n')\n"
    "    sys.exit(5)\n"
    "sys.exit(0 if result.wasSuccessful() else 1)\n"
)


def plugin_test_command(test_path: Path) -> List[str]:
    """Command that really runs a plugin's tests. A unittest-style file without `unittest.main()` runs ZERO tests as
    a script and exits 0, which let a broken plugin pass; such files run through the unittest loader and zero tests
    is a failure. Plain-script tests (print PASS / non-zero exit) still run as scripts, as before."""
    try:
        code = Path(test_path).read_text(encoding="utf-8")
    except Exception:
        code = ""
    if "TestCase" in code and "__main__" not in code:
        return [sys.executable, "-c", _UNITTEST_RUNNER, str(test_path)]
    return [sys.executable, str(test_path)]


class PluginManager:
    """Dynamic lifecycle manager for Ogun plugins and tools."""

    def __init__(
        self,
        custom_tools_dir: Optional[Path] = None,
        tests_dir: Optional[Path] = None,
        quarantine_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        default_timeout: float = 30.0,
    ) -> None:
        self.base_dir = (base_dir or OGUN_ROOT).resolve()
        self.custom_tools_dir = (custom_tools_dir or (self.base_dir / "plugins" / "custom_tools")).resolve()
        self.tests_dir = (tests_dir or (self.base_dir / "plugins" / "tests")).resolve()
        self.quarantine_dir = (quarantine_dir or (self.base_dir / "plugins" / "quarantine")).resolve()
        self.registry_path = (registry_path or (self.base_dir / "plugins" / "registry.json")).resolve()
        self.default_timeout = default_timeout

        self._active_tools: Dict[str, BaseTool] = {}
        self._lock = threading.RLock()
        self.security_guard = PathSecurityGuard(base_dir=self.base_dir)

    # -------------------------------------------------------------------------
    # Registry Management
    # -------------------------------------------------------------------------
    def load_registry(self) -> Dict[str, Any]:
        """Read registry.json safely using utf-8-sig. Backs up corrupt files."""
        with self._lock:
            if not self.registry_path.exists():
                return {"tools": {}}
            try:
                raw_text = self.registry_path.read_text(encoding="utf-8-sig")
                if not raw_text.strip():
                    return {"tools": {}}
                data = json.loads(raw_text)
                if not isinstance(data, dict):
                    raise ValueError("Registry JSON root must be an object.")
            except Exception as exc:
                ts = int(time.time())
                backup_path = self.registry_path.with_name(f"registry.json.corrupt-{ts}")
                try:
                    shutil.copy2(self.registry_path, backup_path)
                except Exception:
                    pass
                print(
                    f"[WARNING] registry.json is corrupt and was backed up to {backup_path.name}; continuing with empty registry",
                    file=sys.stderr,
                )
                return {"tools": {}}

            # Normalize tool keys and filter duplicate hyphenated entries
            tools_dict = data.get("tools", {})
            if not isinstance(tools_dict, dict):
                tools_dict = {}

            normalized_tools: Dict[str, Any] = {}
            for k, v in tools_dict.items():
                norm_k = normalize_plugin_name(k)
                if not norm_k:
                    continue
                if norm_k in normalized_tools and k != norm_k:
                    continue
                if isinstance(v, dict) and "name" in v:
                    v["name"] = normalize_plugin_name(v["name"])
                normalized_tools[norm_k] = v
            data["tools"] = normalized_tools
            return data

    def save_registry(self, data: Dict[str, Any]) -> None:
        """Atomically persist registry data using a temporary file and os.replace."""
        with self._lock:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            tools_dict = data.get("tools", {})
            normalized_tools: Dict[str, Any] = {}
            for k, v in tools_dict.items():
                norm_k = normalize_plugin_name(k)
                if not norm_k:
                    continue
                if norm_k in normalized_tools and k != norm_k:
                    continue
                if isinstance(v, dict) and "name" in v:
                    v["name"] = normalize_plugin_name(v["name"])
                normalized_tools[norm_k] = v
            data["tools"] = normalized_tools

            tmp = self.registry_path.with_suffix(".tmp")
            content = json.dumps(data, indent=2)
            tmp.write_text(content, encoding="utf-8")
            for attempt in range(5):
                try:
                    os.replace(tmp, self.registry_path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05)

    # -------------------------------------------------------------------------
    # AST Pre-Flight Validation
    # -------------------------------------------------------------------------
    @staticmethod
    def validate_plugin_code(code: str) -> Tuple[bool, Optional[str]]:
        """Validate code syntax, BaseTool inheritance, and reject stubs/placeholders."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"AST parse error: {e}"

        # 1. Check for placeholder comments
        for line in code.splitlines():
            if "#" in line:
                m = _PLACEHOLDER_COMMENT_RE.search(line)
                if m:
                    return False, f"Mock/placeholder comment detected ('{m.group(0)}')."

        # 2. Check for placeholder strings in AST docstrings / constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = _PLACEHOLDER_TEXT_RE.search(node.value)
                if m:
                    return False, f"Mock/placeholder phrase detected in code ('{m.group(0)}')."

        # 3. Check for pass or NotImplementedError stubs in methods
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fname = node.name.lower()
                if fname in ("__init__",):
                    continue
                real_stmts = [
                    stmt for stmt in node.body
                    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
                ]
                if len(real_stmts) == 1 and isinstance(real_stmts[0], ast.Pass):
                    return False, f"Stub method '{node.name}' contains only 'pass'."
                if len(real_stmts) == 1 and isinstance(real_stmts[0], ast.Raise):
                    exc = real_stmts[0].exc
                    if (isinstance(exc, ast.Name) and exc.id == "NotImplementedError") or (
                        isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "NotImplementedError"
                    ):
                        return False, f"Stub method '{node.name}' raises NotImplementedError."

                if _ACTION_NAME_RE.search(fname):
                    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
                    call_names = set()
                    for c in calls:
                        if isinstance(c.func, ast.Name):
                            call_names.add(c.func.id.lower())
                        elif isinstance(c.func, ast.Attribute):
                            call_names.add(c.func.attr.lower())
                    real_ops = {
                        "get", "post", "put", "delete", "request", "urlopen", "open", "read",
                        "write", "load", "loads", "dump", "dumps", "run", "popen",
                        "getenv", "exists", "hexdigest", "digest", "encode", "decode",
                    }
                    has_real_op = bool(call_names.intersection(real_ops))
                    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
                    for r in returns:
                        for c_node in ast.walk(r):
                            if isinstance(c_node, ast.Constant) and isinstance(c_node.value, str):
                                if _ACTION_SUCCESS_CLAIM_RE.search(c_node.value) and not has_real_op:
                                    return False, f"Stub action method '{node.name}' returns a fake success claim ('{c_node.value}') without real operations."

        # 4. Check for BaseTool subclass with execute method
        has_tool_class = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                    for b in node.bases
                ]
                has_execute = any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "execute"
                    for item in node.body
                )
                if ("BaseTool" in bases or has_execute) and has_execute:
                    has_tool_class = True
                    break

        if not has_tool_class:
            return False, "Code must define a class subclassing BaseTool with an execute() method."

        return True, None

    # -------------------------------------------------------------------------
    # Subprocess Testing & Dynamic Loading
    # -------------------------------------------------------------------------
    def run_plugin_test(self, test_path: Path, timeout: float = 30.0) -> Tuple[bool, str]:
        """Run unit test in an isolated child process with full UTF-8 encoding."""
        if not test_path.exists():
            return False, f"Test file not found: {test_path}"

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        sep = ";" if sys.platform == "win32" else ":"
        paths = [str(self.base_dir)]
        if str(OGUN_ROOT) != str(self.base_dir):
            paths.append(str(OGUN_ROOT))
        existing_pp = env.get("PYTHONPATH")
        if existing_pp:
            paths.append(existing_pp)
        env["PYTHONPATH"] = sep.join(paths)

        try:
            res = subprocess.run(
                plugin_test_command(test_path),
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
            output = f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            if res.returncode == 0:
                return True, output
            return False, f"Test failed with exit code {res.returncode}:\n{output}"
        except subprocess.TimeoutExpired:
            return False, f"Test timed out after {timeout} seconds."
        except Exception as exc:
            return False, f"Subprocess execution error: {exc}"

    def load_tool_from_file(self, tool_path: Path) -> Optional[BaseTool]:
        """Dynamically load and instantiate a BaseTool subclass from a Python file."""
        if not tool_path.exists():
            return None
        module_name = f"ogun_plugin_{tool_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(tool_path))
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            tool_instance: Optional[BaseTool] = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    tool_instance = attr()
                    break

            if tool_instance:
                name = normalize_plugin_name(tool_instance.name)
                tool_instance.name = name
                with self._lock:
                    self._active_tools[name] = tool_instance
                return tool_instance
        except Exception as exc:
            print(f"[WARNING] Failed to load tool from {tool_path}: {exc}", file=sys.stderr)
            return None
        return None

    def discover_tools(self) -> Dict[str, BaseTool]:
        """Discover tools present on disk without modifying registry.json."""
        if not self.custom_tools_dir.exists():
            return {}
        for f in self.custom_tools_dir.glob("*.py"):
            if f.name.startswith("__"):
                continue
            norm_name = normalize_plugin_name(f.stem)
            if norm_name not in self._active_tools:
                self.load_tool_from_file(f)
        return dict(self._active_tools)

    def list_tools(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [tool.to_tool_schema() for tool in self._active_tools.values()]

    def get_tool(self, name: str) -> Optional[BaseTool]:
        norm = normalize_plugin_name(name)
        with self._lock:
            return self._active_tools.get(norm)

    # -------------------------------------------------------------------------
    # Safe Tool Execution
    # -------------------------------------------------------------------------
    def execute_plugin(self, name: str, timeout: Optional[float] = None, **kwargs: Any) -> ToolResult:
        """Execute a loaded tool with a timeout. Always returns ToolResult, never raises."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult.error(f"Plugin '{name}' is not registered or active.")

        timeout_sec = timeout or self.default_timeout

        def _runner() -> ToolResult:
            try:
                res = tool.execute(**kwargs)
                if isinstance(res, ToolResult):
                    return res
                if isinstance(res, dict):
                    status = res.get("status", "success")
                    output = res.get("output", res)
                    err = res.get("error_trace")
                    return ToolResult(status=status, output=output, error_trace=err)
                return ToolResult.success(output=res)
            except Exception as exc:
                return ToolResult.error(error_trace=f"Execution exception: {exc}\n{traceback.format_exc()}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_runner)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                return ToolResult.error(f"Execution timed out after {timeout_sec}s.")
            except Exception as exc:
                return ToolResult.error(f"Execution failed: {exc}")

    # -------------------------------------------------------------------------
    # Creation, Testing & Quarantine Lifecycle
    # -------------------------------------------------------------------------
    def create_or_update_plugin(
        self,
        name: str,
        code: str,
        test_code: str,
    ) -> ToolResult:
        """Create, test, hot-reload, and register a new dynamic tool."""
        norm_name = normalize_plugin_name(name)
        if not norm_name:
            return ToolResult.error("Plugin name cannot be empty.")

        # 1. Validate AST
        valid, err = self.validate_plugin_code(code)
        if not valid:
            return ToolResult.error(f"Plugin AST validation failed: {err}")

        # 2. Check paths against security guard
        tool_file = self.custom_tools_dir / f"{norm_name}.py"
        test_file = self.tests_dir / f"test_{norm_name}.py"

        try:
            self.security_guard.validate_write(tool_file)
            self.security_guard.validate_write(test_file)
        except PermissionError as pe:
            return ToolResult.error(f"Security violation: {pe}")

        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        plugins_dir = self.custom_tools_dir.parent
        plugins_init = plugins_dir / "__init__.py"
        if not plugins_init.exists():
            plugins_init.write_text("# Ogun dynamic plugins package\n", encoding="utf-8")
        tools_init = self.custom_tools_dir / "__init__.py"
        if not tools_init.exists():
            tools_init.write_text("# Ogun custom tools package\n", encoding="utf-8")

        # Keep the current version of an existing plugin: a failed UPDATE must leave the working plugin in place.
        # (Live bug: a failed update moved the working file_crc32_checksum plugin into quarantine.)
        previous_code = tool_file.read_text(encoding="utf-8") if tool_file.exists() else None
        previous_test = test_file.read_text(encoding="utf-8") if test_file.exists() else None

        def _restore_previous() -> str:
            if previous_code is None:
                return ""
            tool_file.write_text(previous_code, encoding="utf-8")
            if previous_test is not None:
                test_file.write_text(previous_test, encoding="utf-8")
            elif test_file.exists():
                test_file.unlink()
            return f"The previous working version of '{norm_name}' was kept."

        # Write files
        tool_file.write_text(code, encoding="utf-8")
        test_file.write_text(test_code, encoding="utf-8")

        # 3. Run isolated unit test
        passed, test_output = self.run_plugin_test(test_file)
        if not passed:
            # Quarantine the failed candidate; restore the previous version if this was an update
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            quarantine_tool = self.quarantine_dir / f"{norm_name}.py"
            try:
                shutil.copy2(str(tool_file), str(quarantine_tool))
                tool_file.unlink()
                if test_file.exists():
                    shutil.copy2(str(test_file), str(self.quarantine_dir / test_file.name))
                    test_file.unlink()
            except Exception:
                pass
            kept = _restore_previous()
            return ToolResult.error(
                f"Plugin unit tests failed. The new code was moved to quarantine. {kept}\n{test_output}".replace(" \n", "\n")
            )

        # 4. Hot load plugin
        tool_instance = self.load_tool_from_file(tool_file)
        if not tool_instance:
            kept = _restore_previous()
            if previous_code is not None:
                self.load_tool_from_file(tool_file)
            return ToolResult.error(f"Failed to dynamically load validated tool into memory. {kept}".strip())

        # 5. Persist to registry
        registry = self.load_registry()
        registry.setdefault("tools", {})[norm_name] = {
            "name": norm_name,
            "version": tool_instance.version,
            "description": tool_instance.description,
            "file": str(tool_file.relative_to(self.base_dir)),
            "test_file": str(test_file.relative_to(self.base_dir)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.save_registry(registry)

        return ToolResult.success(
            output={
                "message": f"Plugin '{norm_name}' created, verified, and mounted successfully.",
                "tool_schema": tool_instance.to_tool_schema(),
            }
        )

    def disable_plugin(self, name: str) -> ToolResult:
        """Disable an active tool and update registry."""
        norm_name = normalize_plugin_name(name)
        with self._lock:
            if norm_name in self._active_tools:
                del self._active_tools[norm_name]
            registry = self.load_registry()
            if norm_name in registry.get("tools", {}):
                registry["tools"][norm_name]["status"] = "disabled"
                self.save_registry(registry)
                return ToolResult.success(f"Plugin '{norm_name}' disabled.")
        return ToolResult.error(f"Plugin '{norm_name}' not found.")
