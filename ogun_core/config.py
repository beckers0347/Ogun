"""Ogun configuration and prompt definitions.

Model options, generation hyperparameters, and system prompt templates.
"""
from __future__ import annotations

import os
import sys

# Model and host configuration
MODEL_NAME = os.environ.get("OGUN_MODEL", "qwen2.5-coder:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Ollama generation hyperparameters (optimized for Qwen2.5-Coder)
TEMPERATURE = 0.1
TOP_P = 0.95
NUM_CTX = 8192
NUM_PREDICT = 3072
STOP_TOKENS = ["<|im_end|>"]
KEEP_ALIVE = os.environ.get("OGUN_KEEP_ALIVE", "2m")   # idle models must not hold GPU memory for long

# ---------------------------------------------------------------------------------------------------------------------
# Hardware allocation profiles: how many of the model's layers run on the GPU (Ollama option `num_gpu`).
# qwen2.5-coder:14b has 48 transformer blocks + 1 output layer = 49 offload units. Measured on the RTX 5070 (12 GB):
#   num_gpu=48  -> 6% of the model stays on the CPU (the output layer), ~40% SLOWER than everything on the GPU
#   num_gpu>=49 -> 100% GPU          num_gpu=24 -> 50%/50% CPU/GPU, ~8 tok/s, ~5-6 GB less VRAM
#   num_gpu=0   -> 100% CPU, ~4 tok/s, no VRAM at all
# so MAX_PERFORMANCE sends 99 ("every layer") and shows 48 as the layer count.
# ---------------------------------------------------------------------------------------------------------------------
VRAM_PROFILES = {
    "MAX_PERFORMANCE": {"num_gpu": 99, "layers": 48, "summary": "all layers on the GPU (fastest; uses ~10 GB of VRAM)"},
    "SHARED_VRAM": {"num_gpu": 24, "layers": 24, "summary": "half the layers on the GPU, half in system RAM (slower; leaves ~5-6 GB of VRAM for Unreal / Fusion 360)"},
    "CPU_ONLY": {"num_gpu": 0, "layers": 0, "summary": "no GPU at all (slowest; uses no VRAM)"},
}
DEFAULT_VRAM_PROFILE = "SHARED_VRAM"
_VRAM_ALIASES = {
    "max": "MAX_PERFORMANCE", "maximum": "MAX_PERFORMANCE", "performance": "MAX_PERFORMANCE", "max_performance": "MAX_PERFORMANCE",
    "shared": "SHARED_VRAM", "shared_vram": "SHARED_VRAM", "share": "SHARED_VRAM",
    "cpu": "CPU_ONLY", "cpu_only": "CPU_ONLY",
}


def resolve_vram_profile(name):
    """Profile key for a name/alias ('max', 'SHARED_VRAM', ...), or None when it is not a profile."""
    key = str(name or "").strip().replace("-", "_").replace(" ", "_").lower()
    if key in _VRAM_ALIASES:
        return _VRAM_ALIASES[key]
    upper = key.upper()
    return upper if upper in VRAM_PROFILES else None


def _parse_num_gpu(raw, default=24):
    """OGUN_NUM_GPU: a custom layer count. 'auto'/'all'/'-1' -> None (omit the option). Invalid -> default, with a warning."""
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in ("auto", "all", "-1"):
        return None
    try:
        value = int(text)
        if value >= 0:
            return value
    except ValueError:
        pass
    print(f"[Ogun] ignoring invalid OGUN_NUM_GPU={raw!r}; using {default}.", file=sys.stderr)
    return default


_ACTIVE = {"profile": DEFAULT_VRAM_PROFILE, "num_gpu": VRAM_PROFILES[DEFAULT_VRAM_PROFILE]["num_gpu"],
           "num_ctx": None, "num_thread": None, "hardware": None}


def _init_vram_state():
    """Initial profile: OGUN_NUM_GPU (custom count / 'auto') wins, then OGUN_VRAM_PROFILE, then SHARED_VRAM."""
    custom = os.environ.get("OGUN_NUM_GPU")
    if custom is not None and custom.strip() != "":
        value = _parse_num_gpu(custom)
        _ACTIVE.update(profile="AUTO" if value is None else f"CUSTOM({value})", num_gpu=value)
        return
    wanted = os.environ.get("OGUN_VRAM_PROFILE")
    key = resolve_vram_profile(wanted) if wanted else DEFAULT_VRAM_PROFILE
    if wanted and key is None:
        print(f"[Ogun] ignoring invalid OGUN_VRAM_PROFILE={wanted!r}; using {DEFAULT_VRAM_PROFILE}.", file=sys.stderr)
        key = DEFAULT_VRAM_PROFILE
    _ACTIVE.update(profile=key, num_gpu=VRAM_PROFILES[key]["num_gpu"])


_init_vram_state()


def get_vram_profile():
    """Name of the ACTIVE profile (MAX_PERFORMANCE / SHARED_VRAM / CPU_ONLY, or CUSTOM(n) / AUTO from the environment)."""
    return _ACTIVE["profile"]


def get_num_gpu():
    """The ACTIVE num_gpu (None = do not send the option). Read at CALL TIME: never copy this into a module constant."""
    return _ACTIVE["num_gpu"]


def set_vram_profile(name):
    """Switch profile at runtime. Returns (ok, changed, profile, num_gpu). Never raises; unknown names return ok=False."""
    key = resolve_vram_profile(name)
    if key is None:
        return False, False, _ACTIVE["profile"], _ACTIVE["num_gpu"]
    changed = _ACTIVE["profile"] != key or _ACTIVE["num_gpu"] != VRAM_PROFILES[key]["num_gpu"]
    _ACTIVE.update(profile=key, num_gpu=VRAM_PROFILES[key]["num_gpu"])
    return True, changed, key, _ACTIVE["num_gpu"]


def get_num_ctx():
    """Context window in tokens: the auto-tuned value once a hardware profile is applied, else NUM_CTX. Read at CALL TIME."""
    return _ACTIVE["num_ctx"] or NUM_CTX


def get_num_thread():
    """CPU threads for inference (None = let Ollama decide). Read at CALL TIME."""
    return _ACTIVE["num_thread"]


def get_hardware_profile():
    """The applied hardware profile dict (see ogun_core.hardware.profile_to_dict) or None."""
    return _ACTIVE["hardware"]


def apply_hardware_profile(profile):
    """Adopt an auto-tuned hardware profile (dict) as the ACTIVE runtime settings: profile name AUTO, num_gpu, ctx, threads.

    Call once at startup (before any request) and again only when the user asks for /vram auto: Ollama reloads the model
    whenever num_gpu/num_ctx change, so these values must stay constant while the model is loaded. Never raises.
    """
    global MODEL_NAME
    if not isinstance(profile, dict):
        return False
    _ACTIVE["hardware"] = dict(profile)
    _ACTIVE["profile"] = "AUTO"
    _ACTIVE["num_gpu"] = profile.get("num_gpu")
    ctx = profile.get("num_ctx")
    _ACTIVE["num_ctx"] = int(ctx) if ctx else None
    thread = profile.get("num_thread")
    _ACTIVE["num_thread"] = int(thread) if thread else None
    tag = profile.get("model_tag")
    if tag:
        MODEL_NAME = str(tag)
    return True


def set_auto_num_gpu(num_gpu):
    """Update ONLY the AUTO profile's GPU layer count (live VRAM refresh). num_ctx/threads stay as they are, because
    changing num_ctx reloads the model. No effect unless the AUTO profile is active. Returns True when applied."""
    if _ACTIVE.get("profile") != "AUTO":
        return False
    _ACTIVE["num_gpu"] = num_gpu
    if isinstance(_ACTIVE.get("hardware"), dict):
        _ACTIVE["hardware"]["num_gpu"] = num_gpu
    return True


def llm_options(**extra):
    """The Ollama `options` payload for EVERY chat/generate call, including the ACTIVE num_gpu.

    Ollama reloads the model whenever load-affecting options (num_ctx, num_gpu) differ between requests, so all call
    sites must build their options here and nowhere else.
    """
    options = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_ctx": get_num_ctx(),
        "num_predict": NUM_PREDICT,
        "stop": list(STOP_TOKENS),
    }
    num_gpu = get_num_gpu()
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    num_thread = get_num_thread()
    if num_thread:
        options["num_thread"] = num_thread
    options.update(extra)
    return options


def get_request_timeout() -> float:
    """Adaptive inference request timeout in seconds based on active hardware profile.

    Hybrid GPU/CPU and CPU-only profiles receive longer timeouts to accommodate
    slower token generation without premature aborts.
    Read at CALL TIME: never copy into a module constant.
    """
    env_override = os.environ.get("OGUN_REQUEST_TIMEOUT")
    if env_override:
        try:
            val = float(env_override)
            if val > 0:
                return val
        except ValueError:
            pass

    prof = get_vram_profile()
    num_gpu = get_num_gpu()
    hw = get_hardware_profile()
    mode = hw.get("mode") if hw else None

    # CPU-only execution is ~4 tok/s: give 900s (15 min) for large generations
    if prof == "CPU_ONLY" or num_gpu == 0 or mode == "cpu":
        return 900.0

    # Hybrid GPU/CPU execution is ~8 tok/s: give 600s (10 min)
    if prof == "SHARED_VRAM" or mode == "hybrid" or (num_gpu is not None and num_gpu < 48):
        return 600.0

    # Full GPU execution (MAX_PERFORMANCE or >= 48 layers): 300s (5 min)
    return 300.0


REQUEST_TIMEOUT = 300.0

# Tool builds (workspace tools and plugin updates): 0 = no attempt limit. They research and retry until they pass;
# press Ctrl+C to stop one. Set OGUN_BUILD_MAX_ATTEMPTS to a number to cap them again.
BUILD_MAX_ATTEMPTS = int(os.environ.get("OGUN_BUILD_MAX_ATTEMPTS", "0") or 0)

# System instructions
SYSTEM_PROMPT = """You are Ogun, an autonomous technical coding engine and developer assistant.
You specialize in writing clean, reliable, and production-ready Python code, inspecting directories,
diagnosing bugs, and authoring standalone projects and dynamic plugins.
When tool calling is required, output either native tool calls or fenced JSON tool calls.
Always ensure your code is syntactically valid and completely avoids placeholders or stubs.

### Project Routing & Directory Boundaries:
- Standalone CLI utilities / projects (e.g. dupefinder, scripts, tools):
  Save files under `workspace/<project_name>/` (e.g. `workspace/dupefinder/dupefinder.py` and `workspace/dupefinder/test_dupefinder.py`) using `write_file`.
  Run tests using `run_python_test` with the test file path.
- Ogun Dynamic Plugins:
  Use `create_plugin` (inheriting from BaseTool) ONLY when the user explicitly asks for an Ogun plugin or dynamic tool.
- Security Boundaries:
  Writes are strictly confined to `workspace/`, `plugins/custom_tools/`, and `plugins/tests/`.
  Modifying `ogun_core/`, `training/`, or engine files is strictly forbidden.

### Software Quality & Implementation Checklist:
- Requirements Completeness: Re-check all requested CLI flags (e.g. `--json`, `--min-size`), positional arguments, and calculations (e.g. wasted disk space). Every requested item must be fully implemented and covered by unit tests.
- Chunked File Reading: NEVER read entire large files into memory with `f.read()`. Always stream file I/O in binary chunks (e.g. `while chunk := f.read(65536): hasher.update(chunk)`).
- Numeric Parsing & Statistics: Never use `str.isdigit()` or regex for "is numeric"; use `float(cell.strip())` inside `try...except (ValueError, TypeError)` (handle negatives, decimals, exponents; skip empty cells). When computing multi-column statistics from a stream/CSV, accumulate all columns in a single pass over rows. Never consume an iterator and reuse it.
- Table Alignment: For table output with columns, calculate each column width as max of header length and data value lengths. Header row, separator line, and all data rows must share identical column widths and separator positions.
- Safe Path & Error Handling: Wrap file/folder access in `try...except (OSError, PermissionError)` so unreadable, locked, or nonexistent paths produce clean user errors and do not crash. If an input file or folder does not exist or is invalid, print an error to sys.stderr and exit non-zero (sys.exit(1)). NEVER exit 0 on nonexistent or invalid paths.
- Windows UTF-8 Safety: Prevent `UnicodeEncodeError` when printing non-ASCII text on Windows console. Always configure stdout encoding at startup: `if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')`.
- Test Entry Point Guard: Unit test files must always end with `if __name__ == '__main__': unittest.main()`.
- Thorough Testing: Author tests that assert real behavior, duplicate grouping by content (not directory), wasted disk space calculations ((count - 1) * file_size), boundary thresholds, and edge cases.
- Truthful Execution: Never claim files were saved or tests passed unless you executed the real tool calls (`write_file`, `run_python_test`) and they succeeded.
"""

PLUGIN_BUILDER_PROMPT = """You are an expert Python engineer authoring a dynamic plugin for Ogun.
Your response MUST contain EXACTLY TWO fenced ```python code blocks in this order:
1. The tool implementation block.
2. The unit test block.

### Quality & Operational Requirements:
- The tool MUST inherit from `ogun_core.base_tool.BaseTool`.
- Define `name` (string, snake_case), `version` (string, e.g. "1.0.0"), `description` (string), and `parameters_schema` (JSON Schema dictionary).
- Implement `execute(self, **kwargs: Any) -> ToolResult`.
- Return `ToolResult.success(output=...)` on success, or `ToolResult.error(error_trace=...)` on failure.
- KNOWN MODEL FAULT: Treat an empty string as a VALID value! Test with `if param is None:`, NEVER with truthiness like `if not param:`.
- No mock stubs, no `# TODO`, no `# mock`, no dummy returns. Implement real operational logic using Python standard libraries.
- Chunked I/O: Stream large file reads in chunks (never `f.read()` whole files into memory).
- Robust Error Handling: Catch `OSError`, `PermissionError`, and other exceptions gracefully with informative error messages.
- Windows UTF-8 Safe: Ensure output strings and print statements handle non-ASCII characters without charmap errors.

### Unit Test Requirements:
- Standalone runnable via `python plugins/tests/test_<name>.py` (no pytest required).
- Add the Ogun root to `sys.path` using `Path(__file__).resolve().parents[2]`.
- Verify execution on typical inputs, empty strings (if applicable), and edge cases.
- Compute every expected value inside the test with real code (e.g. call the same standard-library function, such as zlib.crc32 or hashlib, on the test's own input file). NEVER hand-calculate checksums, hashes, means or other numbers as constants: a guessed constant fails a correct plugin.
- Print "PASS" and exit with code 0 on success, or print error to stderr and exit with non-zero code on failure.
- Check EVERY example value the request quotes (valid ones and the invalid ones it says must fail), each with its own assertion.
- Every assertion message must name the input and the actual result, e.g. `f"execute(value={v!r}): expected error, got {res.status} {res.output!r}"`. A bare "Expected error" does not say which input failed.

### Template Skeleton:

```python
from __future__ import annotations
from typing import Any, Dict
from ogun_core.base_tool import BaseTool, ToolResult

class SampleTool(BaseTool):
    name: str = "sample_tool"
    version: str = "1.0.0"
    description: str = "A sample tool description."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "input_text": {
                "type": "string",
                "description": "Input text to process."
            }
        },
        "required": ["input_text"]
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        input_text = kwargs.get("input_text")
        if input_text is None:
            return ToolResult.error("Missing required parameter: input_text")
        try:
            # Real operational logic here
            result = input_text.upper()
            return ToolResult.success(output={"result": result})
        except Exception as exc:
            return ToolResult.error(error_trace=f"Execution failed: {exc}")
```

```python
from __future__ import annotations
import sys
from pathlib import Path

# Add Ogun root to sys.path
OGUN_ROOT = Path(__file__).resolve().parents[2]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from plugins.custom_tools.sample_tool import SampleTool

def test_sample_tool() -> None:
    tool = SampleTool()
    res = tool.execute(input_text="hello")
    assert res.status == "success", f"Expected success, got {res.status}: {res.error_trace}"
    assert res.output == {"result": "HELLO"}

    # Test empty string as valid input
    res_empty = tool.execute(input_text="")
    assert res_empty.status == "success"
    assert res_empty.output == {"result": ""}
    print("PASS")

if __name__ == "__main__":
    try:
        test_sample_tool()
        sys.exit(0)
    except AssertionError as ae:
        print(f"TEST ASSERTION FAILED: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"TEST RUNTIME ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
```
"""
