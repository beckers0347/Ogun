"""Exact token counts for the local Ollama model, built from the model file itself.

Ollama has no tokenize endpoint, but its GGUF model file carries the full vocabulary and BPE merges. This reads them
(metadata only: the weights are never touched), builds a byte-level BPE tokenizer with the `tokenizers` package, and
caches it as tokenizer.json next to the other kit caches, so later loads take milliseconds.

    tok = get_tokenizer("qwen2.5-coder:14b")
    tok.count("some text")             # exact
    tok.count_messages(messages)       # exact for the chat template's message framing (ChatML)
    tok.exact                          # False only when the model file or `tokenizers` is unavailable

The fallback (no model file / no `tokenizers`) estimates 1 token per 3.3 characters and says so via `exact`.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Qwen2 / GPT-4-style pre-tokenizer split (as in the model's upstream tokenizer.json).
_QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
                r"\s+(?!\S)|\s+")
_CACHE: Dict[str, "Tokenizer"] = {}


def models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


def model_blob(model: str) -> Optional[Path]:
    """The GGUF blob of an Ollama model tag such as 'qwen2.5-coder:14b' (None when not found)."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["library"] + parts
    manifest = models_dir() / "manifests" / "registry.ollama.ai" / Path(*parts) / tag
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for layer in data.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            blob = models_dir() / "blobs" / layer["digest"].replace(":", "-")
            return blob if blob.exists() else None
    return None


def read_gguf_metadata(path: Path, wanted: Iterable[str]) -> Dict[str, Any]:
    """Selected metadata keys from a GGUF file (stops after the metadata block)."""
    wanted = set(wanted)
    out: Dict[str, Any] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        _tensors, kv_count = struct.unpack("<QQ", f.read(16))

        def read_str() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

        def read_value(t: int, keep: bool):
            if t == 8:
                if keep:
                    return read_str()
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
                return None
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                if et == 8:
                    if keep:
                        return [read_str() for _ in range(count)]
                    for _ in range(count):
                        (n,) = struct.unpack("<Q", f.read(8))
                        f.seek(n, 1)
                    return None
                size = struct.calcsize(scalar[et])
                if keep:
                    raw = f.read(size * count)
                    return list(struct.unpack("<" + scalar[et][1] * count, raw))
                f.seek(size * count, 1)
                return None
            fmt = scalar[t]
            v = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            return v if keep else None

        for _ in range(kv_count):
            key = read_str()
            (t,) = struct.unpack("<I", f.read(4))
            keep = key in wanted
            v = read_value(t, keep)
            if keep:
                out[key] = v
                if len(out) == len(wanted):
                    break
    return out


def _cache_dir() -> Path:
    base = os.environ.get("KIT_CACHE_DIR") or os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    d = Path(base) / "kit_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_tokenizer_json(blob: Path) -> str:
    """tokenizer.json text for a GGUF BPE model ('gpt2'-style byte-level, e.g. Qwen2)."""
    meta = read_gguf_metadata(blob, ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
                                     "tokenizer.ggml.merges", "tokenizer.ggml.token_type"])
    if meta.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError(f"unsupported tokenizer model {meta.get('tokenizer.ggml.model')!r}")
    tokens: List[str] = meta["tokenizer.ggml.tokens"]
    types: List[int] = meta.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    vocab = {tok: i for i, tok in enumerate(tokens)}
    added = [{"id": i, "content": tok, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
              "special": True} for i, (tok, ty) in enumerate(zip(tokens, types)) if ty in (3, 4)]
    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": _QWEN2_SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": meta["tokenizer.ggml.merges"]},
    }
    return json.dumps(spec)


class Tokenizer:
    def __init__(self, model: str) -> None:
        self.model = model
        self.exact = False
        self._tok = None
        blob = model_blob(model)
        if blob is None:
            return
        try:
            from tokenizers import Tokenizer as _HF
        except ImportError:
            return
        key = hashlib.sha256(str(blob.name).encode()).hexdigest()[:16]
        cached = _cache_dir() / f"tokenizer-{key}.json"
        try:
            if not cached.exists():
                tmp = cached.with_suffix(".tmp")
                tmp.write_text(build_tokenizer_json(blob), encoding="utf-8")
                os.replace(tmp, cached)
            self._tok = _HF.from_file(str(cached))
            self.exact = True
        except Exception:  # noqa: BLE001 - counting must never break the caller
            self._tok = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return max(1, int(len(text) / 3.3 + 0.5))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Tokens of the messages as the ChatML template frames them:
        <|im_start|>{role}\\n{content}<|im_end|>\\n per message (tool calls serialized as <tool_call> JSON)."""
        total = 0
        for m in messages:
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                body += "\n<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}) \
                        + "\n</tool_call>"
            total += self.count(f"<|im_start|>{m.get('role', 'user')}\n{body}<|im_end|>\n")
        return total


def get_tokenizer(model: str) -> Tokenizer:
    """Cached per process."""
    tok = _CACHE.get(model)
    if tok is None:
        tok = _CACHE[model] = Tokenizer(model)
    return tok
