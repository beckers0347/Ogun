"""Understand loosely worded plugin requests: is this a NEW plugin, or a change to one the app already has (and which)?

Live 2026-11-10, six plain-language requests through chat, none worked:
- "make a temperature converter plugin" / "i need something that turns a title into a url slug, as a plugin" went to the
  general tool loop, which mounted plugins without ToolResult, schema or a working test;
- "the roman numeral plugin should accept lowercase too", "the plugin that counts days between dates should also ...",
  "can the temperature plugin do kelvin as well" were not seen as updates (no change verb, or no exact plugin name).

resolve_plugin_request(text, plugins_dir) -> ("build", None) | ("update", name) | None
A definite reference ("the ... plugin", "the plugin that ...", "can the ...") to a plugin that matches an installed one by
name or description is an update; "a/an/new ... plugin", "something ... as a plugin" is a new build.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_STOP = set("""a an the this that these those it its is are be to of for and or in on at by with from as into my your our
can could would should will please me i you we us need want make create build write add new another some something
plugin plugins tool tools do does also too well so if then than when what which who how many much get give tell
turns turn up out just like any all one two""".split())
_DEFINITE_RE = re.compile(
    r"\b(?:the|this|that|my|your|our|existing|current)\s+(?:[\w-]+\s+){0,5}?plugin\b|\bthe\s+plugin\s+(?:that|which|for|to)\b",
    re.I)
_INDEFINITE_RE = re.compile(r"\b(?:a|an|another|new|some)\s+(?:[\w-]+\s+){0,5}?plugin\b|\bas\s+a\s+plugin\b", re.I)
_UPDATE_WORDS_RE = re.compile(
    r"\b(?:also|too|as well|should|update|upgrade|change|modify|edit|extend|fix|improve|support|accept|handle|"
    r"understand|add|rework|make it|can it|let it)\b", re.I)
# "give yourself the ability to ...", "teach yourself to ...": a new capability, i.e. a plugin.
_SELF_ABILITY_RE = re.compile(
    r"\b(?:give|teach)\s+yourself\b|\byourself\s+(?:the\s+)?(?:ability|capability)\b",
    re.I)
_STANDALONE_RE = re.compile(r"\b(?:command[- ]line|cli)\s+(?:tool|utility|program|script)\b|\bstandalone\b", re.I)


# Words that say what kind of thing a plugin is, not what it is about ("miles_to_kilometers_converter").
# Live 2026-11-10: "the miles plugin" matched 1 of 3 name words of miles_to_kilometers_converter, was taken for a
# plugin that does not exist, and a second, rounding-only plugin was built.
_GENERIC = {"converter", "calculator", "checker", "counter", "tool", "helper", "util", "utility", "generator", "finder",
            "maker", "manager", "handler", "processor", "service"}


def _stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> Set[str]:
    raw = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "").lower())
    return {_stem(w) for w in raw if w not in _STOP and len(w) > 1}


def installed_plugins(plugins_dir: Path) -> Dict[str, str]:
    """{plugin name: description} for plugins/custom_tools/*.py."""
    out: Dict[str, str] = {}
    d = Path(plugins_dir)
    if not d.is_dir():
        return out
    for f in d.glob("*.py"):
        if f.stem == "__init__":
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"""^\s*description\s*(?::\s*str\s*)?=\s*(?:\(\s*)?["']([^"'\n]{0,300})""", src, re.M)
        out[f.stem] = m.group(1) if m else ""
    return out


def match_plugin(text: str, plugins: Dict[str, str]) -> Optional[str]:
    """The installed plugin the text refers to, by name words first, then description words. None when unclear."""
    words = _words(text)
    best: List[Tuple[float, str]] = []
    for name, desc in plugins.items():
        name_words = _words(name.replace("_", " ")) - _GENERIC or _words(name.replace("_", " "))
        if not name_words:
            continue
        name_hits = len(name_words & words)
        cover = name_hits / len(name_words)
        desc_hits = len(_words(desc) & words)
        if cover >= 0.5 or (name_hits >= 1 and desc_hits >= 2) or desc_hits >= 3:
            best.append((cover * 2 + desc_hits * 0.25, name))
    if not best:
        return None
    best.sort(reverse=True)
    if len(best) > 1 and best[0][0] == best[1][0]:
        return None                       # a tie: do not guess
    return best[0][1]


def _refers_to(text: str, name: str) -> bool:
    """'the palindrome thing', 'my bmi one', 'that vowel counter': a definite reference using a word of the name."""
    name_words = _words(name.replace("_", " "))
    for m in re.finditer(r"\b(?:the|this|that|my|your|our)\s+((?:[\w-]+\s+){0,3}[\w-]+)", text, re.I):
        if _words(m.group(1)) & name_words:
            return True
    return False


def resolve_plugin_request(text: str, plugins_dir: Path) -> Optional[Tuple[str, Optional[str]]]:
    t = (text or "").strip()
    if not t or _STANDALONE_RE.search(t):
        return None
    says_plugin = re.search(r"\bplugins?\b", t, re.I) is not None
    plugins = installed_plugins(plugins_dir)
    target = match_plugin(t, plugins)
    definite = _DEFINITE_RE.search(t) is not None
    if not target and definite:
        # "the miles plugin ...": one installed plugin shares a name word with the request, so that is the one meant.
        words = _words(t)
        hits = [n for n in plugins if (_words(n.replace("_", " ")) - _GENERIC) & words]
        if len(hits) == 1:
            target = hits[0]
    indefinite = _INDEFINITE_RE.search(t) is not None
    if target and definite and not indefinite:
        return ("update", target)
    if says_plugin and indefinite:
        return ("build", None)
    if target and _UPDATE_WORDS_RE.search(t) and (says_plugin or definite or _refers_to(t, target)):
        return ("update", target)
    if _SELF_ABILITY_RE.search(t) and not target:
        return ("build", None)
    if says_plugin and re.search(r"\b(?:make|create|build|write|need|want|add)\b", t, re.I):
        return ("build", None)
    question = re.search(r"^\s*(?:what|which|list|show|do|does|did|is|are|was|were|how|why|where|when)\b", t, re.I)
    # A bare description: "plugin that tells me if a year is a leap year", "plugin to resize images".
    if not target and not question and re.search(r"\bplugin\s+(?:that|which|to|for)\b", t, re.I):
        return ("build", None)
    # "the miles plugin should round to 2 decimal places" when there is no such plugin: build it as described.
    # Live 2026-11-10: this went to the general tool loop, which installed a plugin with no schema or ToolResult.
    if not target and not question and definite and _UPDATE_WORDS_RE.search(t):
        return ("build", None)
    return None
