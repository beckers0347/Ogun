"""Sampling profiles by phase: the one table every model call takes its temperature / top_p from.

Base profiles (the architecture spec):
  plan      planning and reflection                 temperature 0.2,  top_p 0.9
  code      code generation and edits, tool calls   temperature 0.05, top_p 0.85
  verdict   self-verification, arbitration, routing temperature 0.0,  top_p 0.7

Retry-diversity profiles (measured on the plugin builders, 2026-11-09/10): at a low temperature a fix that failed is
re-sampled into the same code (byte-identical replies cost whole attempts), so retries sample more widely:
  code_fix      temperature 0.3, +0.25 per repeat of the same failure (max 0.9), top_p 0.9
  code_redraft  temperature 0.7, top_p 0.95   (a fresh draft after a draft's fixes ran out)
"""
from __future__ import annotations

from typing import Dict

PROFILES: Dict[str, Dict[str, float]] = {
    "plan": {"temperature": 0.2, "top_p": 0.9},
    "code": {"temperature": 0.05, "top_p": 0.85},
    "verdict": {"temperature": 0.0, "top_p": 0.7},
    "code_fix": {"temperature": 0.3, "top_p": 0.9},
    "code_redraft": {"temperature": 0.7, "top_p": 0.95},
}


def top_p_for(temperature: float) -> float:
    """top_p that goes with a temperature taken from this table (exact match first, then the retry ranges)."""
    for p in PROFILES.values():
        if abs(p["temperature"] - temperature) < 1e-9:
            return p["top_p"]
    return 0.95 if temperature >= 0.7 else 0.9


def profile(name: str, streak: int = 0) -> Dict[str, float]:
    """temperature/top_p for a phase. `streak` (code_fix only) = how many times in a row the same failure came back."""
    if name not in PROFILES:
        raise KeyError(f"unknown sampling profile {name!r}; known: {', '.join(PROFILES)}")
    p = dict(PROFILES[name])
    if name == "code_fix" and streak > 1:
        p["temperature"] = min(p["temperature"] + 0.25 * (streak - 1), 0.9)
    return p
