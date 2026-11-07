"""Web search for build research, standard library only.

When a tool build keeps failing the same way, the builder searches the web for the error before asking the model for
a root-cause diagnosis. This uses DuckDuckGo's HTML endpoint through urllib, so it adds no dependency.
It is OFF by default because Ogun promises to stay local: set OGUN_WEB_RESEARCH=1 to allow it. When it is off the
diagnosis still runs, only without web findings.
"""
from __future__ import annotations

import html
import os
import re
import urllib.parse
import urllib.request

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def web_research_enabled() -> bool:
    return os.environ.get("OGUN_WEB_RESEARCH", "0").strip().lower() in ("1", "true", "yes", "on")


def _clean(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment or "")).strip()


def _real_url(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<url>."""
    if "uddg=" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(href)).query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    return html.unescape(href)


def parse_results(page: str, max_results: int = 3) -> str:
    out = []
    for href, title, snippet in _RESULT_RE.findall(page or ""):
        out.append(f"- {_clean(title)} ({_real_url(href)}): {_clean(snippet)[:300]}")
        if len(out) >= max_results:
            break
    return "\n".join(out)


def search_web(query: str, max_results: int = 3, timeout: float = 10.0) -> str:
    """Return up to `max_results` results as text lines. Errors come back as 'Error: ...' (never raised)."""
    if not web_research_enabled():
        return "Error: web research is off (set OGUN_WEB_RESEARCH=1 to allow it)."
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Ogun build research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Error: web search failed: {exc}"
    return parse_results(page, max_results) or "Error: no results."
