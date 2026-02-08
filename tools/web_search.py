"""
tools/web_search.py
-------------------
Minimal web search helper (no heavy deps).

Uses DuckDuckGo HTML results and extracts top 3 snippets.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import Dict, List


_RESULT_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)
_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE)
_RESULT_BLOCK_RE = re.compile(r'<div class="result__body".*?</div>\s*</div>', re.IGNORECASE | re.DOTALL)


def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    if not query:
        return []

    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Orchestrator/0.2)"}

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for block in _RESULT_BLOCK_RE.findall(body):
        title_match = _RESULT_RE.search(block)
        snippet_match = _SNIPPET_RE.search(block)
        if not title_match:
            continue
        url_raw = html.unescape(title_match.group(1))
        title = html.unescape(re.sub(r"<.*?>", "", title_match.group(2)))
        snippet = ""
        if snippet_match:
            snippet = html.unescape(re.sub(r"<.*?>", "", snippet_match.group(1)))
        results.append({"title": title.strip(), "snippet": snippet.strip(), "url": url_raw.strip()})
        if len(results) >= max_results:
            break

    return results
