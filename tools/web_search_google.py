"""
tools/web_search_google.py
--------------------------
Google Custom Search JSON API helper.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple


def web_search_google_status(query: str, num_results: int = 3) -> Tuple[List[Dict[str, str]], str, str]:
    if not query:
        return [], "empty_query", "empty_query"

    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_CX")
    if not api_key or not cx:
        return [], "missing_env", "missing_env"

    params = urllib.parse.urlencode(
        {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": int(num_results),
        }
    )
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Orchestrator/0.2)"}

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return [], "error", str(exc)

    try:
        data = json.loads(body)
    except Exception:
        return [], "error", "invalid_json"

    results: List[Dict[str, str]] = []
    for item in data.get("items", [])[:num_results]:
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
                "url": str(item.get("link", "")).strip(),
            }
        )
    if not results:
        return [], "empty", "empty"
    return results, "ok", ""


def web_search_google(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    results, _, _ = web_search_google_status(query, num_results=num_results)
    return results
