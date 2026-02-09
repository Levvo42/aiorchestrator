"""
tools/web_search_open_web.py
----------------------------
Open web search via Brave Search API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

from core.env import require_env_var


DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def web_search_open_web_status(query: str, num_results: int = 3) -> Tuple[List[Dict[str, str]], str, str]:
    if not query:
        return [], "empty_results", "empty_results"

    try:
        api_key = require_env_var("BRAVE_SEARCH_API_KEY")
    except RuntimeError as exc:
        print(f"WebSearch open_web: missing_env detail=\"{exc}\"")
        return [], "missing_env", str(exc)

    endpoint = os.getenv("BRAVE_SEARCH_ENDPOINT", DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT
    params = {"q": query, "count": str(int(num_results))}
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; AI-Orchestrator/0.2)",
    }

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            status = "auth_failed"
        elif exc.code == 429:
            status = "rate_limited"
        else:
            status = "http_error"
        try:
            err_body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        print(f"WebSearch open_web: http_error status={exc.code} body={err_body}")
        return [], status, f"http_{exc.code}"
    except Exception as exc:
        print(f"WebSearch open_web: request_failed detail=\"{exc}\"")
        return [], "http_error", str(exc)

    try:
        data = json.loads(body)
    except Exception:
        return [], "http_error", "invalid_json"

    results_raw = (data.get("web") or {}).get("results") or data.get("results") or []
    results: List[Dict[str, str]] = []
    for item in results_raw[:num_results]:
        url_val = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("description") or item.get("snippet") or "").strip()
        source_domain = ""
        if url_val:
            source_domain = urllib.parse.urlparse(url_val).netloc
        if url_val or title or snippet:
            results.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url_val,
                    "source_domain": source_domain,
                }
            )

    if not results:
        return [], "empty_results", "empty_results"
    return results, "ok", ""
