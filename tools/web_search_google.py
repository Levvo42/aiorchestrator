"""
tools/web_search_google.py
--------------------------
Google Custom Search JSON API helper.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

from core.env import require_env_var


def _redact_key(url: str) -> str:
    if "key=" not in url:
        return url
    parts = url.split("key=", 1)
    rest = parts[1]
    if "&" in rest:
        _, tail = rest.split("&", 1)
        return f"{parts[0]}key=REDACTED&{tail}"
    return f"{parts[0]}key=REDACTED"


def web_search_google_status(query: str, num_results: int = 3) -> Tuple[List[Dict[str, str]], str, str]:
    if not query:
        return [], "empty_query", "empty_query"

    if load_dotenv:
        load_dotenv()

    try:
        api_key = require_env_var("GOOGLE_CSE_API_KEY")
        cx = require_env_var("GOOGLE_CSE_CX")
    except RuntimeError as exc:
        print(f"WebSearch: google_cse FAILED reason=\"{exc}\"")
        return [], "missing_env", str(exc)

    params = urllib.parse.urlencode(
        {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": int(num_results),
        }
    )
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    print(
        "WebSearch: google_cse DEBUG "
        f"url=\"{_redact_key(url)}\" cx={cx} env_present=True"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Orchestrator/0.2)"}

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        print(f"WebSearch: google_cse HTTPError status={exc.code} body={err_body}")
        return [], "error", f"http_{exc.code}"
    except Exception as exc:
        print(f"WebSearch: google_cse FAILED reason=\"{exc}\"")
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
