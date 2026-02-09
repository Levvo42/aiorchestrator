"""
tools/web_search_vertex.py
--------------------------
Vertex AI Search (Discovery Engine) retrieval helper.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

from core.env import require_env_var


def web_search_vertex_status(query: str, num_results: int = 3) -> Tuple[List[Dict[str, str]], str, str]:
    if not query:
        return [], "empty_query", "empty_query"

    try:
        project_id = require_env_var("VERTEX_PROJECT_ID")
        location = require_env_var("VERTEX_LOCATION")
        collection = require_env_var("VERTEX_COLLECTION")
        engine_id = require_env_var("VERTEX_ENGINE_ID")
        serving_config = require_env_var("VERTEX_SERVING_CONFIG")
        require_env_var("VERTEX_DATA_STORE_ID")
    except RuntimeError as exc:
        print(f"WebSearch vertex_search: missing_env detail=\"{exc}\"")
        return [], "missing_env", str(exc)

    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception as exc:
        print(f"WebSearch vertex_search: provider_unavailable detail=\"{exc}\"")
        return [], "provider_unavailable", str(exc)

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = credentials.token
    except Exception as exc:
        print(f"WebSearch vertex_search: auth_failed detail=\"{exc}\"")
        return [], "auth_failed", str(exc)

    path = (
        f"projects/{project_id}/locations/{location}/collections/{collection}"
        f"/engines/{engine_id}/servingConfigs/{serving_config}:search"
    )
    url = f"https://discoveryengine.googleapis.com/v1/{path}"

    payload = json.dumps({"query": query, "pageSize": int(num_results)}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; AI-Orchestrator/0.2)",
    }

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        status = "permission_denied" if exc.code == 403 else "error"
        try:
            err_body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        print(f"WebSearch vertex_search: http_error status={exc.code} body={err_body}")
        return [], status, f"http_{exc.code}"
    except Exception as exc:
        print(f"WebSearch vertex_search: request_failed detail=\"{exc}\"")
        return [], "error", str(exc)

    try:
        data = json.loads(body)
    except Exception:
        return [], "error", "invalid_json"

    results: List[Dict[str, str]] = []
    for item in data.get("results", [])[:num_results]:
        doc = item.get("document", {})
        derived = doc.get("derivedStructData", {}) or {}
        url_val = str(derived.get("link") or derived.get("url") or doc.get("uri") or "").strip()
        title = str(derived.get("title") or doc.get("title") or "").strip()
        snippet = str(derived.get("snippet") or derived.get("description") or "").strip()
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
