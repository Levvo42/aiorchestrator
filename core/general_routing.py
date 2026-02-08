"""
general_routing.py
------------------
Helpers for local-first general routing with strict escalation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


FACTUAL_KEYWORDS = (
    "when",
    "where",
    "who",
    "how many",
    "latest",
    "current",
    "today",
    "now",
    "as of",
    "price",
    "version",
    "release",
    "news",
    "population",
    "score",
    "time",
    "date",
)

CODING_KEYWORDS = ("bug", "error", "stack trace", "python", "code", "refactor", "function", "class")
WRITING_KEYWORDS = ("rewrite", "tone", "email", "polish", "copy", "caption", "summarize", "summary")
PLANNING_KEYWORDS = ("plan", "roadmap", "milestone", "strategy", "steps", "outline")


def classify_intent(task: str) -> str:
    t = (task or "").lower()
    if any(k in t for k in CODING_KEYWORDS):
        return "coding"
    if any(k in t for k in WRITING_KEYWORDS):
        return "writing"
    if any(k in t for k in PLANNING_KEYWORDS):
        return "planning"
    if is_factual_query(task):
        return "factual"
    return "general"


def is_factual_query(task: str) -> bool:
    t = (task or "").lower()
    if any(k in t for k in FACTUAL_KEYWORDS):
        return True
    if "?" in t and any(w in t for w in ("what", "which", "how many", "how much")):
        return True
    return False


def build_local_assessment_prompt(task: str, threshold: float) -> str:
    return (
        "System: You are a local model. You must respond ONLY with valid JSON.\n"
        "System: Provide a direct answer plus a strict self-assessment.\n"
        "System: If uncertain, set confidence < {thresh:.2f} and explain why.\n"
        "System: If the question is factual/time-sensitive and confidence < {thresh:.2f},\n"
        "System: set needs_web=true and provide suggested_search_query.\n"
        "System: Do not include any text outside JSON.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "confidence": 0.0,\n'
        '  "needs_web": false,\n'
        '  "needs_api": false,\n'
        '  "uncertainty_reasons": ["..."],\n'
        '  "suggested_search_query": "string or empty"\n'
        "}\n\n"
        f"Task:\n{task}\n"
    ).format(thresh=threshold)


def parse_local_assessment(raw_output: str) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        data = json.loads(raw_output)
    except Exception:
        return None, "invalid_json"

    if not isinstance(data, dict):
        return None, "invalid_json"

    answer = data.get("answer")
    confidence = data.get("confidence")
    needs_web = data.get("needs_web")
    needs_api = data.get("needs_api")
    uncertainty_reasons = data.get("uncertainty_reasons")
    suggested_query = data.get("suggested_search_query")

    if not isinstance(answer, str):
        return None, "invalid_json"
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        return None, "invalid_json"
    if not isinstance(needs_web, bool) or not isinstance(needs_api, bool):
        return None, "invalid_json"
    if not isinstance(uncertainty_reasons, list) or not all(isinstance(r, str) for r in uncertainty_reasons):
        return None, "invalid_json"
    if suggested_query is None:
        suggested_query = ""
    if not isinstance(suggested_query, str):
        return None, "invalid_json"

    return {
        "answer": answer.strip(),
        "confidence": float(confidence),
        "needs_web": bool(needs_web),
        "needs_api": bool(needs_api),
        "uncertainty_reasons": [r.strip() for r in uncertainty_reasons if r.strip()],
        "suggested_search_query": suggested_query.strip(),
    }, ""


def decide_general_route(
    task: str,
    assessment: Optional[Dict[str, Any]],
    confidence_threshold: float,
    web_first_threshold: float,
    learned_confidence: float,
) -> Dict[str, Any]:
    factual = is_factual_query(task)
    result = {
        "route": "web",
        "reason": "invalid_json",
        "factual": factual,
        "confidence": 0.0,
    }

    if assessment is None:
        return result

    confidence_used = min(float(assessment.get("confidence", 0.0) or 0.0), learned_confidence)
    result["confidence"] = confidence_used

    uncertainty_reasons = assessment.get("uncertainty_reasons") or []
    needs_web = bool(assessment.get("needs_web"))
    needs_api = bool(assessment.get("needs_api"))

    if uncertainty_reasons:
        result["reason"] = "uncertainty_reasons"
        result["route"] = "web" if factual else "api"
        return result

    if needs_web and needs_api:
        result["reason"] = "needs_web_and_api"
        result["route"] = "web_api"
        return result

    if needs_web:
        result["reason"] = "needs_web"
        result["route"] = "web"
        return result

    if needs_api:
        result["reason"] = "needs_api"
        result["route"] = "api"
        return result

    if confidence_used < confidence_threshold:
        if factual:
            result["reason"] = "confidence_below_threshold_factual"
            result["route"] = "web"
        else:
            result["reason"] = "confidence_below_threshold_non_factual"
            result["route"] = "api"
        return result

    if factual and confidence_used < web_first_threshold:
        result["reason"] = "web_first_threshold"
        result["route"] = "web"
        return result

    result["reason"] = "local_confident"
    result["route"] = "local"
    return result


def summarize_evidence(results: List[Dict[str, str]]) -> str:
    if not results:
        return "No web results available."
    lines: List[str] = ["Top sources:"]
    for item in results[:3]:
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        url = item.get("url", "").strip()
        lines.append(f"- {title} | {snippet} | {url}")
    return "\n".join(lines)
