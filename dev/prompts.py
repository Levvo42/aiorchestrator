"""
dev/prompts.py
--------------
Prompt builders for the self-patching workflow.

This module must handle context being either:
- dict (from build_context_bundle)
- or str (if you later change context builder)

Design goals:
- Authors produce unified diffs ONLY.
- Judge selects the best candidate by INDEX (patch_index) and returns STRICT JSON.
"""

from __future__ import annotations

import json
from typing import Any


def _context_to_text(context: Any) -> str:
    """
    Convert context into a stable text format for LLM prompts.

    If context is a dict (recommended), we serialize it as pretty JSON
    so the model can read it consistently.

    If context is already a string, we return it as-is.
    """
    if context is None:
        return ""

    if isinstance(context, str):
        return context.strip()

    # If it's a dict or list (or any JSON-serializable structure), stringify it.
    try:
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception:
        # Fallback: last resort string conversion
        return str(context)


def build_author_prompt(request: str, context: Any) -> str:
    """
    Prompt for dev "author" models.

    The author MUST return a unified diff patch ONLY.
    No markdown. No explanation. No extra text.
    """
    context_text = _context_to_text(context)

    return (
        "You are a senior software engineer.\n"
        "You will receive a codebase context bundle and a change request.\n"
        "\n"
        "TASK:\n"
        "- Produce ONE unified diff patch that implements the request.\n"
        "- The output MUST be a unified diff suitable for `git apply`.\n"
        "- DO NOT include markdown fences.\n"
        "- DO NOT include explanations.\n"
        "- DO NOT output anything except the diff.\n"
        "\n"
        "IMPORTANT PATCH RULES:\n"
        "- Keep changes minimal and directly related to the request.\n"
        "- Do not rename files or do broad refactors unless the request requires it.\n"
        "- Ensure Python syntax is valid.\n"
        "\n"
        "=== REQUEST ===\n"
        f"{request.strip()}\n"
        "\n"
        "=== CONTEXT (JSON BUNDLE) ===\n"
        f"{context_text}\n"
    )


def build_judge_prompt(request: str, context: Any, patches: list[str]) -> str:
    """
    Prompt for dev "judge" model.

    Input:
      patches: list[str] where each element is the patch text (unified diff)

    Output (STRICT):
      {"patch_index": <int>, "rationale": "<short text>"}

    The judge MUST NOT output a patch/diff.
    It must only select among the given candidates by index.
    """
    context_text = _context_to_text(context)

    lines: list[str] = []

    lines.append(
        "You are a senior software engineer acting as a PATCH JUDGE.\n"
        "Your ONLY job is to choose the best patch among the candidates.\n"
        "You MUST NOT write new code.\n"
        "You MUST NOT modify patches.\n"
        "You MUST NOT output a diff.\n"
        "You MUST pick exactly ONE candidate by its index.\n"
    )

    lines.append("=== USER REQUEST ===")
    lines.append(request.strip())
    lines.append("")

    if context_text:
        lines.append("=== CONTEXT (JSON BUNDLE) ===")
        lines.append(context_text)
        lines.append("")

    lines.append("=== CANDIDATE PATCHES ===")
    for i, patch_text in enumerate(patches):
        lines.append(f"\n[PATCH {i}]")
        lines.append(patch_text.strip())

    lines.append(
        "\n=== EVALUATION CRITERIA ===\n"
        "- Correctness: does it implement the request?\n"
        "- Minimality: smallest necessary change, avoids unrelated edits.\n"
        "- Safety: avoids breaking behavior; avoids risky refactors.\n"
        "- Patch quality: looks like a real unified diff; applies cleanly.\n"
    )

    lines.append(
        "=== OUTPUT RULES (STRICT) ===\n"
        "Return ONLY valid JSON.\n"
        "No markdown.\n"
        "No extra text before or after JSON.\n"
        "\n"
        "Output format:\n"
        "{\n"
        '  "patch_index": <integer>,\n'
        '  "rationale": "<short explanation>"\n'
        "}\n"
    )

    return "\n".join(lines)
