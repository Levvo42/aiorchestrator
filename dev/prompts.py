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
        "Role: senior software engineer.\n"
        "Task: produce one unified diff that implements the request.\n"
        "\n"
        "Output rules:\n"
        "- Output only a unified diff suitable for `git apply`.\n"
        "- No markdown or explanations.\n"
        "- Omit all 'index ...' lines.\n"
        "- File headers must be:\n"
        "  diff --git a/path b/path\n"
        "  --- a/path\n"
        "  +++ b/path\n"
        "- Use exact paths from the repo.\n"
        "- Patch must apply cleanly to the provided contents.\n"
        "- When editing, copy exact original lines to keep context correct.\n"
        "- Do not paraphrase existing strings.\n"
        "- Keep changes minimal and request-scoped.\n"
        "- Avoid renames or refactors unless required.\n"
        "- Ensure Python syntax is valid.\n"
        "\n"
        "REQUEST:\n"
        f"{request.strip()}\n"
        "\n"
        "CONTEXT (JSON BUNDLE):\n"
        f"{context_text}\n"
    )

def build_fix_author_prompt(
    request: str,
    context: Any,
    failed_patch: str,
    apply_error: str,
) -> str:
    """Build a repair prompt after a patch failed to apply."""

    context_text = _context_to_text(context)

    failed_patch = (failed_patch or "").strip()
    apply_error = (apply_error or "").strip()

    return (
        "Role: senior software engineer.\n"
        "A prior diff failed to apply with `git apply`.\n"
        "\n"
        "Task:\n"
        "- Produce a new unified diff that implements the original request.\n"
        "- The new diff must apply cleanly to the current context.\n"
        "\n"
        "Output rules:\n"
        "- Output only a unified diff suitable for `git apply`.\n"
        "- No markdown or prose.\n"
        "- Omit all 'index ...' lines.\n"
        "- File headers must be:\n"
        "  diff --git a/path b/path\n"
        "  --- a/path\n"
        "  +++ b/path\n"
        "- Use exact paths and exact original lines.\n"
        "- Prefer minimal edits over refactors.\n"
        "- Rebuild hunks if prior context was incorrect.\n"
        "\n"
        "ORIGINAL REQUEST:\n"
        f"{request.strip()}\n"
        "\n"
        "GIT APPLY ERROR:\n"
        f"{apply_error}\n"
        "\n"
        "FAILED PATCH (REFERENCE ONLY):\n"
        f"{failed_patch}\n"
        "\n"
        "CURRENT CONTEXT:\n"
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
        "Role: patch judge.\n"
        "Select the best candidate by index.\n"
        "Do not write code, modify patches, or output a diff.\n"
        "Pick exactly one index.\n"
    )

    lines.append("REQUEST")
    lines.append(request.strip())
    lines.append("")

    if context_text:
        lines.append("CONTEXT (JSON BUNDLE)")
        lines.append(context_text)
        lines.append("")

    lines.append("CANDIDATE PATCHES")
    for i, patch_text in enumerate(patches):
        lines.append(f"\n[PATCH {i}]")
        lines.append(patch_text.strip())

    lines.append(
        "\nEVALUATION CRITERIA\n"
        "- Correctness: does it implement the request?\n"
        "- Minimality: smallest necessary change, avoids unrelated edits.\n"
        "- Safety: avoids breaking behavior; avoids risky refactors.\n"
        "- Patch quality: looks like a real unified diff; applies cleanly.\n"
    )

    lines.append(
        "OUTPUT RULES\n"
        "Return only valid JSON.\n"
        "No markdown or extra text.\n"
        "\n"
        "Format:\n"
        "{\n"
        '  "patch_index": <integer>,\n'
        '  "rationale": "<short explanation>"\n'
        "}\n"
    )

    return "\n".join(lines)


def build_local_judge_prompt(request: str, context: Any, patches: list[str]) -> str:
    """
    Prompt for local Ollama judge.

    Output (STRICT JSON):
      {
        "patch_index": <int or null>,
        "confidence": <number 0..1>,
        "uncertainty_reasons": ["..."],
        "rationale": "short"
      }
    """
    context_text = _context_to_text(context)

    lines: list[str] = []

    lines.append(
        "Role: local patch judge (Ollama).\n"
        "Decide if you are confident enough to pick a patch.\n"
        "If uncertain, set patch_index to null or confidence below 0.90,\n"
        "and list uncertainty_reasons.\n"
    )

    lines.append("REQUEST")
    lines.append(request.strip())
    lines.append("")

    if context_text:
        lines.append("CONTEXT (JSON BUNDLE)")
        lines.append(context_text)
        lines.append("")

    lines.append("CANDIDATE PATCHES")
    for i, patch_text in enumerate(patches):
        lines.append(f"\n[PATCH {i}]")
        lines.append(patch_text.strip())

    lines.append(
        "\nEVALUATION CRITERIA\n"
        "- Correctness: does it implement the request?\n"
        "- Minimality: smallest necessary change, avoids unrelated edits.\n"
        "- Safety: avoids breaking behavior; avoids risky refactors.\n"
        "- Patch quality: looks like a real unified diff; applies cleanly.\n"
    )

    lines.append(
        "OUTPUT RULES\n"
        "Return only valid JSON.\n"
        "No markdown or extra text.\n"
        "If uncertain, set patch_index to null or confidence below 0.90,\n"
        "and include uncertainty_reasons.\n"
        "\n"
        "Format:\n"
        "{\n"
        '  "patch_index": <integer or null>,\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "uncertainty_reasons": ["..."],\n'
        '  "rationale": "<short explanation>"\n'
        "}\n"
    )

    return "\n".join(lines)
