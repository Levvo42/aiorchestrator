"""
dev/dev_command.py
------------------
Implements the "Dev: <request>" command for development tasks.

Flow:
1) Build local context bundle (tree + relevant files)
2) DevPolicy decides author providers + judge provider
3) Ask each author provider to produce a unified diff patch
4) Ask judge provider to pick the best patch (returns JSON: patch_index + rationale)
5) Show patch + rationale
6) Ask user to apply (yes/no)
7) Apply patch
8) Validate
9) Return a structured report to be stored in memory

IMPORTANT:
- This module must NEVER treat plain text as a patch.
- Only unified diffs are allowed to be applied.
- Judge should select among already-generated candidate patches using patch_index.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from dev.context import build_context_bundle
from dev.policy import DevPolicy
from dev.prompts import build_author_prompt, build_judge_prompt
from dev.patch_apply import apply_patches
from dev.validate import py_compile_files


def _safe_json_load(s: str) -> Optional[dict]:
    """
    Attempt to parse JSON from a model output.
    Returns dict if successful; otherwise None.
    """
    try:
        return json.loads(s)
    except Exception:
        return None

def _strip_markdown_fences(text: str) -> str:
    """
    Some models wrap diffs in ```diff ... ``` fences.
    This removes the fences so apply_patches receives a raw unified diff.
    """
    t = (text or "").strip()

    if t.startswith("```"):
        lines = t.splitlines()

        # Remove first fence line (``` or ```diff)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        # Remove last fence line if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return t

def _looks_like_unified_diff(text: str) -> bool:
    """
    Small heuristic to detect a unified diff.
    This prevents accidental "plain text" from being treated as a patch.

    Accepts common diff formats:
    - "diff --git ..." (git diff format)
    - "--- a/file" + "+++ b/file" (unified diff format)
    """
    t = (text or "").strip()
    if not t:
        return False

    if "diff --git" in t:
        return True

    # Classic unified diff header
    if t.startswith("--- ") and "\n+++ " in t:
        return True

    return False


def _extract_patch_text(patch_item: Any) -> str:
    """
    Given a candidate patch item, return the patch text.

    Our author_outputs store patches as dicts:
      {"provider": "...", "success": True, "patch": "<diff text>"}

    But we also accept raw strings defensively.
    """
    if isinstance(patch_item, dict):
        return str(patch_item.get("patch", "")).strip()
    return str(patch_item).strip()


def _choose_first_valid_patch(successful_patches: List[Any]) -> Tuple[str, str]:
    """
    Choose the first candidate patch that looks like a unified diff.

    Returns:
        (patch_text, rationale)
    """
    for item in successful_patches:
        patch_text = _strip_markdown_fences(_extract_patch_text(item))
        if _looks_like_unified_diff(patch_text):
            return patch_text, "Fallback: selected first candidate patch that looks like a unified diff."

    # No candidate looked valid
    return "", "Fallback: no candidate patch looked like a unified diff; no patch selected."


def run_dev_request(
    repo_root: str,
    request: str,
    capabilities: dict,
    memory: Any,
    provider_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a dev request and return a report dict.

    memory: your MemoryStore object (used for settings/stats/logging)
    provider_map: {"openai": OpenAIClient(), "claude": ClaudeClient(), ...}
                 each must provide .generate(prompt) -> str
    """
    # Build context for dev models
    context = build_context_bundle(repo_root=repo_root, request=request)

    # Decide policy (authors + judge) locally
    policy = DevPolicy(capabilities)
    provider_stats = memory.get_provider_stats()

    # Dev settings are stored in memory settings
    dev_settings = {
        "dev_mode": memory.get_setting("dev_mode", "auto"),
        "dev_authors": memory.get_setting("dev_authors", None),
        "dev_judge_provider": memory.get_setting("dev_judge_provider", None),
        "dev_min_authors": memory.get_setting("dev_min_authors", None),
        "dev_max_authors": memory.get_setting("dev_max_authors", None),
        "dev_exploration_rate": memory.get_setting("dev_exploration_rate", None),
    }

    decision = policy.decide(provider_stats=provider_stats, settings=dev_settings)

    # ----------------------------
    # 1) Generate candidate patches
    # ----------------------------
    author_outputs: List[Dict[str, Any]] = []
    author_prompt = build_author_prompt(request=request, context=context)

    for provider_name in decision.author_providers:
        client = provider_map.get(provider_name)
        if not client:
            author_outputs.append(
                {
                    "provider": provider_name,
                    "success": False,
                    "error": f"Provider '{provider_name}' not found. Available: {list(provider_map.keys())}",
                }
            )
            continue

        try:
            patch_text = client.generate(author_prompt)
            # Remove ```diff fences early so everything downstream is clean.
            patch_text = _strip_markdown_fences(patch_text)
            author_outputs.append({"provider": provider_name, "success": True, "patch": patch_text})
            memory.update_provider_stats(provider_name, success=True)
        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})
            memory.update_provider_stats(provider_name, success=False)

    successful_patches = [o for o in author_outputs if o.get("success") and o.get("patch")]

    # ----------------------------
    # 2) Judge chooses best patch
    # ----------------------------
    judge_rationale = ""
    chosen_patch = ""

    judge_client = provider_map.get(decision.judge_provider)

    # IMPORTANT:
    # build_judge_prompt should instruct the judge to output JSON:
    #   {"patch_index": <int>, "rationale": "..."}
    # We pass only the list of patch TEXTS to the judge prompt to avoid confusion.
    candidate_patch_texts = [_extract_patch_text(p) for p in successful_patches]

    judge_prompt = build_judge_prompt(
        request=request,
        context=context,
        patches=candidate_patch_texts,  # list[str], not list[dict]
    )

    # If we can't judge (no judge client OR no candidate patches),
    # fall back to "first valid unified diff".
    if not judge_client or not successful_patches:
        chosen_patch, judge_rationale = _choose_first_valid_patch(successful_patches)
        if not judge_client:
            judge_rationale = (
                f"Judge unavailable: '{decision.judge_provider}' not in provider_map. "
                f"{judge_rationale}"
            )
        else:
            judge_rationale = f"No successful patches to judge. {judge_rationale}"

    else:
        # Judge exists and we have candidates; attempt to judge.
        try:
            judge_output = judge_client.generate(judge_prompt)

            # Preferred: strict JSON with patch_index
            judge_json = _safe_json_load(judge_output)

            if judge_json and "patch_index" in judge_json:
                idx = judge_json.get("patch_index")
                judge_rationale = str(judge_json.get("rationale", "")).strip()

                # Validate idx
                if isinstance(idx, int) and 0 <= idx < len(candidate_patch_texts):
                    candidate = candidate_patch_texts[idx].strip()

                    # Some models wrap diffs in ```diff fences; strip them before checking/applying.
                    candidate = _strip_markdown_fences(candidate)

                    # Only accept if it is actually a diff
                    if _looks_like_unified_diff(candidate):
                        chosen_patch = candidate
                    else:
                        # Selected patch doesn't look like a diff; safe fallback
                        chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                        judge_rationale = (
                            f"Judge selected patch_index={idx}, but selected patch did not look like a unified diff. "
                            f"{fallback_reason}"
                        )
                else:
                    # Invalid index; safe fallback
                    chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                    judge_rationale = (
                        f"Judge returned invalid patch_index={idx}. {fallback_reason}"
                    )

            else:
                # Judge did not follow strict JSON. DO NOT treat raw output as a patch.
                chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                judge_rationale = (
                    "Judge did not return valid JSON with patch_index; ignored raw judge output. "
                    f"{fallback_reason}\n"
                    "Raw judge output was:\n"
                    f"{judge_output.strip()}"
                )

            memory.update_provider_stats(decision.judge_provider, success=True)

        except Exception as e:
            # Judge call failed; safe fallback
            chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
            judge_rationale = f"Judge failed: {e}. {fallback_reason}"
            memory.update_provider_stats(decision.judge_provider, success=False)

    # Build the report (this is what main.py prints and later stores in memory)
    report: Dict[str, Any] = {
        "request": request,
        "context": context,
        "policy": {
            "mode": decision.mode,
            "authors": decision.author_providers,
            "judge": decision.judge_provider,
            "reason": decision.reason,
        },
        "authors": author_outputs,  # includes failures and successes
        "judge": {
            "provider": decision.judge_provider,
            "rationale": judge_rationale,
        },
        "chosen_patch": chosen_patch,
        "apply": {
            "attempted": False,
            "applied": False,
            "changed_files": [],
            "validation_ok": False,
            "validation_output": "",
            "error": "",
        },
    }

    return report


def apply_dev_patch(repo_root: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the chosen patch in report to filesystem and validate.

    This is separated so main.py can ask user "yes/no" before calling apply.
    """
    patch = (report.get("chosen_patch") or "").strip()
    # Safety net: if a fenced patch slipped through, clean it now.
    patch = _strip_markdown_fences(patch)

    # Ensure report has an apply section
    report.setdefault("apply", {})
    report["apply"].setdefault("attempted", False)
    report["apply"].setdefault("applied", False)
    report["apply"].setdefault("changed_files", [])
    report["apply"].setdefault("validation_ok", False)
    report["apply"].setdefault("validation_output", "")
    report["apply"].setdefault("error", "")

    if not patch:
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = "No patch available to apply."
        return report

    # NEVER apply non-diff text
    if not _looks_like_unified_diff(patch):
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = "Chosen patch did not look like a unified diff. Refusing to apply."
        return report

    report["apply"]["attempted"] = True

    try:
        backups = apply_patches(repo_root=repo_root, diff_text=patch)
        changed_files = list(backups.keys())

        report["apply"]["changed_files"] = changed_files
        report["apply"]["applied"] = True

        # Validate changed Python files (py_compile)
        ok, out = py_compile_files(repo_root=repo_root, changed_paths=changed_files)
        report["apply"]["validation_ok"] = ok
        report["apply"]["validation_output"] = out

        # If validation failed, keep applied=True (files were written),
        # but error is blank unless apply_patches itself failed.
        # You can later add rollback using "backups" if you want.
        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
