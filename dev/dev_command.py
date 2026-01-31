"""
dev/dev_command.py
------------------
Developer loop for self-patching the repo.

Correctness-first invariant:
- Self-patching must be reproducible from repo state.

Two-step flow (matches main.py):
1) run_dev_request(...) proposes a patch (does NOT apply).
2) apply_dev_patch(repo_root=".", report=...) applies only after user confirmation.

Report schema returned by run_dev_request is designed to be consumed directly by main.py:
- report["policy"]["mode"]
- report["policy"]["author_providers"]
- report["policy"]["judge_provider"]
- report["policy"]["reason"]
- report["judge"]["rationale"]
- report["chosen_patch"]
- report["base_commit"]

apply_dev_patch mutates and returns the same report with:
- report["apply"] filled with results
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dev.context import build_context_bundle
from dev.patch_apply import apply_patches
from dev.policy import DevPolicy
from dev.prompts import build_author_prompt, build_judge_prompt
from dev.validate import py_compile_files


# -----------------------------
# Git invariants (core safety)
# -----------------------------

def _git_head(repo_root: str) -> str:
    """Return current HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git rev-parse failed")
        return result.stdout.strip()
    except FileNotFoundError as e:
        raise RuntimeError("git not found on PATH; cannot enforce reproducible self-patching") from e


def _git_is_clean(repo_root: str) -> Tuple[bool, str]:
    """Returns (is_clean, porcelain_output_if_dirty)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr.strip() or result.stdout.strip() or "git status failed")
    out = result.stdout.strip()
    return (out == ""), out


def _require_clean_tree(repo_root: str, phase: str) -> None:
    clean, out = _git_is_clean(repo_root)
    if not clean:
        raise RuntimeError(
            f"Refusing to proceed: repo is dirty during '{phase}'.\n"
            f"git status --porcelain:\n{out}"
        )


def _require_head(repo_root: str, expected_head: str, phase: str) -> None:
    head = _git_head(repo_root)
    if head != expected_head:
        raise RuntimeError(
            f"Refusing to proceed: HEAD changed during '{phase}'.\n"
            f"expected: {expected_head}\n"
            f"actual:   {head}"
        )


# -----------------------------
# Patch text helpers
# -----------------------------

_DIFF_START_RE = re.compile(r"^(diff --git|---\s|\+\+\+\s|@@\s)", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    """Remove surrounding ``` fences (including ```diff) if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t, count=1)
        t = re.sub(r"\n```$", "", t, count=1)
    return t.strip()


def _looks_like_unified_diff(text: str) -> bool:
    return bool(text) and isinstance(text, str) and bool(_DIFF_START_RE.search(text))


def _choose_first_valid_patch(candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Fallback: return first candidate patch that looks like a unified diff."""
    successful = [c for c in candidates if c.get("success") and isinstance(c.get("patch"), str)]
    for item in successful:
        patch_text = _strip_markdown_fences(item["patch"])
        if _looks_like_unified_diff(patch_text):
            return patch_text, "Fallback: selected first candidate patch that looks like a unified diff."
    return "", "Fallback: no candidate patch looked like a unified diff; no patch selected."


# -----------------------------
# Step 1: propose patch (no apply)
# -----------------------------

def run_dev_request(
    repo_root: str,
    request: str,
    capabilities: dict,
    memory: Any,
    provider_map: Dict[str, Any],
) -> Dict[str, Any]:
    repo_root = str(Path(repo_root).resolve())

    report: Dict[str, Any] = {
        "request": request,
        "base_commit": "",
        "policy": {
            "mode": None,
            "author_providers": [],
            "judge_provider": None,
            "reason": "",
        },
        "authors": [],  # list of {provider, success, patch|error}
        "judge": {
            "provider": None,
            "success": False,
            "rationale": "",
        },
        "chosen_patch": "",
        "apply": {
            "attempted": False,
            "applied": False,
            "changed_files": [],
            "validation_ok": False,
            "validation_output": "",
            "error": "",
        },
    }

    # Invariant: must start clean, and pin the base commit used for context + patch generation
    _require_clean_tree(repo_root, phase="start")
    base_commit = _git_head(repo_root)
    report["base_commit"] = base_commit

    _require_head(repo_root, base_commit, phase="pre-context")
    _require_clean_tree(repo_root, phase="pre-context")

    # Context (already filtered by dev/context.py)
    context = build_context_bundle(repo_root=repo_root, request=request)

    # Decide dev policy
    policy = DevPolicy(capabilities)
    provider_stats = memory.get_provider_stats()

    dev_settings = {
        "dev_mode": memory.get_setting("dev_mode", "auto"),
        "dev_authors": memory.get_setting("dev_authors", None),
        "dev_judge_provider": memory.get_setting("dev_judge_provider", None),
        "dev_min_authors": memory.get_setting("dev_min_authors", None),
        "dev_max_authors": memory.get_setting("dev_max_authors", None),
        "dev_exploration_rate": memory.get_setting("dev_exploration_rate", None),
    }

    decision = policy.decide(provider_stats=provider_stats, settings=dev_settings)

    author_providers = list(getattr(decision, "author_providers", []) or [])
    judge_provider = getattr(decision, "judge_provider", None)
    mode = getattr(decision, "mode", None)

    report["policy"] = {
        "mode": mode,
        "author_providers": author_providers,
        "judge_provider": judge_provider,
        "reason": "Policy decided providers and mode.",
    }

    # 1) Generate candidate patches
    author_prompt = build_author_prompt(request=request, context=context)
    author_outputs: List[Dict[str, Any]] = []

    for provider_name in author_providers:
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
            patch_text = _strip_markdown_fences(client.generate(author_prompt))
            author_outputs.append({"provider": provider_name, "success": True, "patch": patch_text})
            memory.update_provider_stats(provider_name, success=True)
        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})
            memory.update_provider_stats(provider_name, success=False)

    report["authors"] = author_outputs

    # 2) Judge/select
    chosen_patch = ""
    rationale = ""

    judge_client = provider_map.get(judge_provider) if judge_provider else None
    if judge_client:
        try:
            judge_prompt = build_judge_prompt(request=request, context=context, candidates=author_outputs)
            judge_text = _strip_markdown_fences(judge_client.generate(judge_prompt))

            if _looks_like_unified_diff(judge_text):
                chosen_patch = judge_text
                rationale = f"Judge({judge_provider}) returned a unified diff directly."
            else:
                chosen_patch, rationale = _choose_first_valid_patch(author_outputs)

            report["judge"] = {
                "provider": judge_provider,
                "success": True,
                "rationale": rationale,
            }
            memory.update_provider_stats(judge_provider, success=True)

        except Exception as e:
            chosen_patch, fallback_reason = _choose_first_valid_patch(author_outputs)
            report["judge"] = {
                "provider": judge_provider,
                "success": False,
                "rationale": f"Judge failed: {e}. {fallback_reason}",
            }
            memory.update_provider_stats(judge_provider, success=False)
    else:
        chosen_patch, rationale = _choose_first_valid_patch(author_outputs)
        report["judge"] = {
            "provider": judge_provider,
            "success": False,
            "rationale": f"No judge provider available. {rationale}",
        }

    report["chosen_patch"] = chosen_patch
    return report


# -----------------------------
# Step 2: apply patch (yes/no confirmation)
# -----------------------------

def apply_dev_patch(repo_root: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the patch contained in report["chosen_patch"].

    Mutates and returns 'report' by setting report["apply"] results.

    Invariants enforced:
    - repo must be clean pre-apply
    - HEAD must equal report["base_commit"] pre-apply
    """
    repo_root = str(Path(repo_root).resolve())

    # Ensure apply block exists
    report.setdefault("apply", {})
    report["apply"].setdefault("attempted", False)
    report["apply"].setdefault("applied", False)
    report["apply"].setdefault("changed_files", [])
    report["apply"].setdefault("validation_ok", False)
    report["apply"].setdefault("validation_output", "")
    report["apply"].setdefault("error", "")

    patch = report.get("chosen_patch", "") or ""
    base_commit = report.get("base_commit", "") or ""

    report["apply"]["attempted"] = True

    if not patch:
        report["apply"]["applied"] = False
        report["apply"]["error"] = "No patch produced."
        return report

    if not _looks_like_unified_diff(patch):
        report["apply"]["applied"] = False
        report["apply"]["error"] = "Chosen patch did not look like a unified diff. Refusing to apply."
        return report

    if not base_commit:
        report["apply"]["applied"] = False
        report["apply"]["error"] = "Missing base_commit in report; cannot enforce reproducibility."
        return report

    try:
        _require_clean_tree(repo_root, phase="pre-apply")
        _require_head(repo_root, base_commit, phase="pre-apply")

        backups = apply_patches(repo_root=repo_root, diff_text=patch)
        changed_files = list(backups.keys())

        report["apply"]["changed_files"] = changed_files
        report["apply"]["applied"] = True

        ok, out = py_compile_files(repo_root=repo_root, changed_paths=changed_files)
        report["apply"]["validation_ok"] = ok
        report["apply"]["validation_output"] = out

        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
