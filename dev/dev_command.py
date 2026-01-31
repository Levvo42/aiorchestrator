"""
dev/dev_command.py
------------------
Developer loop for self-patching the repo.

Correctness-first invariants:
- Self-patching must be reproducible from repo state.
- Candidate patches must pass `git apply --check` before they are eligible.

Two-step flow (matches main.py):
1) run_dev_request(...) proposes a patch (does NOT apply).
2) apply_dev_patch(repo_root=".", report=...) applies only after user confirmation.

Judge contract (dev/prompts.py):
- build_judge_prompt(request, context, patches) -> prompt
- Judge returns STRICT JSON:
    {"patch_index": <int>, "rationale": "<short text>"}
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
# Patch helpers
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


def _git_apply_check_patch(repo_root: str, patch_text: str) -> Tuple[bool, str]:
    """
    Ground-truth validation:
    A patch is only considered valid if `git apply --check` accepts it.
    """
    try:
        result = subprocess.run(
            ["git", "apply", "--check", "--3way", "--whitespace=nowarn", "-"],
            cwd=repo_root,
            input=patch_text,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return True, "git apply --check ok"
        return False, (result.stderr or result.stdout or "git apply --check failed")
    except FileNotFoundError:
        # If git is unavailable, we cannot enforce this invariant here.
        # In that case, we allow "looks like unified diff" and rely on strict fallback in patch_apply.py.
        return True, "git not found; skipping git apply --check gating"
    except Exception as e:
        return False, f"git apply --check exception: {e}"


def _first_valid_patch_from_authors(repo_root: str, author_outputs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Fallback: choose first author patch that:
    1) looks like unified diff, AND
    2) passes git apply --check (when git exists).
    """
    for item in author_outputs:
        if not item.get("success"):
            continue
        patch = _strip_markdown_fences(item.get("patch", "") or "")
        if not _looks_like_unified_diff(patch):
            continue
        ok, msg = _git_apply_check_patch(repo_root, patch)
        if ok:
            return patch, "Fallback: selected first candidate patch that passed git apply --check."
    return "", "Fallback: no candidate patch passed git apply --check; no patch selected."


def _extract_author_patches(repo_root: str, author_outputs: List[Dict[str, Any]]) -> List[str]:
    """
    Return list of patch texts (in author order), but ONLY those that pass git apply --check (when available).
    """
    patches: List[str] = []
    for item in author_outputs:
        if not item.get("success") or not isinstance(item.get("patch"), str):
            continue
        patch = _strip_markdown_fences(item["patch"])
        if not _looks_like_unified_diff(patch):
            continue
        ok, _ = _git_apply_check_patch(repo_root, patch)
        if ok:
            patches.append(patch)
    return patches


def _parse_judge_json(text: str) -> Tuple[int, str]:
    """
    Parse judge output JSON: {"patch_index": int, "rationale": str}
    Raises ValueError on invalid format.
    """
    raw = (text or "").strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Judge output JSON was not an object.")
    if "patch_index" not in data:
        raise ValueError("Judge JSON missing 'patch_index'.")
    if "rationale" not in data:
        raise ValueError("Judge JSON missing 'rationale'.")

    idx = data["patch_index"]
    rationale = data["rationale"]

    if not isinstance(idx, int):
        raise ValueError("'patch_index' must be an integer.")
    if not isinstance(rationale, str):
        raise ValueError("'rationale' must be a string.")

    return idx, rationale.strip()


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
        "authors": [],
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

    context = build_context_bundle(repo_root=repo_root, request=request)

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
            raw = client.generate(author_prompt)
            patch_text = _strip_markdown_fences(raw)

            # Gate: must look like diff AND pass git apply --check (when git exists)
            if not _looks_like_unified_diff(patch_text):
                author_outputs.append(
                    {"provider": provider_name, "success": False, "error": "Output did not look like a unified diff."}
                )
                memory.update_provider_stats(provider_name, success=False)
                continue

            ok, msg = _git_apply_check_patch(repo_root, patch_text)
            if not ok:
                author_outputs.append(
                    {"provider": provider_name, "success": False, "error": f"Rejected by git apply --check: {msg}"}
                )
                memory.update_provider_stats(provider_name, success=False)
                continue

            author_outputs.append({"provider": provider_name, "success": True, "patch": patch_text})
            memory.update_provider_stats(provider_name, success=True)

        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})
            memory.update_provider_stats(provider_name, success=False)

    report["authors"] = author_outputs

    # Only check-passing patches get judged
    patches = _extract_author_patches(repo_root, author_outputs)

    if not patches:
        # Make it explicit WHY no patch is available (the per-provider errors are in report["authors"])
        report["judge"] = {
            "provider": judge_provider,
            "success": False,
            "rationale": "No candidate patches passed git apply --check; refusing to choose a patch.",
        }
        report["chosen_patch"] = ""
        return report

    # 2) Judge/select by index (STRICT JSON)
    chosen_patch = ""
    rationale = ""

    judge_client = provider_map.get(judge_provider) if judge_provider else None
    if judge_client:
        try:
            judge_prompt = build_judge_prompt(request=request, context=context, patches=patches)
            judge_text = _strip_markdown_fences(judge_client.generate(judge_prompt))

            idx, rationale = _parse_judge_json(judge_text)
            if idx < 0 or idx >= len(patches):
                raise ValueError(f"Judge patch_index out of range: {idx} (0..{len(patches)-1})")

            chosen_patch = patches[idx]
            report["judge"] = {
                "provider": judge_provider,
                "success": True,
                "rationale": rationale or "Judge selected a patch.",
            }
            memory.update_provider_stats(judge_provider, success=True)

        except Exception as e:
            chosen_patch, fallback_reason = _first_valid_patch_from_authors(repo_root, author_outputs)
            report["judge"] = {
                "provider": judge_provider,
                "success": False,
                "rationale": f"Judge failed: {e}. {fallback_reason}",
            }
            if judge_provider:
                memory.update_provider_stats(judge_provider, success=False)
    else:
        chosen_patch, rationale = _first_valid_patch_from_authors(repo_root, author_outputs)
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

    # Extra safety: re-check the chosen patch before applying
    ok, msg = _git_apply_check_patch(repo_root, patch)
    if not ok:
        report["apply"]["applied"] = False
        report["apply"]["error"] = f"Chosen patch rejected by git apply --check. Refusing to apply:\n{msg}"
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

        ok2, out = py_compile_files(repo_root=repo_root, changed_paths=changed_files)
        report["apply"]["validation_ok"] = ok2
        report["apply"]["validation_output"] = out

        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
