"""
dev/dev_command.py
------------------
Developer loop for self-patching the repo.

Correctness-first invariant (locked down here):
- Self-patching must be reproducible from repo state.

This file enforces:
1) Repo must be clean BEFORE context is built.
2) base_commit is pinned at start of run.
3) Repo must be clean BEFORE apply.
4) HEAD must still equal base_commit BEFORE apply.
If any of these fail, we refuse to apply.

Public API expected by main.py:
- run_dev_request(...)
- apply_dev_patch(...)
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
    """
    Returns (is_clean, status_output).
    status_output is the porcelain output if dirty, else "".
    """
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
# Patch parsing / selection
# -----------------------------

_DIFF_START_RE = re.compile(r"^(diff --git|---\s|\+\+\+\s|@@\s)", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    """
    Remove surrounding ``` or ```diff fences if present.
    Keeps inner content.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        # Remove first fence line (``` or ```diff etc)
        t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t, count=1)
        # Remove last fence
        t = re.sub(r"\n```$", "", t, count=1)
    return t.strip()


def _looks_like_unified_diff(text: str) -> bool:
    """
    Quick heuristic: require at least one known diff marker.
    """
    if not text or not isinstance(text, str):
        return False
    return bool(_DIFF_START_RE.search(text))


def _extract_patch_text(item: Dict[str, Any]) -> str:
    """
    Standardize reading patch text from author/judge outputs.
    """
    if "patch" in item and isinstance(item["patch"], str):
        return item["patch"]
    if "text" in item and isinstance(item["text"], str):
        return item["text"]
    if "output" in item and isinstance(item["output"], str):
        return item["output"]
    return ""


def _choose_first_valid_patch(candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Fallback selection: return first candidate that looks like unified diff.
    """
    successful = [c for c in candidates if c.get("success") and c.get("patch")]
    for item in successful:
        patch_text = _strip_markdown_fences(_extract_patch_text(item))
        if _looks_like_unified_diff(patch_text):
            return patch_text, "Fallback: selected first candidate patch that looks like a unified diff."
    return "", "Fallback: no candidate patch looked like a unified diff; no patch selected."


# -----------------------------
# Public function expected by main.py
# -----------------------------

def apply_dev_patch(repo_root: str, patch: str, base_commit: str) -> Dict[str, Any]:
    """
    Apply a chosen unified-diff patch with strict reproducibility guards.

    Returns an 'apply' report dict:
      {
        attempted: bool,
        applied: bool,
        changed_files: [..],
        validation_ok: bool,
        validation_output: str,
        error: str
      }

    This function enforces:
    - clean tree pre-apply
    - HEAD == base_commit pre-apply
    - unified diff shape
    """
    report: Dict[str, Any] = {
        "attempted": False,
        "applied": False,
        "changed_files": [],
        "validation_ok": False,
        "validation_output": "",
        "error": "",
    }

    if not patch:
        report["attempted"] = True
        report["error"] = "No patch available to apply."
        return report

    if not _looks_like_unified_diff(patch):
        report["attempted"] = True
        report["error"] = "Chosen patch did not look like a unified diff. Refusing to apply."
        return report

    report["attempted"] = True

    try:
        # Invariant checks
        _require_clean_tree(repo_root, phase="pre-apply")
        _require_head(repo_root, base_commit, phase="pre-apply")

        backups = apply_patches(repo_root=repo_root, diff_text=patch)
        changed_files = list(backups.keys())

        report["changed_files"] = changed_files
        report["applied"] = True

        ok, out = py_compile_files(repo_root=repo_root, changed_paths=changed_files)
        report["validation_ok"] = ok
        report["validation_output"] = out

        return report

    except Exception as e:
        report["applied"] = False
        report["error"] = str(e)
        return report


# -----------------------------
# Main dev runner
# -----------------------------

def run_dev_request(
    repo_root: str,
    request: str,
    capabilities: dict,
    memory: Any,
    provider_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a dev request and return a report dict.

    memory: MemoryStore object (used for settings/stats/logging)
    provider_map: {"openai": OpenAIClient(), "claude": ClaudeClient(), ...}
                  each must provide .generate(prompt) -> str
    """
    repo_root = str(Path(repo_root).resolve())

    report: Dict[str, Any] = {
        "request": request,
        "base_commit": "",
        "policy": {},
        "authors": [],
        "judge": {},
        "selection": {},
        "apply": {},
    }

    # ---- Lock invariant: reproducible from repo state ----
    try:
        _require_clean_tree(repo_root, phase="start")
        base_commit = _git_head(repo_root)
        report["base_commit"] = base_commit

        _require_head(repo_root, base_commit, phase="pre-context")
        _require_clean_tree(repo_root, phase="pre-context")
        context = build_context_bundle(repo_root=repo_root, request=request)

    except Exception as e:
        report["apply"] = {
            "attempted": False,
            "applied": False,
            "changed_files": [],
            "validation_ok": False,
            "validation_output": "",
            "error": str(e),
        }
        return report

    # Decide policy (authors + judge) locally
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
    report["policy"] = {
        "mode": getattr(decision, "mode", None),
        "author_providers": getattr(decision, "author_providers", []),
        "judge_provider": getattr(decision, "judge_provider", None),
    }

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
            patch_text = _strip_markdown_fences(patch_text)
            author_outputs.append({"provider": provider_name, "success": True, "patch": patch_text})
            memory.update_provider_stats(provider_name, success=True)
        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})
            memory.update_provider_stats(provider_name, success=False)

    report["authors"] = author_outputs

    # ----------------------------
    # 2) Judge / select best patch
    # ----------------------------
    judge_provider_name = getattr(decision, "judge_provider", None)
    judge_client = provider_map.get(judge_provider_name) if judge_provider_name else None

    chosen_patch = ""
    selection_reason = ""

    if judge_client:
        try:
            judge_prompt = build_judge_prompt(request=request, context=context, candidates=author_outputs)
            judge_text = judge_client.generate(judge_prompt)
            judge_text = _strip_markdown_fences(judge_text)

            if _looks_like_unified_diff(judge_text):
                chosen_patch = judge_text
                selection_reason = f"Judge({judge_provider_name}) returned a unified diff directly."
                report["judge"] = {"success": True, "provider": judge_provider_name, "reason": selection_reason, "patch": chosen_patch}
            else:
                chosen_patch, selection_reason = _choose_first_valid_patch(author_outputs)
                report["judge"] = {
                    "success": True,
                    "provider": judge_provider_name,
                    "reason": "Judge output was not a diff; used fallback selection.",
                    "patch": chosen_patch,
                }

            memory.update_provider_stats(judge_provider_name, success=True)

        except Exception as e:
            chosen_patch, selection_reason = _choose_first_valid_patch(author_outputs)
            report["judge"] = {"success": False, "provider": judge_provider_name, "reason": str(e), "patch": chosen_patch}
            memory.update_provider_stats(judge_provider_name, success=False)
    else:
        chosen_patch, selection_reason = _choose_first_valid_patch(author_outputs)
        report["judge"] = {
            "success": False,
            "provider": judge_provider_name,
            "reason": "No judge provider available; used fallback selection.",
            "patch": chosen_patch,
        }

    report["selection"] = {"reason": selection_reason, "has_patch": bool(chosen_patch)}

    # ----------------------------
    # 3) Apply selected patch
    # ----------------------------
    report["apply"] = apply_dev_patch(
        repo_root=repo_root,
        patch=chosen_patch,
        base_commit=base_commit,
    )
    return report
