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
8) Validate (compile; tests if available)

IMPORTANT:
- Proposal must be side-effect free (no memory writes / no stats updates).
- Only unified diffs are allowed to be applied.
- Judge must output STRICT JSON:
    {"patch_index": <int>, "rationale": "<short text>"}
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from dev.context import build_context_bundle
from dev.policy import DevPolicy
from dev.prompts import build_author_prompt, build_judge_prompt
from dev.patch_apply import apply_patches
from dev.validate import py_compile_files, run_tests_if_available


def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _strip_markdown_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _looks_like_unified_diff(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "diff --git" in t:
        return True
    if t.startswith("--- ") and "\n+++ " in t:
        return True
    return False


def _extract_patch_text(patch_item: Any) -> str:
    if isinstance(patch_item, dict):
        return str(patch_item.get("patch", "")).strip()
    return str(patch_item).strip()


def _choose_first_valid_patch(successful_patches: List[Any]) -> Tuple[str, str]:
    for item in successful_patches:
        patch_text = _strip_markdown_fences(_extract_patch_text(item))
        if _looks_like_unified_diff(patch_text):
            return patch_text, "Fallback: selected first candidate patch that looks like a unified diff."
    return "", "Fallback: no candidate patch looked like a unified diff; no patch selected."


def _git_head(repo_root: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return (completed.stdout or "").strip() or "nogit"
    except Exception:
        pass
    return "nogit"


def _rewrite_to_dev_providers(selected: List[str], provider_map: Dict[str, Any]) -> List[str]:
    """
    If runtime has *_dev providers, rewrite base provider names to *_dev equivalents.
    This avoids "Authors: []" when capabilities lists openai/claude but runtime only
    exposes openai_dev/claude_dev.
    """
    runtime = set(provider_map.keys())
    has_any_dev = any(k.endswith("_dev") for k in runtime)
    if not has_any_dev:
        # No special rewriting needed; just filter to runtime-available.
        return [p for p in selected if p in runtime]

    rewritten: List[str] = []
    for p in selected:
        if p in runtime:
            rewritten.append(p)
            continue

        dev_name = f"{p}_dev"
        if dev_name in runtime:
            rewritten.append(dev_name)
            continue

        # Some repos might name dev variants differently; last resort: skip.
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in rewritten:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _best_available_judge(preferred: str, provider_map: Dict[str, Any]) -> str:
    runtime = sorted(provider_map.keys())
    if not runtime:
        return ""

    # Prefer rewriting to *_dev if present
    has_any_dev = any(k.endswith("_dev") for k in runtime)
    if has_any_dev:
        if preferred in provider_map:
            return preferred
        if f"{preferred}_dev" in provider_map:
            return f"{preferred}_dev"
        # Prefer openai_dev if it exists
        if "openai_dev" in provider_map:
            return "openai_dev"

    # Otherwise use preferred if exists, else first
    return preferred if preferred in provider_map else runtime[0]


def run_dev_request(
    repo_root: str,
    request: str,
    capabilities: dict,
    memory: Any,
    provider_map: Dict[str, Any],
) -> Dict[str, Any]:
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

    head = _git_head(repo_root)
    determinism_key = f"{head}\n{request.strip()}"

    decision = policy.decide(
        provider_stats=provider_stats,
        settings=dev_settings,
        determinism_key=determinism_key,
    )

    # Rewrite author providers to *_dev equivalents when applicable
    decision.author_providers = _rewrite_to_dev_providers(decision.author_providers, provider_map)

    # Ensure we have at least one author; if policy picked none, fall back to any dev providers.
    if not decision.author_providers:
        runtime = sorted(provider_map.keys())
        devs = [p for p in runtime if p.endswith("_dev")]
        decision.author_providers = devs[:2] if devs else runtime[:2]

    # Choose judge deterministically among available runtime providers
    decision.judge_provider = _best_available_judge(decision.judge_provider, provider_map)

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
        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})

    successful_patches = [o for o in author_outputs if o.get("success") and o.get("patch")]

    # ----------------------------
    # 2) Judge chooses best patch
    # ----------------------------
    judge_rationale = ""
    chosen_patch = ""
    judge_ok = False
    judge_raw_output = ""

    judge_client = provider_map.get(decision.judge_provider)
    candidate_patch_texts = [_extract_patch_text(p) for p in successful_patches]

    judge_prompt = build_judge_prompt(
        request=request,
        context=context,
        patches=candidate_patch_texts,
    )

    if not judge_client or not successful_patches:
        chosen_patch, judge_rationale = _choose_first_valid_patch(successful_patches)
        if not judge_client:
            judge_rationale = (
                f"Judge unavailable: '{decision.judge_provider}' not in provider_map. {judge_rationale}"
            )
        else:
            judge_rationale = f"No successful patches to judge. {judge_rationale}"

    else:
        try:
            judge_output = judge_client.generate(judge_prompt)
            judge_raw_output = judge_output
            judge_json = _safe_json_load(judge_output)

            if judge_json and "patch_index" in judge_json:
                idx = judge_json.get("patch_index")
                judge_rationale = str(judge_json.get("rationale", "")).strip()

                if isinstance(idx, int) and 0 <= idx < len(candidate_patch_texts):
                    candidate = _strip_markdown_fences(candidate_patch_texts[idx].strip())
                    if _looks_like_unified_diff(candidate):
                        chosen_patch = candidate
                        judge_ok = True
                    else:
                        chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                        judge_rationale = (
                            f"Judge selected patch_index={idx}, but it didn't look like a unified diff. "
                            f"{fallback_reason}"
                        )
                else:
                    chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                    judge_rationale = f"Judge returned invalid patch_index={idx}. {fallback_reason}"
            else:
                chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                judge_rationale = (
                    "Judge did not return strict JSON with patch_index; ignored raw judge output. "
                    f"{fallback_reason}\nRaw judge output:\n{judge_output.strip()}"
                )

        except Exception as e:
            chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
            judge_rationale = f"Judge failed: {e}. {fallback_reason}"

    report: Dict[str, Any] = {
        "request": request,
        "context": context,
        "policy": {
            "mode": decision.mode,
            "authors": decision.author_providers,
            "judge": decision.judge_provider,
            "reason": decision.reason,
        },
        "authors": author_outputs,
        "judge": {
            "provider": decision.judge_provider,
            "success": judge_ok,
            "rationale": judge_rationale,
            "raw_output": judge_raw_output.strip() if judge_raw_output else "",
        },
        "chosen_patch": chosen_patch,
        "apply": {
            "attempted": False,
            "applied": False,
            "changed_files": [],
            "validation_ok": False,
            "validation_output": "",
            "tests_ran": False,
            "tests_ok": True,
            "tests_output": "",
            "error": "",
        },
    }

    return report


def apply_dev_patch(repo_root: str, report: Dict[str, Any]) -> Dict[str, Any]:
    patch = (report.get("chosen_patch") or "").strip()
    patch = _strip_markdown_fences(patch)

    report.setdefault("apply", {})
    for k, v in {
        "attempted": False,
        "applied": False,
        "changed_files": [],
        "validation_ok": False,
        "validation_output": "",
        "tests_ran": False,
        "tests_ok": True,
        "tests_output": "",
        "error": "",
    }.items():
        report["apply"].setdefault(k, v)

    if not patch:
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = "No patch available to apply."
        return report

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

        compile_ok, compile_out = py_compile_files(repo_root=repo_root, changed_paths=changed_files)
        tests_ok, tests_out, tests_ran = run_tests_if_available(repo_root=repo_root)

        report["apply"]["tests_ran"] = tests_ran
        report["apply"]["tests_ok"] = tests_ok
        report["apply"]["tests_output"] = tests_out

        report["apply"]["validation_ok"] = bool(compile_ok and tests_ok)
        report["apply"]["validation_output"] = (
            "=== py_compile ===\n"
            + (compile_out or "")
            + "\n\n=== tests ===\n"
            + (tests_out or "")
        ).strip()

        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
