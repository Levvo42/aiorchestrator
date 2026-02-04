"""
dev/dev_command.py
------------------
Correctness-first dev patching:

- Proposal step must be side-effect free (no memory writes).
- Authors must output a unified diff ONLY.
- We validate patches structurally before judging or applying.
- Apply requires explicit user confirmation (handled in main.py).
- After apply: compile + tests (if available).

Judge contract (STRICT JSON):
{"patch_index": <int>, "rationale": "<short text>"}
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from dev.context import build_context_bundle
from dev.policy import DevPolicy
from dev.prompts import build_author_prompt, build_fix_author_prompt, build_judge_prompt
from dev.patch_apply import apply_patches
from dev.validate import py_compile_files, run_tests_if_available


# ----------------------------
# Patch parsing / validation
# ----------------------------

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a\/(.+) b\/(.+)$")


def _strip_markdown_fences(text: str) -> str:
    t = (text or "").strip()

    # Remove common accidental prompt echo
    if t.startswith("> "):
        t = t[2:].lstrip()

    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _validate_unified_diff(diff_text: str) -> Tuple[bool, str]:
    """
    Structural validation to prevent 'corrupt patch' errors.

    Rules:
    - Must include at least one 'diff --git a/... b/...'
    - For each diff --git block:
        - Must contain '--- a/...' then '+++ b/...'
        - Must contain at least one hunk header '@@ -.. +.. @@'
    - No leading prompt garbage
    """
    t = (diff_text or "").strip()
    if not t:
        return False, "Empty patch."

    lines = t.splitlines()

    # Reject obvious non-diff outputs early
    if not any(l.startswith("diff --git ") for l in lines):
        return False, "Patch missing 'diff --git' header."

    # Reject patches with leading garbage (common LLM failure mode)
    first_diff_idx = next((i for i, l in enumerate(lines) if l.startswith("diff --git ")), None)
    if first_diff_idx and first_diff_idx > 0:
        return False, f"Patch has {first_diff_idx} lines of garbage before first 'diff --git' header."

    i = 0
    blocks = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git "):
            m = _DIFF_GIT_RE.match(line)
            if not m:
                return False, f"Malformed diff header: '{line}'"

            blocks += 1
            saw_minus = False
            saw_plus = False
            saw_hunk = False

            i += 1
            while i < len(lines) and not lines[i].startswith("diff --git "):
                l = lines[i]

                # allow metadata lines like: new file mode, deleted file mode, similar, rename from/to
                if l.startswith("--- "):
                    # must be exactly '--- a/path' or '--- /dev/null'
                    if not (l.startswith("--- a/") or l.startswith("--- /dev/null")):
                        return False, f"Malformed '---' line: '{l}'"
                    saw_minus = True

                if l.startswith("+++ "):
                    # must be exactly '+++ b/path' or '+++ /dev/null'
                    if not (l.startswith("+++ b/") or l.startswith("+++ /dev/null")):
                        return False, f"Malformed '+++' line: '{l}'"
                    saw_plus = True

                if _HUNK_RE.match(l):
                    saw_hunk = True

                # A very common corruption is stray quotes or prose
                # We don't reject '+'/'-' content lines, only obvious prose BEFORE headers.
                i += 1

            if not (saw_minus and saw_plus):
                return False, "Missing '--- a/...' and/or '+++ b/...' in a diff block."
            if not saw_hunk:
                return False, "Missing hunk header '@@ -.. +.. @@' in a diff block."

            continue

        i += 1

    if blocks == 0:
        return False, "No diff blocks found."

    return True, "OK"


def _extract_changed_files(diff_text: str) -> List[str]:
    changed: List[str] = []
    for line in (diff_text or "").splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            b_path = m.group(2)
            if b_path:
                changed.append(b_path)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for p in changed:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _extract_patch_text(patch_item: Any) -> str:
    if isinstance(patch_item, dict):
        return str(patch_item.get("patch", "")).strip()
    return str(patch_item).strip()


def _choose_first_valid_patch(successful_patches: List[Any]) -> Tuple[str, str]:
    for item in successful_patches:
        patch_text = _strip_markdown_fences(_extract_patch_text(item))
        ok, _ = _validate_unified_diff(patch_text)
        if ok:
            return patch_text, "Fallback: selected first structurally valid unified diff."
    return "", "Fallback: no candidate patch was a structurally valid unified diff."


# ----------------------------
# Provider selection helpers
# ----------------------------

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


def _prefer_dev_providers(selected: List[str], provider_map: Dict[str, Any]) -> List[str]:
    """
    If runtime has *_dev providers, map base providers to *_dev equivalents.
    """
    runtime = set(provider_map.keys())
    has_dev = any(p.endswith("_dev") for p in runtime)

    if not has_dev:
        return [p for p in selected if p in runtime]

    out: List[str] = []
    for p in selected:
        if p.endswith("_dev") and p in runtime:
            out.append(p)
            continue
        if p in runtime and not p.endswith("_dev"):
            # If both exist, prefer _dev
            dev = f"{p}_dev"
            out.append(dev if dev in runtime else p)
            continue
        # Map base -> dev
        dev = f"{p}_dev"
        if dev in runtime:
            out.append(dev)

    # Deduplicate preserve order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    return uniq


def _best_available_judge(preferred: str, provider_map: Dict[str, Any]) -> str:
    runtime = sorted(provider_map.keys())
    if not runtime:
        return ""

    if preferred in provider_map:
        return preferred

    # Prefer dev judge if any dev providers exist
    if any(p.endswith("_dev") for p in runtime):
        dev_pref = f"{preferred}_dev"
        if dev_pref in provider_map:
            return dev_pref
        if "openai_dev" in provider_map:
            return "openai_dev"

    return runtime[0]


# ----------------------------
# Public API
# ----------------------------

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

    # Prefer *_dev providers when available
    decision.author_providers = _prefer_dev_providers(decision.author_providers, provider_map)
    decision.judge_provider = _best_available_judge(decision.judge_provider, provider_map)

    # Ensure we have authors
    if not decision.author_providers:
        runtime = sorted(provider_map.keys())
        devs = [p for p in runtime if p.endswith("_dev")]
        decision.author_providers = devs[:2] if devs else runtime[:2]

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

            ok, why = _validate_unified_diff(patch_text)
            if not ok:
                author_outputs.append(
                    {
                        "provider": provider_name,
                        "success": False,
                        "error": f"Rejected invalid diff: {why}",
                    }
                )
                continue

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
                    ok, why = _validate_unified_diff(candidate)
                    if ok:
                        chosen_patch = candidate
                        judge_ok = True
                    else:
                        chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                        judge_rationale = (
                            f"Judge selected patch_index={idx} but patch was invalid ({why}). {fallback_reason}"
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

def run_dev_fix_request(
    repo_root: str,
    failed_report: Dict[str, Any],
    capabilities: dict,
    memory: Any,
    provider_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-propose a patch after a failed apply attempt."""

    original_request = str(failed_report.get("request", "")).strip()
    failed_patch = str(failed_report.get("chosen_patch", "")).strip()
    apply_error = str((failed_report.get("apply", {}) or {}).get("error", "")).strip()

    context = build_context_bundle(repo_root=repo_root, request=original_request)

    policy = DevPolicy(capabilities)
    provider_stats = memory.get_provider_stats()

    decision = policy.decide(
        provider_stats=provider_stats,
        settings={},
        determinism_key=f"{_git_head(repo_root)}\nFIX_APPLY\n{original_request}\n{apply_error[:300]}",
    )

    author_prompt = build_fix_author_prompt(
        request=original_request,
        context=context,
        failed_patch=failed_patch,
        apply_error=apply_error,
    )

    author_outputs = []
    for provider_name in decision.author_providers:
        client = provider_map.get(provider_name)
        if not client:
            continue
        try:
            patch = _strip_markdown_fences(client.generate(author_prompt))
            ok, _ = _validate_unified_diff(patch)
            if ok:
                author_outputs.append(patch)
        except Exception:
            pass

    chosen_patch = author_outputs[0] if author_outputs else ""

    return {
        "request": original_request,
        "context": context,
        "policy": {
            "mode": decision.mode,
            "authors": decision.author_providers,
            "judge": decision.judge_provider,
            "reason": "FIX_APPLY",
        },
        "chosen_patch": chosen_patch,
        "apply": {
            "attempted": False,
            "applied": False,
            "error": "",
        },
    }


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

    ok, why = _validate_unified_diff(patch)
    if not ok:
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = f"Chosen patch invalid: {why}"
        return report

    report["apply"]["attempted"] = True

    try:
        changed_files = _extract_changed_files(patch)
        backups = apply_patches(repo_root=repo_root, diff_text=patch)
        if not changed_files:
            changed_files = list(backups.keys())

        report["apply"]["changed_files"] = changed_files
        report["apply"]["applied"] = True

        python_files = [p for p in changed_files if p.endswith(".py")]
        if python_files:
            compile_ok, compile_out = py_compile_files(repo_root=repo_root, changed_paths=python_files)
        else:
            compile_ok, compile_out = True, "Skipped py_compile (no .py files changed)."

        if compile_ok:
            tests_ok, tests_out, tests_ran = run_tests_if_available(repo_root=repo_root)
        else:
            tests_ok, tests_out, tests_ran = True, "Skipped tests because py_compile failed.", False

        # If validation fails, rollback
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

        if not report["apply"]["validation_ok"]:
            # Rollback: restore files from backups
            rollback_errors = []
            for rel_path, backup_content in backups.items():
                try:
                    target = Path(repo_root) / rel_path
                    target.write_text(backup_content, encoding="utf-8")
                except Exception as e:
                    rollback_errors.append(f"{rel_path}: {e}")
            # Remove newly added files that had no backup
            for rel_path in changed_files:
                if rel_path in backups:
                    continue
                try:
                    target = Path(repo_root) / rel_path
                    if target.exists():
                        if target.is_file():
                            target.unlink()
                        elif target.is_dir():
                            target.rmdir()
                except Exception as e:
                    rollback_errors.append(f"{rel_path}: {e}")

            report["apply"]["applied"] = False
            error_msg = (
                "Validation failed after apply. Files have been restored from backups.\n"
                f"Validation output:\n{report['apply']['validation_output']}"
            )
            if rollback_errors:
                error_msg += f"\n\nRollback errors:\n" + "\n".join(rollback_errors)

            report["apply"]["error"] = error_msg

        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
