"""
dev/dev_command.py
------------------
Correctness-first dev patching:

- Proposal step must be side-effect free (no memory writes).
- Authors must output a unified diff ONLY.
- We validate patches structurally before judging or applying.
- Apply requires explicit user confirmation (handled in main.py).
- After apply: compile + tests (if available).

Judge contract (STRICT JSON, API judge):
{"patch_index": <int>, "rationale": "<short text>"}

Local judge contract (STRICT JSON, Ollama):
{
  "patch_index": <int or null>,
  "confidence": <number 0..1>,
  "uncertainty_reasons": ["..."],
  "rationale": "short"
}
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from dev.context import build_context_bundle
from dev.policy import DevPolicy
from dev.prompts import build_author_prompt, build_fix_author_prompt, build_judge_prompt, build_local_judge_prompt
from dev.patch_apply import apply_patches, _normalize_patch_text
from dev.validate import py_compile_files, run_tests_if_available


# python
# Add to the top of dev/dev_command.py if the real functions have different signatures.
# These wrappers accept the parameters used by main.py and forward them.

def run_dev_request(repo_root: str, request: str, capabilities: dict, memory, provider_map=None, **kwargs):
    # forward to actual implementation (replace `_run_dev_request` with the real function name if different)
    return _run_dev_request(repo_root=repo_root, request=request, capabilities=capabilities, memory=memory, provider_map=provider_map, **kwargs)

def run_dev_fix_request(repo_root: str, failed_report: dict, capabilities: dict, memory, provider_map=None, **kwargs):
    # forward to actual implementation (replace `_run_dev_fix_request` with the real function name if different)
    return _run_dev_fix_request(repo_root=repo_root, failed_report=failed_report, capabilities=capabilities, memory=memory, provider_map=provider_map, **kwargs)

def apply_dev_patch(repo_root: str, report: dict, **kwargs):
    # forward to actual implementation (replace `_apply_dev_patch` with the real function name if different)
    return _apply_dev_patch(repo_root=repo_root, report=report, **kwargs)


# ----------------------------
# Patch parsing / validation
# ----------------------------

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_HUNK_PARSE_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a\/(.+) b\/(.+)$")
_INDEX_RE = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(?: \d{6})?$")
_META_PREFIXES = (
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "copy from ",
    "copy to ",
    "rename from ",
    "rename to ",
    "similarity index ",
    "dissimilarity index ",
    "index ",
)

LOCAL_JUDGE_PROVIDER = "ollama_local"
LOCAL_JUDGE_RISKY_PATHS = ("core/", "dev/", "providers/", "main.py")

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

def _normalize_unified_diff(raw_text: str) -> str:
    """Extract the diff portion from noisy model output.

    We accept that some models may include code fences or a short prelude.
    The returned diff always ends with a newline.
    """
    t = _strip_markdown_fences(raw_text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    lines = (t or "").splitlines()
    first = next((i for i, l in enumerate(lines) if l.startswith("diff --git ")), None)
    if first is None:
        return (t or "").strip()

    lines = lines[first:]

    def _looks_like_diff_line(l: str) -> bool:
        if not l:
            return True
        if l.startswith((
            "diff --git ",
            "--- ",
            "+++ ",
            "@@ ",
            " ",
            "+",
            "-",
            "Binary files ",
            "GIT binary patch",
            "\\ No newline at end of file",
        )):
            return True
        if any(l.startswith(p) for p in _META_PREFIXES):
            return True
        return False

    last = None
    for i in range(len(lines) - 1, -1, -1):
        if _looks_like_diff_line(lines[i].rstrip("\n")):
            last = i
            break
    if last is not None:
        lines = lines[: last + 1]

    return "\n".join(lines).strip("\n") + "\n"


def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _infer_intent(request: str, capabilities: dict) -> str:
    t = (request or "").lower()
    judge_cfg = capabilities.get("judge", {})
    intent_keywords = judge_cfg.get("task_intent_keywords", {})

    for intent, keywords in intent_keywords.items():
        for kw in keywords:
            if kw.lower() in t:
                return intent
    return "general_judge"


def _normalize_judge_mode(mode: Optional[str]) -> str:
    if not mode:
        return "auto"
    m = str(mode).strip().lower()
    if m in ("auto", "local_only", "api_only"):
        return m
    if m == "fixed":
        return "api_only"
    return "auto"


def _clamp_threshold(value: Any, default: float = 0.90) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _get_learned_confidence(stats: Dict[str, Any], intent: str, provider: str) -> float:
    per_intent = stats.get("per_intent", {}).get(intent, {})
    per_provider = stats.get("per_provider", {}).get(provider, {})
    intent_conf = float(per_intent.get("confidence", 0.0) or 0.0)
    provider_conf = float(per_provider.get("confidence", 0.0) or 0.0)
    return min(intent_conf, provider_conf)


def _parse_local_judge_output(raw_output: str) -> Tuple[Optional[Dict[str, Any]], str]:
    data = _safe_json_load(raw_output)
    if not isinstance(data, dict):
        return None, "invalid_json"

    patch_index = data.get("patch_index")
    if patch_index is not None and not isinstance(patch_index, int):
        return None, "invalid_json"

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        return None, "invalid_json"
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        return None, "invalid_json"

    uncertainty_reasons = data.get("uncertainty_reasons")
    if not isinstance(uncertainty_reasons, list) or not all(isinstance(r, str) for r in uncertainty_reasons):
        return None, "invalid_json"

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        return None, "invalid_json"

    return {
        "patch_index": patch_index,
        "confidence": confidence,
        "uncertainty_reasons": uncertainty_reasons,
        "rationale": rationale.strip(),
    }, ""


def _patch_touches_risky_paths(patch_text: str) -> bool:
    for path in _extract_changed_files(patch_text):
        if path == "main.py":
            return True
        for prefix in LOCAL_JUDGE_RISKY_PATHS:
            if path.startswith(prefix):
                return True
    return False


def _evaluate_local_judge_output(
    raw_output: str,
    candidate_patch_texts: List[str],
    threshold: float,
    learned_confidence: float,
) -> Dict[str, Any]:
    decision: Dict[str, Any] = {
        "valid_json": False,
        "patch_index": None,
        "confidence": 0.0,
        "model_confidence": 0.0,
        "learned_confidence": learned_confidence,
        "uncertainty_reasons": [],
        "rationale": "",
        "escalate": True,
        "escalation_reason": "invalid_json",
        "risky_paths": False,
    }

    parsed, error = _parse_local_judge_output(raw_output)
    if error:
        return decision

    decision["valid_json"] = True
    decision["patch_index"] = parsed["patch_index"]
    decision["model_confidence"] = parsed["confidence"]
    decision["uncertainty_reasons"] = parsed["uncertainty_reasons"]
    decision["rationale"] = parsed["rationale"]

    confidence_used = min(parsed["confidence"], learned_confidence)
    decision["confidence"] = confidence_used

    if parsed["patch_index"] is None:
        decision["escalation_reason"] = "patch_index_null"
        return decision

    if not (0 <= parsed["patch_index"] < len(candidate_patch_texts)):
        decision["escalation_reason"] = "invalid_patch_index"
        return decision

    if confidence_used < threshold:
        decision["escalation_reason"] = "confidence_below_threshold"
        return decision

    if parsed["uncertainty_reasons"]:
        decision["escalation_reason"] = "uncertainty_reasons"
        return decision

    candidate = _strip_markdown_fences(candidate_patch_texts[parsed["patch_index"]].strip())
    decision["risky_paths"] = _patch_touches_risky_paths(candidate)
    if decision["risky_paths"] and confidence_used < 1.0:
        decision["escalation_reason"] = "risky_paths"
        return decision

    decision["escalate"] = False
    decision["escalation_reason"] = ""
    return decision


def _run_api_judge(
    judge_client: Any,
    judge_prompt: str,
    candidate_patch_texts: List[str],
    successful_patches: List[Dict[str, Any]],
) -> Tuple[str, str, bool, str, Optional[int]]:
    judge_rationale = ""
    chosen_patch = ""
    judge_ok = False
    judge_raw_output = ""
    api_patch_index: Optional[int] = None

    if not judge_client or not successful_patches:
        chosen_patch, judge_rationale = _choose_first_valid_patch(successful_patches)
        if not judge_client:
            judge_rationale = f"Judge unavailable; {judge_rationale}"
        else:
            judge_rationale = f"No successful patches; {judge_rationale}"
        return chosen_patch, judge_rationale, judge_ok, judge_raw_output, api_patch_index

    try:
        judge_output = judge_client.generate(judge_prompt)
        judge_raw_output = judge_output
        judge_json = _safe_json_load(judge_output)

        if judge_json and "patch_index" in judge_json:
            idx = judge_json.get("patch_index")
            judge_rationale = str(judge_json.get("rationale", "")).strip()
            api_patch_index = idx if isinstance(idx, int) else None

            if isinstance(idx, int) and 0 <= idx < len(candidate_patch_texts):
                candidate = _strip_markdown_fences(candidate_patch_texts[idx].strip())
                ok, why = _validate_unified_diff(candidate)
                if ok:
                    chosen_patch = candidate
                    judge_ok = True
                else:
                    chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                    judge_rationale = (
                        f"Judge selected patch_index={idx}, but patch invalid ({why}). {fallback_reason}"
                    )
            else:
                chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                judge_rationale = f"Judge returned invalid patch_index={idx}; {fallback_reason}"
        else:
            chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
            judge_rationale = (
                "Judge output invalid JSON; ignored. "
                f"{fallback_reason}\nRaw judge output:\n{judge_output.strip()}"
            )

    except Exception as e:
        chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
        judge_rationale = f"Judge error: {e}. {fallback_reason}"

    return chosen_patch, judge_rationale, judge_ok, judge_raw_output, api_patch_index


def _validate_unified_diff(diff_text: str) -> Tuple[bool, str]:
    """
    Structural validation to prevent git-apply errors like:
      - 'corrupt patch'
      - 'patch fragment without header'

    Assumes _normalize_unified_diff(...) already:
      - strips markdown fences
      - removes leading/trailing non-diff text
      - returns only the diff starting at first 'diff --git'
    """
    t = _normalize_unified_diff(diff_text).strip()
    if not t:
        return False, "Patch is empty."

    lines = t.splitlines()
    if not any(l.startswith("diff --git ") for l in lines):
        return False, "Patch missing 'diff --git' header."

    i = 0
    while i < len(lines):
        line = lines[i]

        # Disallow orphan hunks (hunk headers can't exist outside a file diff block)
        if _HUNK_RE.match(line):
            return False, "Orphan hunk '@@' outside any diff block."

        if not line.startswith("diff --git "):
            return False, f"Unexpected line before first diff header: '{line}'"

        # --- Begin diff block ---
        if not _DIFF_GIT_RE.match(line):
            return False, f"Malformed diff header: '{line}'"

        saw_minus = False
        saw_plus = False
        saw_hunk = False
        in_hunk = False
        hunk_old_remaining = 0
        hunk_new_remaining = 0

        i += 1
        while i < len(lines) and not lines[i].startswith("diff --git "):
            l = lines[i]

            if l.startswith("--- "):
                if saw_hunk:
                    return False, "Malformed diff: '---' appears after hunks started."
                if not (l.startswith("--- a/") or l.startswith("--- /dev/null")):
                    return False, f"Malformed '---' line: '{l}'"
                saw_minus = True

            elif l.startswith("+++ "):
                if saw_hunk:
                    return False, "Malformed diff: '+++' appears after hunks started."
                if not (l.startswith("+++ b/") or l.startswith("+++ /dev/null")):
                    return False, f"Malformed '+++' line: '{l}'"
                saw_plus = True

            elif _HUNK_RE.match(l):
                if not (saw_minus and saw_plus):
                    return False, "Orphan hunk '@@' before file headers '---'/'+++'."
                m_hunk = _HUNK_PARSE_RE.match(l)
                if not m_hunk:
                    return False, f"Malformed hunk header: '{l}'"

                old_count = int(m_hunk.group(2) or "1")
                new_count = int(m_hunk.group(4) or "1")
                hunk_old_remaining = old_count
                hunk_new_remaining = new_count
                in_hunk = True
                saw_hunk = True

            elif in_hunk:
                if l.startswith("\\ No newline at end of file"):
                    pass
                elif l.startswith(" "):
                    hunk_old_remaining -= 1
                    hunk_new_remaining -= 1
                elif l.startswith("-"):
                    hunk_old_remaining -= 1
                elif l.startswith("+"):
                    hunk_new_remaining -= 1
                else:
                    return False, f"Unexpected line inside hunk: '{l}'"

                if hunk_old_remaining < 0 or hunk_new_remaining < 0:
                    return False, "Hunk line counts do not match header."
                if hunk_old_remaining == 0 and hunk_new_remaining == 0:
                    in_hunk = False

            else:
                # Pre-hunk metadata we allow; reject anything else.
                if l.startswith("index "):
                    if not _INDEX_RE.match(l.strip()):
                        return False, f"Malformed index line: '{l}'"
                elif any(l.startswith(p) for p in _META_PREFIXES):
                    pass
                elif l.startswith("Binary files ") or l.startswith("GIT binary patch"):
                    return False, "Binary patches are not supported."
                else:
                    return False, f"Unexpected line before hunks: '{l}'"

            i += 1

        # End diff block checks
        if in_hunk:
            return False, "Hunk line counts do not match header."
        if not (saw_minus and saw_plus):
            return False, "Missing '--- a/...' and/or '+++ b/...' in a diff block."
        if not saw_hunk:
            return False, "Missing hunk header '@@ -.. +.. @@' in a diff block."

        # continue outer while at current i (either next diff block or end)
        continue

    return True, "OK"


def _check_patch_applies(repo_root: str, diff_text: str) -> Tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "--recount"],
            cwd=repo_root,
            input=diff_text,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
    except Exception as e:
        return False, str(e)
    if completed.returncode != 0:
        msg = (completed.stderr or completed.stdout or "").strip()
        return False, msg or "Patch does not apply cleanly; check context."
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
            return patch_text, "Fallback: selected first valid unified diff."
    return "", "Fallback: no candidate patch was a valid unified diff."


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


def _best_available_api_judge(preferred: str, provider_map: Dict[str, Any]) -> str:
    runtime = sorted(p for p in provider_map.keys() if p != LOCAL_JUDGE_PROVIDER)
    if not runtime:
        return ""
    if preferred in runtime:
        return preferred
    if any(p.endswith("_dev") for p in runtime):
        dev_pref = f"{preferred}_dev"
        if dev_pref in runtime:
            return dev_pref
        if "openai_dev" in runtime:
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

    raw_judge_mode = memory.get_setting("dev_judge_mode", None)
    if raw_judge_mode is None:
        raw_judge_mode = memory.get_setting("judge_mode", "auto")
    judge_mode = _normalize_judge_mode(raw_judge_mode)
    judge_threshold = _clamp_threshold(memory.get_setting("judge_threshold", 0.90), default=0.90)
    judge_intent = _infer_intent(request, capabilities)
    local_judge_stats = memory.get_local_judge_stats()
    learned_confidence = _get_learned_confidence(local_judge_stats, judge_intent, LOCAL_JUDGE_PROVIDER)

    head = _git_head(repo_root)
    determinism_key = f"{head}\n{request.strip()}"

    decision = policy.decide(
        provider_stats=provider_stats,
        settings=dev_settings,
        determinism_key=determinism_key,
    )

    # Prefer *_dev providers when available
    decision.author_providers = _prefer_dev_providers(decision.author_providers, provider_map)
    api_judge_provider = _best_available_api_judge(decision.judge_provider, provider_map)

    # Ensure we have authors
    if not decision.author_providers:
        runtime = sorted(provider_map.keys())
        devs = [p for p in runtime if p.endswith("_dev")]
        decision.author_providers = devs[:2] if devs else runtime[:2]

    # ----------------------------
    # 1) Generate candidate patches
    # ----------------------------
    print(f"[1/4] Generate patches: {len(decision.author_providers)} author(s)")

    author_outputs: List[Dict[str, Any]] = []
    author_prompt = build_author_prompt(request=request, context=context)

    for provider_name in decision.author_providers:
        client = provider_map.get(provider_name)
        if not client:
            author_outputs.append(
                {
                    "provider": provider_name,
                    "success": False,
                    "error": f"Provider '{provider_name}' not available. Configured: {list(provider_map.keys())}",
                }
            )
            continue

        try:
            patch_text = client.generate(author_prompt)
            # Normalize patch candidate before any validation
            normalized = _normalize_patch_text(patch_text)
            if normalized.strip() == "NO_DIFF":
                patch_text = "NO_DIFF"
            else:
                patch_text = normalized

            # Use normalized patch_text for all subsequent checks
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
            ok, why = _check_patch_applies(repo_root=repo_root, diff_text=patch_text)
            if not ok:
                author_outputs.append(
                    {
                        "provider": provider_name,
                        "success": False,
                        "error": f"Rejected non-applying diff: {why}",
                    }
                )
                continue

            author_outputs.append({"provider": provider_name, "success": True, "patch": patch_text})

        except Exception as e:
            author_outputs.append({"provider": provider_name, "success": False, "error": str(e)})

    # Only patches that are normalized and validated
    successful_patches = [o for o in author_outputs if o.get("success") and o.get("patch") and o.get("patch") != "NO_DIFF"]

    # ----------------------------
    # 2) Judge chooses best patch
    # ----------------------------
    print(f"[2/4] Judge ({api_judge_provider}): select from {len(successful_patches)} candidate(s)")

    judge_rationale = ""
    chosen_patch = ""
    judge_ok = False
    judge_raw_output = ""
    judge_provider_used = ""
    judge_summary = ""
    api_patch_index: Optional[int] = None

    candidate_patch_texts = [_extract_patch_text(p) for p in successful_patches]
    api_judge_client = provider_map.get(api_judge_provider) if api_judge_provider else None
    local_judge_client = provider_map.get(LOCAL_JUDGE_PROVIDER)

    local_judge_raw = ""
    local_judge_decision: Dict[str, Any] = {
        "attempted": False,
        "valid_json": False,
        "patch_index": None,
        "confidence": 0.0,
        "model_confidence": 0.0,
        "learned_confidence": learned_confidence,
        "uncertainty_reasons": [],
        "rationale": "",
        "escalated": False,
        "escalation_reason": "",
        "risky_paths": False,
    }

    selected_source = ""

    api_judge_prompt = build_judge_prompt(
        request=request,
        context=context,
        patches=candidate_patch_texts,
    )

    if not successful_patches:
        chosen_patch, judge_rationale = _choose_first_valid_patch(successful_patches)
        judge_provider_used = api_judge_provider or ""
        selected_source = "fallback"
        judge_summary = "Judge: api reason=no_candidates"
    elif judge_mode == "api_only":
        chosen_patch, judge_rationale, judge_ok, judge_raw_output, api_patch_index = _run_api_judge(
            judge_client=api_judge_client,
            judge_prompt=api_judge_prompt,
            candidate_patch_texts=candidate_patch_texts,
            successful_patches=successful_patches,
        )
        judge_provider_used = api_judge_provider or ""
        selected_source = "api"
        judge_summary = "Judge: api reason=api_only"
    else:
        if local_judge_client:
            local_judge_decision["attempted"] = True
            local_prompt = build_local_judge_prompt(
                request=request,
                context=context,
                patches=candidate_patch_texts,
            )
            local_judge_raw = local_judge_client.generate(local_prompt)
            local_eval = _evaluate_local_judge_output(
                raw_output=local_judge_raw,
                candidate_patch_texts=candidate_patch_texts,
                threshold=judge_threshold,
                learned_confidence=learned_confidence,
            )
            local_judge_decision.update(local_eval)
            local_judge_decision["escalated"] = local_eval.get("escalate", True)
            local_judge_decision["escalation_reason"] = local_eval.get("escalation_reason", "")

        if judge_mode == "local_only":
            judge_provider_used = LOCAL_JUDGE_PROVIDER
            selected_source = "local"
            local_judge_decision["escalated"] = False
            local_judge_decision["escalation_reason"] = ""
            if local_judge_decision["valid_json"] and isinstance(local_judge_decision["patch_index"], int):
                idx = local_judge_decision["patch_index"]
                candidate = _strip_markdown_fences(candidate_patch_texts[idx].strip())
                ok, why = _validate_unified_diff(candidate)
                if ok:
                    chosen_patch = candidate
                    judge_ok = True
                    judge_rationale = local_judge_decision.get("rationale", "")
                else:
                    chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                    judge_rationale = f"Local judge selected invalid patch ({why}). {fallback_reason}"
                    selected_source = "fallback"
            else:
                chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                judge_rationale = f"Local judge output invalid JSON. {fallback_reason}"
                selected_source = "fallback"

            judge_summary = (
                f"Judge: local confidence={local_judge_decision['confidence']:.2f} "
                f"threshold={judge_threshold:.2f} escalated=no"
            )
        else:
            if not local_judge_client:
                local_judge_decision["escalated"] = True
                local_judge_decision["escalation_reason"] = "local_unavailable"

            if local_judge_decision["escalated"]:
                chosen_patch, judge_rationale, judge_ok, judge_raw_output, api_patch_index = _run_api_judge(
                    judge_client=api_judge_client,
                    judge_prompt=api_judge_prompt,
                    candidate_patch_texts=candidate_patch_texts,
                    successful_patches=successful_patches,
                )
                judge_provider_used = api_judge_provider or ""
                selected_source = "api"
                reason = local_judge_decision.get("escalation_reason") or "uncertain"
                judge_summary = f"Judge: local->api reason={reason}"
            else:
                idx = local_judge_decision["patch_index"]
                candidate = _strip_markdown_fences(candidate_patch_texts[idx].strip())
                ok, why = _validate_unified_diff(candidate)
                if ok:
                    chosen_patch = candidate
                    judge_ok = True
                    judge_rationale = local_judge_decision.get("rationale", "")
                else:
                    chosen_patch, fallback_reason = _choose_first_valid_patch(successful_patches)
                    judge_rationale = f"Local judge selected invalid patch ({why}). {fallback_reason}"
                    selected_source = "fallback"
                judge_provider_used = LOCAL_JUDGE_PROVIDER
                selected_source = "local"
                judge_summary = (
                    f"Judge: local confidence={local_judge_decision['confidence']:.2f} "
                    f"threshold={judge_threshold:.2f} escalated=no"
                )

    print(judge_summary)
    print(f"[3/4] Patch selected. Validation: {'OK' if judge_ok else 'fallback'}")

    report: Dict[str, Any] = {
        "request": request,
        "context": context,
        "policy": {
            "mode": decision.mode,
            "authors": decision.author_providers,
            "judge": api_judge_provider,
            "reason": decision.reason,
        },
        "authors": author_outputs,
        "judge": {
            "provider": judge_provider_used,
            "success": judge_ok,
            "rationale": judge_rationale,
            "raw_output": judge_raw_output.strip() if judge_raw_output else "",
        },
        "local_judge": {
            "mode": judge_mode,
            "intent": judge_intent,
            "threshold": judge_threshold,
            "local_provider": LOCAL_JUDGE_PROVIDER,
            "api_provider": api_judge_provider,
            "summary": judge_summary,
            "selected_source": selected_source,
            "local": {
                "attempted": local_judge_decision.get("attempted", False),
                "valid_json": local_judge_decision.get("valid_json", False),
                "patch_index": local_judge_decision.get("patch_index"),
                "confidence": local_judge_decision.get("confidence", 0.0),
                "model_confidence": local_judge_decision.get("model_confidence", 0.0),
                "learned_confidence": local_judge_decision.get("learned_confidence", learned_confidence),
                "uncertainty_reasons": local_judge_decision.get("uncertainty_reasons", []),
                "rationale": local_judge_decision.get("rationale", ""),
                "raw_output": local_judge_raw.strip() if local_judge_raw else "",
                "escalated": local_judge_decision.get("escalated", False),
                "escalation_reason": local_judge_decision.get("escalation_reason", ""),
                "risky_paths": local_judge_decision.get("risky_paths", False),
            },
            "api": {
                "patch_index": api_patch_index,
            },
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

    print(f"[4/4] Proposal complete. Ready for review.")
    
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
    prev_fix = failed_report.get("fix", {}) or {}
    original_patch = str(prev_fix.get("original_patch") or failed_patch).strip()
    original_error = str(prev_fix.get("original_error") or apply_error).strip()
    fix_depth = int(prev_fix.get("depth", 0) or 0) + 1

    context = build_context_bundle(repo_root=repo_root, request=original_request)

    policy = DevPolicy(capabilities)
    provider_stats = memory.get_provider_stats()

    dev_min = memory.get_setting("dev_min_authors", None) or 1
    dev_max = memory.get_setting("dev_max_authors", None) or 2
    dev_max = min(int(dev_max), 2)
    dev_min = min(int(dev_min), dev_max)

    decision = policy.decide(
        provider_stats=provider_stats,
        settings={
            "dev_mode": memory.get_setting("dev_mode", "auto"),
            "dev_authors": memory.get_setting("dev_authors", None),
            "dev_judge_provider": memory.get_setting("dev_judge_provider", None),
            "dev_min_authors": dev_min,
            "dev_max_authors": dev_max,
            "dev_exploration_rate": memory.get_setting("dev_exploration_rate", None),
        },
        determinism_key=f"{_git_head(repo_root)}\nFIX_APPLY\n{original_request}\n{original_error[:300]}",
    )

    decision.author_providers = _prefer_dev_providers(decision.author_providers, provider_map)
    if not decision.author_providers:
        runtime = sorted(provider_map.keys())
        devs = [p for p in runtime if p.endswith("_dev")]
        decision.author_providers = devs[:2] if devs else runtime[:2]
    decision.author_providers = decision.author_providers[:dev_max]

    author_prompt = build_fix_author_prompt(
        request=original_request,
        context=context,
        failed_patch=original_patch,
        apply_error=original_error,
    )

    author_outputs = []
    for provider_name in decision.author_providers:
        client = provider_map.get(provider_name)
        if not client:
            continue
        try:
            patch = client.generate(author_prompt)
            # Normalize patch candidate before any validation
            normalized = _normalize_patch_text(patch)
            if normalized.strip() == "NO_DIFF":
                patch = "NO_DIFF"
            else:
                patch = normalized
            ok, _ = _validate_unified_diff(patch)
            if ok and patch != "NO_DIFF":
                ok, _ = _check_patch_applies(repo_root=repo_root, diff_text=patch)
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
        "fix": {
            "depth": fix_depth,
            "original_patch": original_patch,
            "original_error": original_error,
            "source_patch": failed_patch,
            "source_error": apply_error,
        },
        "apply": {
            "attempted": False,
            "applied": False,
            "error": "",
        },
    }


def apply_dev_patch(repo_root: str, report: Dict[str, Any]) -> Dict[str, Any]:
    print("[Apply] Start")
    
    patch = (report.get("chosen_patch") or "").strip()
    # Normalize patch before apply
    normalized = _normalize_patch_text(patch)
    if normalized.strip() == "NO_DIFF":
        patch = "NO_DIFF"
    else:
        patch = normalized

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

    if not patch or patch == "NO_DIFF":
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = "No patch to apply."
        return report

    ok, why = _validate_unified_diff(patch)
    if not ok:
        report["apply"]["attempted"] = True
        report["apply"]["applied"] = False
        report["apply"]["error"] = f"Selected patch invalid: {why}"
        return report

    report["apply"]["attempted"] = True

    print("[Apply] Applying patch")
    
    try:
        changed_files = _extract_changed_files(patch)
        backups = apply_patches(repo_root=repo_root, diff_text=patch)
        if not changed_files:
            changed_files = list(backups.keys())

        report["apply"]["changed_files"] = changed_files
        report["apply"]["applied"] = True

        print(f"[Apply] Applied. Validating {len([p for p in changed_files if p.endswith('.py')])} Python file(s)")
        
        python_files = [p for p in changed_files if p.endswith(".py")]
        if python_files:
            compile_ok, compile_out = py_compile_files(repo_root=repo_root, changed_paths=python_files)
        else:
            compile_ok, compile_out = True, "Skipped py_compile (no .py files changed)."

        if compile_ok:
            tests_ok, tests_out, tests_ran = run_tests_if_available(repo_root=repo_root)
        else:
            tests_ok, tests_out, tests_ran = True, "Tests skipped; py_compile failed.", False

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
            print("[Apply] Validation failed; rolling back")
            
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
                "Validation failed after apply. Backups restored.\n"
                f"Validation output:\n{report['apply']['validation_output']}"
            )
            if rollback_errors:
                error_msg += f"\n\nRollback errors:\n" + "\n".join(rollback_errors)

            report["apply"]["error"] = error_msg

        print(f"[Apply] Complete. Validation: {'OK' if report['apply']['validation_ok'] else 'FAILED'}")
        
        return report

    except Exception as e:
        report["apply"]["applied"] = False
        report["apply"]["error"] = str(e)
        return report
