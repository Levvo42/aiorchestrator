"""
memory.py
---------
A JSON-backed memory store with a clean dev workflow.

Design:
- memory/state.json          => seed/default (can be tracked in git)
- memory/state.local.json    => runtime writable state (must be gitignored)

Goal:
- Never dirty the repo during normal runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _default_local_judge_bucket() -> Dict[str, Any]:
    return {
        "local_decisions": 0,
        "local_escalations": 0,
        "local_json_total": 0,
        "local_json_valid": 0,
        "local_valid_json_rate": 0.0,
        "local_vs_api_compared": 0,
        "local_vs_api_agreement": 0,
        "local_vs_api_agreement_rate": 0.0,
        "post_apply_local_total": 0,
        "post_apply_local_success": 0,
        "post_apply_local_success_rate": 0.0,
        "post_apply_api_total": 0,
        "post_apply_api_success": 0,
        "post_apply_api_success_rate": 0.0,
        "confidence": 0.0,
    }


def _default_general_routing_bucket() -> Dict[str, Any]:
    return {
        "total_general_prompts": 0,
        "local_only_answers": 0,
        "web_escalations": 0,
        "api_escalations": 0,
        "local_json_total": 0,
        "local_json_valid": 0,
        "local_json_valid_rate": 0.0,
        "confidence": 0.0,
    }


DEFAULT_STATE: Dict[str, Any] = {
    "runs": [],
    "provider_stats": {},
    "local_judge_stats": {
        "overall": _default_local_judge_bucket(),
        "per_intent": {},
        "per_provider": {},
    },
    "general_routing_stats": {
        "overall": _default_general_routing_bucket(),
        "per_intent": {},
    },
    "notes": [],
    "settings": {
        "judge_mode": "auto",
        "judge_provider": None,
        "judge_threshold": 0.90,
        "general_mode": "auto",
        "general_confidence_threshold": 0.90,
        "web_first_threshold": 0.70,
        "dev_mode": "auto",
        "dev_authors": None,
        "dev_judge_provider": None,
        "dev_judge_mode": None,
        "dev_min_authors": None,
        "dev_max_authors": None,
        "dev_exploration_rate": None,
        "verbosity": "full",
    },
}


class MemoryStore:
    """
    A simple memory object that reads/writes JSON.

    Default behavior:
    - Load runtime state from memory/state.local.json if it exists.
    - Else bootstrap from memory/state.json (seed) if it exists.
    - Else start from DEFAULT_STATE.
    - Persist ONLY to memory/state.local.json.

    You can replace this later with SQLite/Postgres/etc.
    """

    def __init__(
        self,
        state_path: str = "memory/state.local.json",
        seed_path: str = "memory/state.json",
    ) -> None:
        self.state_path = Path(state_path)
        self.seed_path = Path(seed_path)

        # Ensure memory/ exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Load order: runtime -> seed -> default
        if self.state_path.exists():
            self.state: Dict[str, Any] = self._load(self.state_path)
        elif self.seed_path.exists():
            self.state = self._load(self.seed_path)
            self._normalize_state()
            self._save()  # write normalized bootstrap to runtime file
        else:
            self.state = json.loads(json.dumps(DEFAULT_STATE))  # deep-ish copy
            self._save()

        # Always normalize/migrate (and ensure runtime file exists)
        self._normalize_state()
        self._save()

    def _load(self, path: Path) -> Dict[str, Any]:
        """Read JSON from disk."""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        """Write JSON to runtime state file only."""
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _normalize_state(self) -> None:
        """
        Ensure required keys exist (migration-safe).
        """
        if not isinstance(self.state, dict):
            self.state = {}

        self.state.setdefault("runs", [])
        self.state.setdefault("provider_stats", {})
        self.state.setdefault("local_judge_stats", {})
        self.state.setdefault("general_routing_stats", {})
        self.state.setdefault("notes", [])
        self.state.setdefault("settings", {})

        settings = self.state["settings"]
        if not isinstance(settings, dict):
            settings = {}
            self.state["settings"] = settings

        # Judge settings
        settings.setdefault("judge_mode", "auto")
        settings.setdefault("judge_provider", None)
        settings.setdefault("judge_threshold", 0.90)
        settings.setdefault("general_mode", "auto")
        settings.setdefault("general_confidence_threshold", 0.90)
        settings.setdefault("web_first_threshold", 0.70)

        # Dev settings
        settings.setdefault("dev_mode", "auto")
        settings.setdefault("dev_authors", None)
        settings.setdefault("dev_judge_provider", None)
        settings.setdefault("dev_judge_mode", None)
        settings.setdefault("dev_min_authors", None)
        settings.setdefault("dev_max_authors", None)
        settings.setdefault("dev_exploration_rate", None)

        # Other settings
        settings.setdefault("verbosity", "full")

        local_stats = self.state.get("local_judge_stats")
        if not isinstance(local_stats, dict):
            local_stats = {}
            self.state["local_judge_stats"] = local_stats

        local_stats.setdefault("overall", _default_local_judge_bucket())
        local_stats.setdefault("per_intent", {})
        local_stats.setdefault("per_provider", {})

        general_stats = self.state.get("general_routing_stats")
        if not isinstance(general_stats, dict):
            general_stats = {}
            self.state["general_routing_stats"] = general_stats

        general_stats.setdefault("overall", _default_general_routing_bucket())
        general_stats.setdefault("per_intent", {})

    # -----------------------
    # Run logging
    # -----------------------

    def add_run(self, run_record: Dict[str, Any]) -> None:
        """Append a run record and persist."""
        self.state["runs"].append(run_record)
        self._save()

    # -----------------------
    # Provider stats
    # -----------------------

    def update_provider_stats(self, provider_name: str, success: bool) -> None:
        """Track basic stats per provider (success/failure counts)."""
        stats = self.state.setdefault("provider_stats", {})
        p = stats.setdefault(provider_name, {"success": 0, "failure": 0})

        if success:
            p["success"] += 1
        else:
            p["failure"] += 1

        self._save()

    def get_provider_stats(self) -> Dict[str, Any]:
        """Return provider stats dictionary."""
        return self.state.get("provider_stats", {})

    # -----------------------
    # Local judge stats
    # -----------------------

    def get_local_judge_stats(self) -> Dict[str, Any]:
        """Return local judge stats dictionary."""
        return self.state.get("local_judge_stats", {})

    def update_local_judge_stats(self, info: Dict[str, Any], apply_result: Optional[Dict[str, Any]] = None) -> None:
        """
        Update local judge stats after a dev apply attempt.

        Expected info shape (keys are optional):
          - intent, local_provider, api_provider
          - local_attempted, local_valid_json, escalated
          - local_patch_index, api_patch_index
          - selected_source ("local"|"api"|...)
        """
        if not isinstance(info, dict):
            return

        local_stats = self.state.setdefault("local_judge_stats", {})
        overall = local_stats.setdefault("overall", _default_local_judge_bucket())
        per_intent = local_stats.setdefault("per_intent", {})
        per_provider = local_stats.setdefault("per_provider", {})

        intent = str(info.get("intent") or "general_judge")
        local_provider = str(info.get("local_provider") or "ollama_local")
        api_provider = str(info.get("api_provider") or "")

        intent_bucket = per_intent.setdefault(intent, _default_local_judge_bucket())
        local_bucket = per_provider.setdefault(local_provider, _default_local_judge_bucket())
        api_bucket = per_provider.setdefault(api_provider, _default_local_judge_bucket()) if api_provider else None

        def _local_buckets():
            return [overall, intent_bucket, local_bucket]

        def _api_buckets():
            buckets = [overall, intent_bucket]
            if api_bucket is not None:
                buckets.append(api_bucket)
            return buckets

        local_attempted = bool(info.get("local_attempted"))
        local_valid_json = bool(info.get("local_valid_json"))
        escalated = bool(info.get("escalated"))
        local_patch_index = info.get("local_patch_index")
        api_patch_index = info.get("api_patch_index")
        selected_source = str(info.get("selected_source") or "")

        if local_attempted:
            for b in _local_buckets():
                b["local_decisions"] += 1
                b["local_json_total"] += 1
                if local_valid_json:
                    b["local_json_valid"] += 1
                if escalated:
                    b["local_escalations"] += 1

        if escalated and isinstance(local_patch_index, int) and isinstance(api_patch_index, int):
            for b in _local_buckets():
                b["local_vs_api_compared"] += 1
                if local_patch_index == api_patch_index:
                    b["local_vs_api_agreement"] += 1

        apply_ok = False
        if isinstance(apply_result, dict):
            apply_ok = bool(apply_result.get("applied")) and bool(apply_result.get("validation_ok"))

        if selected_source == "local":
            for b in _local_buckets():
                b["post_apply_local_total"] += 1
                if apply_ok:
                    b["post_apply_local_success"] += 1
        elif selected_source == "api":
            for b in _api_buckets():
                b["post_apply_api_total"] += 1
                if apply_ok:
                    b["post_apply_api_success"] += 1

        def _update_rates(bucket: Dict[str, Any]) -> None:
            json_total = bucket.get("local_json_total", 0) or 0
            json_valid = bucket.get("local_json_valid", 0) or 0
            bucket["local_valid_json_rate"] = (json_valid / json_total) if json_total else 0.0

            vs_total = bucket.get("local_vs_api_compared", 0) or 0
            vs_agree = bucket.get("local_vs_api_agreement", 0) or 0
            bucket["local_vs_api_agreement_rate"] = (vs_agree / vs_total) if vs_total else 0.0

            local_total = bucket.get("post_apply_local_total", 0) or 0
            local_success = bucket.get("post_apply_local_success", 0) or 0
            bucket["post_apply_local_success_rate"] = (local_success / local_total) if local_total else 0.0

            api_total = bucket.get("post_apply_api_total", 0) or 0
            api_success = bucket.get("post_apply_api_success", 0) or 0
            bucket["post_apply_api_success_rate"] = (api_success / api_total) if api_total else 0.0

        touched = set()
        for b in _local_buckets():
            touched.add(id(b))
            _update_rates(b)
        for b in _api_buckets():
            if id(b) in touched:
                continue
            _update_rates(b)

        # Confidence update (conservative)
        if escalated and isinstance(local_patch_index, int) and isinstance(api_patch_index, int):
            if local_patch_index == api_patch_index and apply_ok:
                for b in _local_buckets():
                    b["confidence"] = min(1.0, float(b.get("confidence", 0.0) or 0.0) + 0.05)
        if selected_source == "local" and not apply_ok:
            for b in _local_buckets():
                b["confidence"] = max(0.0, float(b.get("confidence", 0.0) or 0.0) - 0.30)

        self._save()

    # -----------------------
    # General routing stats
    # -----------------------

    def get_general_routing_stats(self) -> Dict[str, Any]:
        """Return general routing stats dictionary."""
        return self.state.get("general_routing_stats", {})

    def update_general_routing_stats(self, info: Dict[str, Any]) -> None:
        """
        Update general routing stats for local-first routing decisions.

        Expected info shape (keys are optional):
          - intent
          - local_json_valid
          - local_only_answer
          - web_escalated
          - api_escalated
        """
        if not isinstance(info, dict):
            return

        stats = self.state.setdefault("general_routing_stats", {})
        overall = stats.setdefault("overall", _default_general_routing_bucket())
        per_intent = stats.setdefault("per_intent", {})

        intent = str(info.get("intent") or "general")
        intent_bucket = per_intent.setdefault(intent, _default_general_routing_bucket())

        local_json_valid = bool(info.get("local_json_valid"))
        local_only_answer = bool(info.get("local_only_answer"))
        web_escalated = bool(info.get("web_escalated"))
        api_escalated = bool(info.get("api_escalated"))
        invalid_json = bool(info.get("invalid_json"))

        for b in (overall, intent_bucket):
            b["total_general_prompts"] += 1
            b["local_json_total"] += 1
            if local_json_valid:
                b["local_json_valid"] += 1
            if local_only_answer:
                b["local_only_answers"] += 1
            if web_escalated:
                b["web_escalations"] += 1
            if api_escalated:
                b["api_escalations"] += 1

        def _update_rates(bucket: Dict[str, Any]) -> None:
            total = bucket.get("local_json_total", 0) or 0
            valid = bucket.get("local_json_valid", 0) or 0
            bucket["local_json_valid_rate"] = (valid / total) if total else 0.0

        for b in (overall, intent_bucket):
            _update_rates(b)

        for b in (overall, intent_bucket):
            conf = float(b.get("confidence", 0.0) or 0.0)
            if local_only_answer and local_json_valid and not (web_escalated or api_escalated):
                conf = min(1.0, conf + 0.05)
            if invalid_json:
                conf = max(0.0, conf - 0.20)
            elif web_escalated or api_escalated:
                conf = max(0.0, conf - 0.05)
            b["confidence"] = conf

        self._save()

    # -----------------------
    # Settings
    # -----------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a persistent setting stored in runtime memory."""
        return self.state.get("settings", {}).get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a persistent setting and persist."""
        self.state.setdefault("settings", {})
        self.state["settings"][key] = value
        self._save()

    def get_judge_config(self) -> Dict[str, Optional[str]]:
        """
        Convenience helper for judge config.
        Returns:
          - judge_mode: 'auto' or 'fixed'
          - judge_provider: provider name or None
        """
        return {
            "judge_mode": self.get_setting("judge_mode", "auto"),
            "judge_provider": self.get_setting("judge_provider", None),
        }

    def get_verbosity(self) -> str:
        """Return current verbosity level."""
        return self.get_setting("verbosity", "full")

    def set_verbosity(self, level: str) -> None:
        """Set verbosity level."""
        self.set_setting("verbosity", level)
