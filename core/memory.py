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


DEFAULT_STATE: Dict[str, Any] = {
    "runs": [],
    "provider_stats": {},
    "notes": [],
    "settings": {
        "judge_mode": "auto",
        "judge_provider": None,
        "dev_mode": "auto",
        "dev_authors": None,
        "dev_judge_provider": None,
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
        self.state.setdefault("notes", [])
        self.state.setdefault("settings", {})

        settings = self.state["settings"]
        if not isinstance(settings, dict):
            settings = {}
            self.state["settings"] = settings

        # Judge settings
        settings.setdefault("judge_mode", "auto")
        settings.setdefault("judge_provider", None)

        # Dev settings
        settings.setdefault("dev_mode", "auto")
        settings.setdefault("dev_authors", None)
        settings.setdefault("dev_judge_provider", None)
        settings.setdefault("dev_min_authors", None)
        settings.setdefault("dev_max_authors", None)
        settings.setdefault("dev_exploration_rate", None)

        # Other settings
        settings.setdefault("verbosity", "full")

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
