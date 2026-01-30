"""
memory.py
---------
An efficient JSON-backed memory store.

Purpose:
- Save run logs (task, plan, outputs, judge choice, final answer).
- Track basic provider stats (success/failure counts).
- Store persistent settings (like judge mode/provider).

This stays simple and readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class MemoryStore:
    """
    A simple "memory" object that reads/writes a JSON file.

    Later you can replace this with:
    - SQLite
    - Postgres
    - Vector DB + embeddings
    """

    def __init__(self, state_path: str = "memory/state.json") -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self.state: Dict[str, Any] = self._load()
        else:
            # Default structure if file doesn't exist yet
            self.state = {
                "runs": [],
                "provider_stats": {},
                "notes": [],
                "settings": {"judge_mode": "auto", "judge_provider": None}
            }
            self._save()

        # Ensure settings exist even if you had an older state.json
        self.state.setdefault("settings", {"judge_mode": "auto", "judge_provider": None})
        self.state["settings"].setdefault("judge_mode", "auto")
        self.state["settings"].setdefault("judge_provider", None)
        self._save()
        self.state["settings"].setdefault("dev_mode", "auto")
        self.state["settings"].setdefault("dev_authors", None)
        self.state["settings"].setdefault("dev_judge_provider", None)
        self.state["settings"].setdefault("dev_min_authors", None)
        self.state["settings"].setdefault("dev_max_authors", None)
        self.state["settings"].setdefault("dev_exploration_rate", None)


    def _load(self) -> Dict[str, Any]:
        """Read JSON from disk."""
        with self.state_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        """Write JSON to disk."""
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

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
    # Settings (Judge control)
    # -----------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a persistent setting stored in memory/state.json."""
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
