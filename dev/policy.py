"""
dev/policy.py
-------------
Local policy for selecting:
- how many author models to use
- which author providers
- which judge provider

Design goals:
- Start with MORE authors when data is low.
- Reduce authors as confidence grows.
- Maintain exploration to keep evaluating alternatives.
- Allow manual overrides via settings.

This does NOT implement a fancy learning algorithm yet.
It provides stable structure so you can improve it later without rewriting everything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DevPolicyDecision:
    author_providers: List[str]
    judge_provider: str
    mode: str  # 'auto' or 'fixed'
    reason: str


class DevPolicy:
    def __init__(self, capabilities: dict) -> None:
        self.capabilities = capabilities

        # Defaults from capabilities.json
        dev_cfg = capabilities.get("dev", {})
        self.default_mode = dev_cfg.get("default_mode", "auto")
        self.default_judge_provider = dev_cfg.get("default_judge_provider", "gemini")
        self.default_min_authors = int(dev_cfg.get("min_authors", 2))
        self.default_max_authors = int(dev_cfg.get("max_authors", 3))
        self.default_exploration_rate = float(dev_cfg.get("exploration_rate", 0.25))

    def decide(
        self,
        provider_stats: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> DevPolicyDecision:
        """
        Decide dev authors + judge using:
        - enabled providers in capabilities.json
        - memory provider_stats (success/failure counts)
        - dev settings overrides (fixed/auto, preferred authors/judge, ranges)
        """
        enabled = self._enabled_providers()

        # Settings overrides (stored in memory/state.json)
        mode = (settings.get("dev_mode") or self.default_mode).lower()
        fixed_authors = settings.get("dev_authors")  # list or None
        fixed_judge = settings.get("dev_judge_provider")  # str or None

        min_authors = int(settings.get("dev_min_authors") or self.default_min_authors)
        max_authors = int(settings.get("dev_max_authors") or self.default_max_authors)
        exploration_rate = float(settings.get("dev_exploration_rate") or self.default_exploration_rate)

        # Clamp
        min_authors = max(1, min_authors)
        max_authors = max(min_authors, max_authors)

        # FIXED mode: user has explicitly set authors and/or judge
        if mode == "fixed":
            authors = self._filter_available(fixed_authors, enabled) if fixed_authors else enabled[:min_authors]
            judge = fixed_judge if fixed_judge in enabled else (enabled[0] if enabled else "")
            return DevPolicyDecision(
                author_providers=authors,
                judge_provider=judge,
                mode="fixed",
                reason="Fixed dev mode: user-selected authors/judge (or best-effort fallback)."
            )

        # AUTO mode:
        # 1) Choose judge: prefer default_judge_provider if available, else first enabled
        judge = self.default_judge_provider if self.default_judge_provider in enabled else (enabled[0] if enabled else "")

        # 2) Choose number of authors:
        # If we have low data, use more authors.
        # We'll define "low data" as: total runs across all providers < threshold.
        total_observations = self._total_observations(provider_stats)

        # Simple rule:
        # - Very low data => use max_authors
        # - Moderate data => use min_authors+1
        # - Higher data => use min_authors (plus exploration sometimes)
        if total_observations < 10:
            k = max_authors
            reason_k = f"Low data (total_observations={total_observations}): using max_authors={max_authors}."
        elif total_observations < 30:
            k = min(max_authors, min_authors + 1)
            reason_k = f"Moderate data (total_observations={total_observations}): using k={k} authors."
        else:
            k = min_authors
            reason_k = f"Enough data (total_observations={total_observations}): using min_authors={min_authors}."

        # Exploration: sometimes add one extra author (if available)
        if enabled and random.random() < exploration_rate:
            k = min(len(enabled), k + 1)
            reason_k += f" Exploration triggered (rate={exploration_rate})."

        # 3) Choose which authors:
        # Score providers by their historical "dev usefulness" proxy.
        # For now, we use success/failure as a crude proxy. Later you'll refine.
        scored = sorted(enabled, key=lambda p: self._score_provider(p, provider_stats), reverse=True)

        # Ensure judge can also be an author if it scores high; that's ok.
        authors = scored[:min(k, len(scored))]

        return DevPolicyDecision(
            author_providers=authors,
            judge_provider=judge,
            mode="auto",
            reason=f"Auto dev policy: {reason_k} Authors chosen by success/failure scoring."
        )

    def _enabled_providers(self) -> List[str]:
        providers = self.capabilities.get("providers", {})
        return [name for name, cfg in providers.items() if cfg.get("enabled", False)]

    def _filter_available(self, requested: Optional[List[str]], enabled: List[str]) -> List[str]:
        if not requested:
            return []
        return [p for p in requested if p in enabled]

    def _total_observations(self, provider_stats: Dict[str, Any]) -> int:
        total = 0
        for p, s in provider_stats.items():
            total += int(s.get("success", 0))
            total += int(s.get("failure", 0))
        return total

    def _score_provider(self, provider: str, provider_stats: Dict[str, Any]) -> int:
        """
        Crude scoring:
        +1 per success, -2 per failure.

        Later you can:
        - track dev-specific stats separately
        - track patch acceptance rate
        - track validation pass rate
        """
        s = provider_stats.get(provider, {"success": 0, "failure": 0})
        succ = int(s.get("success", 0))
        fail = int(s.get("failure", 0))
        return succ * 1 - fail * 2
