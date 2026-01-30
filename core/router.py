"""
router.py
---------
The router decides *how* to solve the task before doing any work.

Long-term goal:
- This becomes a smarter decision-maker that can use:
  - heuristics
  - provider performance stats
  - even an LLM-based router

For v0:
- We do understandable rules.
- We can choose:
  - local_only
  - llm_single (one provider)
  - llm_multi (consult multiple providers on purpose)
  - hybrid (local tools + LLM)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List


Strategy = Literal["local_only", "llm_single", "llm_multi", "hybrid"]


@dataclass
class RouteDecision:
    strategy: Strategy
    providers: List[str]
    reason: str


class Router:
    """
    Simple rule-based router.

    It does NOT "try providers until one works".
    It decides up front which approach it wants.
    """

    def __init__(self, capabilities: dict) -> None:
        self.capabilities = capabilities

        # Rules are stored in capabilities.json so you can tweak without editing code.
        self.local_first_keywords = set(
            k.lower() for k in capabilities.get("routing_rules", {}).get("local_first_keywords", [])
        )
        self.multi_model_keywords = set(
            k.lower() for k in capabilities.get("routing_rules", {}).get("multi_model_keywords", [])
        )

    def decide(self, task: str) -> RouteDecision:
        t = task.lower()

        # 1) If the user asks for obvious filesystem actions, do local-only or hybrid.
        if any(k in t for k in self.local_first_keywords):
            return RouteDecision(
                strategy="local_only",
                providers=[],
                reason="Task looks like a local file/directory operation."
            )

        # 2) If the user explicitly wants comparison/validation, consult multiple on purpose.
        if any(k in t for k in self.multi_model_keywords):
            return RouteDecision(
                strategy="llm_multi",
                providers=self._enabled_providers(),
                reason="Task asks for comparison/validation; consult multiple models."
            )

        # 3) Default: use one good general provider.
        # Prefer Gemini for general chat by default in this v0 (you can change this).
        enabled = self._enabled_providers()
        preferred = "gemini" if "gemini" in enabled else (enabled[0] if enabled else "")

        if preferred:
            return RouteDecision(
                strategy="llm_single",
                providers=[preferred],
                reason=f"Default single-provider strategy using {preferred}."
            )

        # If no providers are enabled, fallback to local_only (but many tasks will fail).
        return RouteDecision(
            strategy="local_only",
            providers=[],
            reason="No external providers enabled; local only."
        )

    def _enabled_providers(self) -> list[str]:
        """Return providers that are enabled in capabilities.json."""
        providers = self.capabilities.get("providers", {})
        enabled = [name for name, cfg in providers.items() if cfg.get("enabled", False)]
        return enabled
