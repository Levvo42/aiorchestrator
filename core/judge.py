"""
judge.py
--------
This module decides how to pick a final answer when multiple worker models respond.

Your goal:
- The system should NOT just print multiple answers.
- It should choose (or merge) answers into ONE final answer.
- Which judge model is used should be configurable and eventually dynamic.

We support two modes:
1) fixed: you choose the judge provider (e.g. "openai")
2) auto: the system selects a judge provider locally using scoring rules

Later upgrades:
- Add "agreement scoring" between providers
- Add rubric-based judging
- Add cost budgets
- Use local model for judge selection (or judge itself)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class JudgeDecision:
    judge_provider: str
    mode: str  # 'auto' or 'fixed'
    intent: str  # e.g. 'code_judge', 'comparison_judge', 'general_judge'
    score_table: Dict[str, int]
    rationale: str


class JudgeRouter:
    """
    Local (non-LLM) judge selection using:
    - capabilities.json provider metadata (judge_strengths)
    - task keywords (intent)
    - memory provider stats (success/failure)
    - whether required env keys are available

    This keeps your "AI" as the control plane.
    """

    def __init__(self, capabilities: dict) -> None:
        self.capabilities = capabilities
        self.judge_cfg = capabilities.get("judge", {})
        self.intent_keywords = self.judge_cfg.get("task_intent_keywords", {})

    def infer_intent(self, task: str) -> str:
        """
        Infer what kind of judging is needed based on keywords.
        Falls back to 'general_judge'.
        """
        t = task.lower()

        # Check each intent bucket
        for intent, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw.lower() in t:
                    return intent

        return "general_judge"

    def select_judge_provider(
        self,
        task: str,
        provider_stats: Dict[str, Any],
        mode: str,
        fixed_provider: Optional[str],
    ) -> JudgeDecision:
        """
        Choose which provider should act as judge.
        Returns a JudgeDecision including scores and reasoning.
        """
        intent = self.infer_intent(task)

        # If fixed mode and provider specified, use it (if enabled and has key)
        if mode == "fixed" and fixed_provider:
            if self._provider_is_available(fixed_provider):
                return JudgeDecision(
                    judge_provider=fixed_provider,
                    mode=mode,
                    intent=intent,
                    score_table={fixed_provider: 999},
                    rationale="Fixed judge mode: user-selected provider."
                )

        # Otherwise auto mode: score and pick best
        scores: Dict[str, int] = {}
        for provider_name in self._enabled_providers():
            if not self._provider_is_available(provider_name):
                # Not available = not scorable
                continue

            score = 0
            meta = self.capabilities["providers"][provider_name]

            # 1) If provider is strong as a judge for this intent, big bonus
            judge_strengths = meta.get("judge_strengths", [])
            if intent in judge_strengths:
                score += 5

            # 2) Otherwise, if provider is generally a judge, small bonus
            if "general_judge" in judge_strengths:
                score += 2

            # 3) Reliability bonus/penalty from memory stats
            stat = provider_stats.get(provider_name, {"success": 0, "failure": 0})
            succ = int(stat.get("success", 0))
            fail = int(stat.get("failure", 0))

            # A very simple reliability measure:
            # +1 per success, -2 per failure (failures hurt more than successes help)
            score += succ * 1
            score -= fail * 2

            # 4) Cheap/faster models get a slight bias (optional)
            # This helps keep costs down long-term.
            if meta.get("cost_tier") == "low":
                score += 1
            if meta.get("latency_tier") == "fast":
                score += 1

            scores[provider_name] = score

        # If no provider scored (no keys), fallback to default provider if possible
        default_provider = self.judge_cfg.get("default_provider", "gemini")
        if not scores:
            chosen = default_provider if self._provider_is_available(default_provider) else ""
            return JudgeDecision(
                judge_provider=chosen,
                mode="auto",
                intent=intent,
                score_table={},
                rationale="No available providers scored; falling back to default provider (if available)."
            )

        # Pick highest score
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        return JudgeDecision(
            judge_provider=chosen,
            mode="auto",
            intent=intent,
            score_table=scores,
            rationale="Auto judge selection using provider judge_strengths + reliability stats + cheap/fast bias."
        )

    def _enabled_providers(self) -> List[str]:
        providers = self.capabilities.get("providers", {})
        return [name for name, cfg in providers.items() if cfg.get("enabled", False)]

    def _provider_is_available(self, provider_name: str) -> bool:
        """
        Provider is considered available if:
        - it is enabled in capabilities.json
        - its required env key exists (e.g. OPENAI_API_KEY)
        """
        providers = self.capabilities.get("providers", {})
        cfg = providers.get(provider_name)
        if not cfg or not cfg.get("enabled", False):
            return False

        env_key = cfg.get("env_key_required")
        if env_key and not os.getenv(env_key):
            return False

        return True


class Judge:
    """
    The Judge takes worker outputs and returns ONE final answer.

    In v0:
    - We ask the chosen judge provider to evaluate and synthesize.
    - This is easiest and works well.
    - The selection of which provider is judge is controlled locally (JudgeRouter).
    """

    def __init__(self, capabilities: dict, provider_map: Dict[str, Any]) -> None:
        self.capabilities = capabilities
        self.provider_map = provider_map
        self.router = JudgeRouter(capabilities)

    def judge(
        self,
        task: str,
        worker_outputs: List[Dict[str, Any]],
        provider_stats: Dict[str, Any],
        mode: str,
        fixed_provider: Optional[str],
    ) -> Tuple[JudgeDecision, str]:
        """
        Returns:
          - JudgeDecision (who judged + why)
          - final_answer (string)
        """
        decision = self.router.select_judge_provider(
            task=task,
            provider_stats=provider_stats,
            mode=mode,
            fixed_provider=fixed_provider,
        )

        # If no judge provider is available, return a simple fallback
        if not decision.judge_provider:
            fallback = self._fallback_merge(worker_outputs)
            return decision, fallback

        judge_client = self.provider_map.get(decision.judge_provider)
        if not judge_client:
            fallback = self._fallback_merge(worker_outputs)
            return decision, fallback

        # Build a judging prompt
        prompt = self._build_judge_prompt(task, worker_outputs)

        # Ask the chosen judge model to synthesize a final answer
        final = judge_client.generate(prompt)
        return decision, final

    def _build_judge_prompt(self, task: str, worker_outputs: List[Dict[str, Any]]) -> str:
        """
        Create a clear, structured prompt for the judge model.
        """
        # Only include successful worker outputs
        successful = [o for o in worker_outputs if o.get("success") and o.get("text")]

        # If nothing to judge, ask it to answer directly
        if not successful:
            return (
                "You are the judge model for an AI orchestrator.\n"
                "No worker outputs were available.\n"
                f"Task: {task}\n\n"
                "Provide the best possible answer."
            )

        # Build a numbered list of worker answers
        answers_block = ""
        for i, o in enumerate(successful, start=1):
            answers_block += f"\n[Answer {i} from {o['provider']}]\n{o['text']}\n"

        return (
            "You are the judge model for an AI orchestrator.\n"
            "Your job:\n"
            "- Produce ONE final answer that best satisfies the task.\n"
            "- If answers conflict, explain briefly which is more reliable and why.\n"
            "- If answers agree, merge them into a cleaner, stronger response.\n"
            "- Be practical and avoid fluff.\n\n"
            f"Task:\n{task}\n\n"
            f"Worker answers:\n{answers_block}\n"
            "Return ONLY the final answer (no extra sections)."
        )

    def _fallback_merge(self, worker_outputs: List[Dict[str, Any]]) -> str:
        """
        If we cannot use any judge model, do a very simple local fallback:
        - Return the first successful output.
        """
        for o in worker_outputs:
            if o.get("success") and o.get("text"):
                return o["text"]
        return "No worker outputs were available, and no judge model could be used."
