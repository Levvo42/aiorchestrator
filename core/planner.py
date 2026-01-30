"""
planner.py
----------
The planner turns a task + routing decision into an explicit plan.

Long-term:
- This could be LLM-generated plans + tool calls.
- Could output structured JSON actions.
- Could include safety checks and budgets.

For v0:
- If strategy is local_only, we do simple tool command suggestions.
- If strategy uses LLM(s), we build a prompt that asks for:
  - a short plan
  - the final answer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from core.router import RouteDecision


@dataclass
class Plan:
    """
    A plan is:
    - steps: what we intend to do
    - prompts: prompts to send to providers (if any)
    - local_actions: small structured actions for local tools (if any)
    """
    steps: List[str]
    prompts: Dict[str, str]
    local_actions: List[Dict[str, Any]]


class Planner:
    def __init__(self, capabilities: dict) -> None:
        self.capabilities = capabilities

    def make_plan(self, task: str, route: RouteDecision) -> Plan:
        # Local-only plan (very minimal)
        if route.strategy == "local_only":
            return Plan(
                steps=[
                    "Interpret the task as a local tool request",
                    "Execute local tool actions",
                    "Return results"
                ],
                prompts={},
                local_actions=self._infer_local_actions(task)
            )

        # LLM-based plan
        prompts = {}
        for provider in route.providers:
            prompts[provider] = self._build_llm_prompt(task, route)

        steps = [
            f"Use strategy: {route.strategy}",
            f"Consult providers: {', '.join(route.providers)}",
            "Collect responses",
            "Evaluate and return final output"
        ]

        return Plan(
            steps=steps,
            prompts=prompts,
            local_actions=[]
        )

    def _build_llm_prompt(self, task: str, route: RouteDecision) -> str:
        """
        The prompt format is important because later you can parse it:
        - Ask for a short plan first.
        - Then ask for the final answer.

        Keep it simple and consistent in v0.
        """
        return (
            "You are part of a self-hosted AI orchestrator.\n"
            "Task:\n"
            f"{task}\n\n"
            "Return output in two sections:\n"
            "1) PLAN: 3-7 bullet points\n"
            "2) ANSWER: the final response\n"
            "Be concise and practical.\n"
        )

    def _infer_local_actions(self, task: str) -> List[Dict[str, Any]]:
        """
        Very naive local action inference.
        This exists so you can later replace it with a real parser/LLM tool-call format.

        Supported actions in v0:
        - list_dir
        - read_file
        - write_file (not auto-triggered unless you explicitly request writing)
        """
        t = task.lower()

        if "list" in t and "file" in t:
            return [{"tool": "list_dir", "args": {"path": "."}}]

        if "read file" in t or "open file" in t:
            # You will likely want to specify a filename in your prompt.
            # Example: "read file core/router.py"
            parts = task.split()
            # crude: last token as path
            path = parts[-1] if parts else "."
            return [{"tool": "read_file", "args": {"path": path}}]

        return [{"tool": "list_dir", "args": {"path": "."}}]
