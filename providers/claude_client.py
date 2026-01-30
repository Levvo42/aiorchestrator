"""
providers/claude_client.py
--------------------------
Anthropic (Claude) provider wrapper used by your orchestrator.

Exposes:
    generate(prompt: str) -> str

Uses LangChain's ChatAnthropic integration.
"""

from __future__ import annotations

import os
from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
from langchain_core.messages import HumanMessage


class ClaudeClient:
    """
    Simple Claude wrapper.

    You can swap model via:
      - ANTHROPIC_DEV_MODEL in .env
      - or by passing model=... in __init__
    """

    def __init__(self, model: str | None = None, temperature: float = 0.2) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")

        self.model = model or os.getenv("ANTHROPIC_DEV_MODEL", "claude-sonnet-4-5")

        # ChatAnthropic is LangChain's Claude wrapper. :contentReference[oaicite:6]{index=6}
        self.llm = ChatAnthropic(
            model=self.model,
            temperature=temperature,
            anthropic_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a single response from Claude.

        Returns:
          str: model output text
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
