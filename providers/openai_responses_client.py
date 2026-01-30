"""
openai_responses_client.py
--------------------------
OpenAI Responses API client.

Use this for models that are NOT available via Chat Completions / ChatOpenAI,
such as GPT-5.2-Codex (Responses-only per OpenAI docs).

Exposes one clean method:
- generate(prompt: str) -> str
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class OpenAIResponsesClient:
    """
    Minimal wrapper around OpenAI Responses API.

    Env vars used:
    - OPENAI_API_KEY (required)
    - OPENAI_DEV_MODEL (optional; e.g. "gpt-5.2" or "gpt-5.2-codex")
    - OPENAI_DEV_REASONING_EFFORT (optional; e.g. "low"|"medium"|"high"|"xhigh")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_DEV_MODEL", "gpt-5.2")
        self.reasoning_effort = reasoning_effort or os.getenv("OPENAI_DEV_REASONING_EFFORT", None)

        # Create the SDK client once; reuse for all calls
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        """
        Generate a response via /v1/responses.

        Returns:
            The model's output text.

        Raises:
            RuntimeError if OPENAI_API_KEY is missing.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env file.")

        # Build request payload
        payload = {
            "model": self.model,
            "input": prompt,
        }

        # Reasoning effort is supported by reasoning/codex-style models in Responses API
        # (If you set it for models that don't support it, the API may reject it.)
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        resp = self.client.responses.create(**payload)

        # The SDK provides output_text convenience on Responses
        return resp.output_text or ""
