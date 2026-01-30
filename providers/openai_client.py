"""
openai_client.py
----------------
Wrapper around OpenAI via LangChain.

This file hides provider-specific details and exposes one clean function:
- generate(text) -> string
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class OpenAIClient:
    """
    A minimal OpenAI client.

    Env vars used:
    - OPENAI_API_KEY (required)
    - OPENAI_MODEL (default model for normal tasks)
    - OPENAI_DEV_MODEL (recommended model for dev/self-patching tasks)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2) -> None:
        # If model is explicitly provided, use it.
        # Otherwise fall back to OPENAI_MODEL (cheap/default).
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature


    def generate(self, prompt: str) -> str:
        """
        Generate a response from OpenAI.

        Raises:
            RuntimeError if OPENAI_API_KEY is missing.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env file.")

        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
        )

        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content
