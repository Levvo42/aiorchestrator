"""
gemini_client.py
----------------
Wrapper around Google Gemini via LangChain.

This file hides provider-specific details and exposes one clean function:
- generate(text) -> string

Why do this?
- Your agent should not care "how" a provider works; it just asks for text.
- Later, you can add retries, logging, token usage tracking, etc.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient:
    """
    A minimal Gemini client.

    Env vars used:
    - GOOGLE_API_KEY (required)
    - GEMINI_MODEL (optional; defaults to gemini-2.5-flash)
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """
        Generate a response from Gemini.

        Raises:
            RuntimeError if GOOGLE_API_KEY is missing.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is missing. Put it in your .env file.")

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
        )

        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content
