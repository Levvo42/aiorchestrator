"""
providers/ollama_client.py
--------------------------
Local Ollama provider wrapper used by your orchestrator.

Exposes:
    generate(prompt: str) -> str

Uses only the Python standard library.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class OllamaClient:
    """
    Simple Ollama HTTP client.

    Env vars used (optional):
      - OLLAMA_BASE_URL (default: http://localhost:11434)
      - OLLAMA_MODEL (default: gpt-oss:20b)
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    def generate(self, prompt: str) -> str:
        """
        Generate a response via Ollama's /api/generate endpoint.

        Returns:
          str: model output text
        """
        system_prefix = (
            "System: Role: local model served by Ollama.\n"
            "System: Do not claim cloud hosting or OpenAI infrastructure.\n\n"
        )
        full_prompt = f"{system_prefix}{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc.reason}. Check OLLAMA_BASE_URL.") from exc

        data = json.loads(body)
        return data.get("response", "")
