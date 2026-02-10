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
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional


def _ollama_healthcheck(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if Ollama responds at /api/tags."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 300
    except Exception:
        return False


def _start_ollama_best_effort() -> None:
    """Try to start Ollama in the background (best-effort)."""
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        return

    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }

        if os.name == "nt":
            # Detached background process on Windows.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        subprocess.Popen([ollama_exe, "serve"], **kwargs)
    except Exception:
        # Best-effort: if this fails, caller will handle by retrying healthcheck and raising a clear error.
        return


def ensure_ollama_running(base_url: str) -> bool:
    """
    Ensure Ollama is reachable at base_url.
    If not reachable, try to start it (best-effort), then retry.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return False

    if _ollama_healthcheck(base_url):
        return True

    _start_ollama_best_effort()

    # Retry healthcheck a few times.
    for _ in range(12):
        time.sleep(0.4)
        if _ollama_healthcheck(base_url):
            return True

    return False


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
        if not ensure_ollama_running(self.base_url):
            raise RuntimeError(
                "Ollama is not reachable at OLLAMA_BASE_URL and could not be started automatically. "
                "Install/start Ollama, or set OLLAMA_BASE_URL to the correct address "
                "(often http://127.0.0.1:11434)."
            )

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
