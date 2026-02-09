"""
core/env.py
-----------
Environment variable helpers.
"""

from __future__ import annotations

import os
from typing import Iterable

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    if load_dotenv:
        load_dotenv()
    _DOTENV_LOADED = True


def require_env_var(name: str) -> str:
    _ensure_dotenv_loaded()
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Required env var {name} missing or empty. Set it in .env or shell.")
    return value


def validate_required_env_vars(names: Iterable[str]) -> None:
    for name in names:
        require_env_var(name)
