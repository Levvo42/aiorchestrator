"""
core/trusted_sources.py
-----------------------
Helpers for trusted sources registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_PATH = Path("data/trusted_sources.json")


def load_trusted_sources(path: Path | str = DEFAULT_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"sources": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_trusted_sources(data: Dict[str, Any], path: Path | str = DEFAULT_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_trusted_sources(domain: str, categories: list[str], path: Path | str = DEFAULT_PATH) -> None:
    data = load_trusted_sources(path)
    sources = data.setdefault("sources", [])
    sources.append({"domain": domain, "categories": categories})
    save_trusted_sources(data, path)
