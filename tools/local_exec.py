"""
local_exec.py
-------------
Local tools your agent can use without calling an external AI.

For v0 we keep it simple:
- read_file(path)
- write_file(path, content)
- list_dir(path)

This is the beginning of "do tasks locally first".
"""

from __future__ import annotations

from pathlib import Path


def read_file(path: str) -> str:
    """Read text from a file."""
    p = Path(path)
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> None:
    """Write text to a file (overwrites existing)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def list_dir(path: str = ".") -> list[str]:
    """List files/folders in a directory."""
    p = Path(path)
    return [x.name for x in p.iterdir()]
