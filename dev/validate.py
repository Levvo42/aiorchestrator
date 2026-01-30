"""
dev/validate.py
---------------
Validation after applying a patch.

We keep this minimal:
- Run python compilation check on changed .py files.

Later upgrades:
- run unit tests
- run type checks
- run linting
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple


def py_compile_files(repo_root: str, changed_paths: List[str]) -> Tuple[bool, str]:
    """
    Compile changed Python files. Returns (ok, output).
    """
    root = Path(repo_root).resolve()
    py_files = [p for p in changed_paths if p.endswith(".py")]

    if not py_files:
        return True, "No Python files changed; skipping py_compile."

    # Use python -m py_compile file1 file2 ...
    cmd = ["python", "-m", "py_compile"] + [str((root / p).resolve()) for p in py_files]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = completed.returncode == 0
        output = (completed.stdout or "") + (completed.stderr or "")
        return ok, output.strip() if output else "py_compile OK"
    except Exception as e:
        return False, f"Validation error running py_compile: {e}"
