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

import re
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

    cmd = ["python", "-m", "py_compile"] + [str((root / p).resolve()) for p in py_files]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = completed.returncode == 0
        output = (completed.stdout or "") + (completed.stderr or "")
        return ok, output.strip() if output else "py_compile OK"
    except Exception as e:
        return False, f"Validation error running py_compile: {e}"


def run_tests_if_available(repo_root: str) -> Tuple[bool, str, bool]:
    """Run pytest if it looks available for this repo.

    Returns:
        (ok, output, ran)

    - ok is True if tests passed OR if tests were skipped.
    - ran is True only if pytest actually ran.
    """
    root = Path(repo_root).resolve()

    has_tests_dir = (root / "tests").exists()
    has_pytest_ini = (root / "pytest.ini").exists() or (root / "tox.ini").exists()

    has_pyproject_pytest = False
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            txt = pyproject.read_text(encoding="utf-8")
            if re.search(r"\bpytest\b", txt):
                has_pyproject_pytest = True
        except Exception:
            has_pyproject_pytest = False

    if not (has_tests_dir or has_pytest_ini or has_pyproject_pytest):
        return True, "No test configuration detected; skipping tests.", False

    try:
        version = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if version.returncode != 0:
            out = (version.stdout or "") + (version.stderr or "")
            return True, f"pytest not available; skipping tests.\n{out.strip()}", False
    except Exception as e:
        return True, f"pytest not available; skipping tests. ({e})", False

    try:
        completed = subprocess.run(
            ["python", "-m", "pytest", "-q"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (completed.stdout or "") + (completed.stderr or "")
        ok = completed.returncode == 0
        return ok, out.strip() if out else ("pytest OK" if ok else "pytest FAILED"), True
    except Exception as e:
        return False, f"Validation error running pytest: {e}", True
