from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List


_INDEX_OK = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(?: \d{6})?$")


def _sanitize_diff(diff_text: str) -> str:
    """
    Remove invalid 'index ...' lines that models sometimes hallucinate (e.g. 'index ..index..').
    'index' lines are optional metadata; git apply does not require them.
    """
    out_lines: List[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("index "):
            if not _INDEX_OK.match(line.strip()):
                continue
        out_lines.append(line)
    # Ensure trailing newline to avoid some patch parsing edge-cases
    return ("\n".join(out_lines).rstrip() + "\n")


def _run_git_apply(repo_root: str, args: List[str], diff_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply"] + args,
        cwd=repo_root,
        input=diff_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _has_rejects(repo_root: str) -> bool:
    root = Path(repo_root).resolve()
    return any(root.rglob("*.rej"))


def apply_patches(repo_root: str, diff_text: str) -> Dict[str, str]:
    """
    Apply a unified diff to the repo. Returns {changed_file: backup_content}.

    Strategy (correctness-first):
    1) Sanitize diff (remove invalid index lines).
    2) Pre-check with `git apply --check` for clearer errors.
    3) Try `git apply --3way` (no --reject allowed with --3way).
    4) Fall back to direct `git apply --reject` (but refuse if rejects produced).
    """
    repo_root = str(Path(repo_root).resolve())
    diff_text = _sanitize_diff(diff_text)

    # Identify touched files (best-effort from diff headers)
    changed_files: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith("b/"):
                    changed_files.append(b_path[2:])

    # Backup current file contents before applying
    backups: Dict[str, str] = {}
    for rel in changed_files:
        p = Path(repo_root) / rel
        if p.exists():
            backups[rel] = p.read_text(encoding="utf-8")

    # (2) Pre-check gives the most helpful "corrupt patch" location
    chk = _run_git_apply(repo_root, ["--check", "--whitespace=nowarn"], diff_text)
    if chk.returncode != 0:
        raise RuntimeError(
            "Refusing to apply invalid patch (git apply --check failed):\n"
            + (chk.stderr or chk.stdout or "").strip()
        )

    # (3) Try 3-way apply first (valid combo)
    p1 = _run_git_apply(repo_root, ["--3way", "--whitespace=nowarn"], diff_text)
    if p1.returncode == 0:
        # No rejects should be produced here, but verify anyway
        if _has_rejects(repo_root):
            raise RuntimeError("Patch produced rejects (.rej). Refusing partial apply.")
        return backups

    # (4) Fallback: direct apply with rejects for debugging (valid combo)
    p2 = _run_git_apply(repo_root, ["--whitespace=nowarn", "--reject"], diff_text)
    if p2.returncode == 0:
        if _has_rejects(repo_root):
            raise RuntimeError("Patch produced rejects (.rej). Refusing partial apply.")
        return backups

    raise RuntimeError(
        "Refusing to apply patch. 3-way apply failed, and direct apply also failed.\n"
        "\n--- 3way error ---\n"
        + (p1.stderr or p1.stdout or "").strip()
        + "\n\n--- direct apply error ---\n"
        + (p2.stderr or p2.stdout or "").strip()
    )
