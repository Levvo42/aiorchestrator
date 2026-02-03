from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List


_INDEX_OK = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(?: \d{6})?$")


def _sanitize_diff(diff_text: str) -> str:
    out_lines: List[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("index "):
            if not _INDEX_OK.match(line.strip()):
                continue
        out_lines.append(line)
    return ("\n".join(out_lines).rstrip() + "\n")


def _run(repo_root: str, args: List[str], diff_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply"] + args,
        cwd=repo_root,
        input=diff_text,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def _has_rejects(repo_root: str) -> bool:
    root = Path(repo_root).resolve()
    return any(root.rglob("*.rej"))


def apply_patches(repo_root: str, diff_text: str) -> Dict[str, str]:
    repo_root = str(Path(repo_root).resolve())
    diff_text = _sanitize_diff(diff_text)

    # Identify touched files (best-effort)
    changed_files: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                changed_files.append(parts[3][2:])

    # Backup current contents
    backups: Dict[str, str] = {}
    for rel in changed_files:
        p = Path(repo_root) / rel
        if p.exists():
            backups[rel] = p.read_text(encoding="utf-8")

    # 1) Try 3-way first (often succeeds even if --check fails)
    p1 = _run(repo_root, ["--3way", "--whitespace=nowarn"], diff_text)
    if p1.returncode == 0:
        if _has_rejects(repo_root):
            raise RuntimeError("Patch produced rejects (.rej). Refusing partial apply.")
        return backups

    # 2) Try direct apply with recount (helps when line counts drift)
    p2 = _run(repo_root, ["--whitespace=nowarn", "--recount"], diff_text)
    if p2.returncode == 0:
        if _has_rejects(repo_root):
            raise RuntimeError("Patch produced rejects (.rej). Refusing partial apply.")
        return backups

    # 3) If still failing, run --check purely for a clean diagnostic
    chk = _run(repo_root, ["--check", "--whitespace=nowarn"], diff_text)

    raise RuntimeError(
        "Refusing to apply patch. 3-way apply failed, direct apply failed.\n"
        "\n--- 3way error ---\n"
        + (p1.stderr or p1.stdout or "").strip()
        + "\n\n--- direct apply error ---\n"
        + (p2.stderr or p2.stdout or "").strip()
        + "\n\n--- check diagnostic ---\n"
        + (chk.stderr or chk.stdout or "").strip()
    )
