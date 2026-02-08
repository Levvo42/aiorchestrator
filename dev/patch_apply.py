from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List


_INDEX_OK = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(?: \d{6})?$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a\/(.+) b\/(.+)$")
_META_PREFIXES = (
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "copy from ",
    "copy to ",
    "rename from ",
    "rename to ",
    "similarity index ",
    "dissimilarity index ",
    "index ",
)


def _validate_unified_diff(diff_text: str) -> None:
    t = (diff_text or "").strip()
    if not t:
        raise RuntimeError("Refusing to apply patch: empty diff.")

    lines = t.splitlines()
    first_diff_idx = next((i for i, l in enumerate(lines) if l.startswith("diff --git ")), None)
    if first_diff_idx is None:
        raise RuntimeError("Refusing to apply patch: missing 'diff --git' header.")
    if first_diff_idx > 0:
        raise RuntimeError("Refusing to apply patch: leading garbage before first diff header.")

    i = 0
    in_block = False
    while i < len(lines):
        line = lines[i]

        if _HUNK_RE.match(line) and not in_block:
            raise RuntimeError("Refusing to apply patch: orphan hunk outside diff block.")

        if line.startswith("diff --git "):
            if not _DIFF_GIT_RE.match(line):
                raise RuntimeError(f"Refusing to apply patch: malformed diff header '{line}'.")
            in_block = True
            saw_minus = False
            saw_plus = False
            saw_hunk = False

            i += 1
            while i < len(lines) and not lines[i].startswith("diff --git "):
                l = lines[i]
                if l.startswith("--- "):
                    if saw_hunk:
                        raise RuntimeError("Refusing to apply patch: '---' appears after hunks started.")
                    if not (l.startswith("--- a/") or l.startswith("--- /dev/null")):
                        raise RuntimeError(f"Refusing to apply patch: malformed '---' line '{l}'.")
                    saw_minus = True
                elif l.startswith("+++ "):
                    if saw_hunk:
                        raise RuntimeError("Refusing to apply patch: '+++' appears after hunks started.")
                    if not (l.startswith("+++ b/") or l.startswith("+++ /dev/null")):
                        raise RuntimeError(f"Refusing to apply patch: malformed '+++' line '{l}'.")
                    saw_plus = True
                elif _HUNK_RE.match(l):
                    if not (saw_minus and saw_plus):
                        raise RuntimeError("Refusing to apply patch: hunk before file headers.")
                    saw_hunk = True
                elif saw_hunk:
                    if not (l.startswith(" ") or l.startswith("+") or l.startswith("-") or l.startswith("\\")):
                        raise RuntimeError(f"Refusing to apply patch: unexpected line after hunks '{l}'.")
                else:
                    if l.startswith("index "):
                        if not _INDEX_OK.match(l.strip()):
                            raise RuntimeError(f"Refusing to apply patch: malformed index line '{l}'.")
                    elif any(l.startswith(p) for p in _META_PREFIXES):
                        pass
                    elif l.startswith("Binary files ") or l.startswith("GIT binary patch"):
                        raise RuntimeError("Refusing to apply patch: binary patches not supported.")
                    else:
                        raise RuntimeError(f"Refusing to apply patch: unexpected pre-hunk line '{l}'.")

                i += 1
            continue

        i += 1


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
    _validate_unified_diff(diff_text)
    diff_text = _sanitize_diff(diff_text)

    # 0) Preflight for CORRUPT PATCH ONLY (no working-tree modifications)
    # We do NOT abort on normal "patch does not apply" here because 3-way may still succeed.
    chk0 = _run(repo_root, ["--check", "--whitespace=nowarn"], diff_text)
    if chk0.returncode != 0:
        msg = (chk0.stderr or chk0.stdout or "").strip()
        lower = msg.lower()
        if "corrupt patch" in lower or "patch fragment without header" in lower:
            raise RuntimeError(
                "Refusing to apply patch: patch is structurally invalid.\n\n"
                + msg
            )

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
