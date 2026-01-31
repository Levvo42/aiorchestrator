"""
dev/patch_apply.py
------------------
Applies unified diff patches to the local filesystem.

v1 improvement:
- Try `git apply` FIRST (more robust; supports fuzzy matching).
- If git apply fails (or git not available), fall back to our strict applier.

Why this matters:
LLM-generated patches often have slightly different context / line numbers.
Your strict applier will refuse those patches (by design).
`git apply` is the correct tool for patch application and will handle drift.

Safety:
- We still refuse to write outside repo_root.
- We still refuse non-unified-diff input (handled earlier in dev_command.py).
- We optionally snapshot target files (best-effort backups) before applying.

Returned backups:
- Dict[path -> old_content] for the files the diff touches (best-effort).
  If a file did not exist before, backup is "".
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class FilePatch:
    path: str
    old_path: Optional[str]
    new_path: Optional[str]
    hunks: List[Tuple[int, int, int, int, List[str]]]  # (old_start, old_len, new_start, new_len, lines)
    is_new_file: bool


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff_text: str) -> List[FilePatch]:
    """
    Parse unified diff into file patches.
    Raises ValueError if the diff doesn't look like a unified diff.
    """
    lines = diff_text.splitlines(keepends=False)
    patches: List[FilePatch] = []

    i = 0
    current_old = None
    current_new = None
    current_path = None
    hunks: List[Tuple[int, int, int, int, List[str]]] = []
    is_new_file = False

    def flush_current():
        nonlocal current_old, current_new, current_path, hunks, is_new_file
        if current_path and (hunks or is_new_file):
            patches.append(
                FilePatch(
                    path=current_path,
                    old_path=current_old,
                    new_path=current_new,
                    hunks=hunks,
                    is_new_file=is_new_file,
                )
            )
        current_old = None
        current_new = None
        current_path = None
        hunks = []
        is_new_file = False

    # Allow both "diff --git" and just "---/+++"
    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git"):
            flush_current()
            i += 1
            continue

        if line.startswith("--- "):
            flush_current()
            current_old = line[4:].strip()

            # Detect new file
            if current_old == "/dev/null":
                is_new_file = True

            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("Invalid diff: expected '+++' after '---'")

            current_new = lines[i][4:].strip()

            # Determine the target path (prefer new path)
            candidate = current_new

            # Strip a/ or b/ prefixes if present
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            if candidate.startswith("a/"):
                candidate = candidate[2:]

            if candidate == "/dev/null":
                # File deletion not supported in this project v0/v1
                raise ValueError("File deletion patches not supported.")

            current_path = candidate
            i += 1
            continue

        # Hunks
        m = HUNK_RE.match(line)
        if m:
            old_start = int(m.group(1))
            old_len = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_len = int(m.group(4) or "1")
            i += 1
            hunk_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith(("@@ ", "--- ", "diff --git")):
                hunk_lines.append(lines[i])
                i += 1
            hunks.append((old_start, old_len, new_start, new_len, hunk_lines))
            continue

        i += 1

    flush_current()

    if not patches:
        raise ValueError("No file patches found. Is this a unified diff?")

    return patches


def _safe_target_path(repo_root: Path, rel_path: str) -> Path:
    """
    Build an absolute path and ensure it stays inside repo_root.
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError(f"Refusing to write outside repo root: {rel_path}")
    return target


def _snapshot_backups(repo_root: str, diff_text: str) -> Dict[str, str]:
    """
    Best-effort backups: read the current content of every file touched by the diff.

    This gives you rollback ability even when we apply via `git apply`.
    """
    root = Path(repo_root).resolve()
    file_patches = parse_unified_diff(diff_text)
    backups: Dict[str, str] = {}

    for fp in file_patches:
        target = _safe_target_path(root, fp.path)

        if fp.is_new_file:
            backups[fp.path] = ""  # new file didn't exist
        else:
            if target.exists():
                backups[fp.path] = target.read_text(encoding="utf-8")
            else:
                # If the patch targets a missing file, record empty; strict fallback will raise.
                backups[fp.path] = ""

    return backups


def _try_git_apply(repo_root: str, diff_text: str) -> Tuple[bool, str]:
    """
    Try applying a patch using git (more robust than our strict applier).
    Returns (ok, message).

    Notes:
    - Requires repo_root to be inside a git repository (git init).
    - Uses --whitespace=nowarn to avoid failures on whitespace differences.
    """
    try:
        # `git apply` can take patch data via stdin with "-".
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=repo_root,
            input=diff_text,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return True, "git apply succeeded"
        return False, (result.stderr or result.stdout or "git apply failed")
    except FileNotFoundError:
        return False, "git not found on PATH"
    except Exception as e:
        return False, f"git apply exception: {e}"


def apply_patches(repo_root: str, diff_text: str) -> Dict[str, str]:
    """
    Apply the diff to files under repo_root.

    Returns:
        backups: dict of {path: old_content} for rollback.

    Behavior:
    1) Snapshot backups (best-effort)
    2) Try `git apply` (robust/fuzzy)
    3) If git apply fails, fall back to strict applier

    Raises:
        ValueError on mismatch or parse errors (strict fallback path).
    """
    # 1) Snapshot backups so caller can rollback later if needed.
    backups = _snapshot_backups(repo_root, diff_text)

    # 2) Try git apply first (best user experience)
    ok, msg = _try_git_apply(repo_root, diff_text)
    if ok:
        return backups  # applied successfully via git

    # If git apply failed, fall back to strict applier.
    # This keeps your original conservative behavior as a safety net.
    # The strict applier will raise ValueError if hunks mismatch.
    # You can comment out fallback if you prefer "git only".
    # print(f"DEBUG: git apply failed, falling back to strict applier. Reason: {msg}")

    # --- STRICT FALLBACK (your original implementation) ---
    root = Path(repo_root).resolve()
    file_patches = parse_unified_diff(diff_text)

    for fp in file_patches:
        target = _safe_target_path(root, fp.path)

        if fp.is_new_file:
            if target.exists():
                raise ValueError(f"Patch wants to create new file but it already exists: {fp.path}")
            new_content = _apply_to_lines([], fp.hunks)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(new_content) + ("\n" if new_content else ""), encoding="utf-8")
            continue

        # Existing file update
        if not target.exists():
            raise ValueError(f"Patch targets missing file: {fp.path}")

        old_text = target.read_text(encoding="utf-8")
        old_lines = old_text.splitlines()
        new_lines = _apply_to_lines(old_lines, fp.hunks)
        target.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return backups


def _apply_to_lines(old_lines: List[str], hunks: List[Tuple[int, int, int, int, List[str]]]) -> List[str]:
    """
    Apply hunks to a list of lines.

    This implementation is strict:
    - It checks that context/removal lines match the current file at the expected location.
    - If mismatch occurs, it raises ValueError.

    Note: old_start/new_start are 1-based line numbers in unified diff.
    """
    lines = old_lines[:]
    offset = 0  # track how insertions/deletions shift subsequent hunk positions

    for (old_start, old_len, new_start, new_len, hunk_lines) in hunks:
        # Convert to 0-based index, applying current offset
        idx = (old_start - 1) + offset

        new_chunk: List[str] = []
        consume_idx = idx

        for hl in hunk_lines:
            if hl == "":
                # Some generators may produce empty strings (malformed).
                raise ValueError("Malformed hunk line (empty).")

            tag = hl[0]
            text = hl[1:]  # rest of the line without prefix

            if tag == " ":
                if consume_idx >= len(lines) or lines[consume_idx] != text:
                    raise ValueError(
                        f"Hunk context mismatch at line {consume_idx + 1}. "
                        f"Expected '{text}', found '{lines[consume_idx] if consume_idx < len(lines) else 'EOF'}'"
                    )
                new_chunk.append(text)
                consume_idx += 1

            elif tag == "-":
                if consume_idx >= len(lines) or lines[consume_idx] != text:
                    raise ValueError(
                        f"Hunk removal mismatch at line {consume_idx + 1}. "
                        f"Expected '{text}', found '{lines[consume_idx] if consume_idx < len(lines) else 'EOF'}'"
                    )
                consume_idx += 1

            elif tag == "+":
                new_chunk.append(text)

            else:
                raise ValueError(f"Unknown hunk tag '{tag}' in line: {hl}")

        before = lines[:idx]
        after = lines[consume_idx:]
        lines = before + new_chunk + after

        consumed_old = consume_idx - idx
        offset += (len(new_chunk) - consumed_old)

    return lines
