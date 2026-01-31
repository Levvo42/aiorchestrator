"""
dev/patch_apply.py
------------------
Applies unified diff patches to the local filesystem.

Correctness-first:
- Refuse to touch anything outside an explicit allowlist surface.
- Try `git apply --3way` first for robust patching.
- Fall back to strict applier if needed.

Allowlist surface:
- main.py
- core/**
- dev/**
- providers/**
- tools/**
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


# Patch-visible allowlist (must match dev/context.py surface)
_ALLOWED_TOP_DIRS = {"core", "dev", "providers", "tools"}
_ALLOWED_TOP_FILES = {"main.py"}


def _is_allowed_relpath(rel_path: str) -> bool:
    """
    Return True if rel_path is inside allowed patch-visible surface.
    """
    rel = rel_path.replace("\\", "/").lstrip("./")
    if rel in _ALLOWED_TOP_FILES:
        return True
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return False
    return parts[0] in _ALLOWED_TOP_DIRS


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

    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git"):
            flush_current()
            i += 1
            continue

        if line.startswith("--- "):
            flush_current()
            current_old = line[4:].strip()

            if current_old == "/dev/null":
                is_new_file = True

            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("Invalid diff: expected '+++' after '---'")

            current_new = lines[i][4:].strip()

            candidate = current_new
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            if candidate.startswith("a/"):
                candidate = candidate[2:]

            if candidate == "/dev/null":
                raise ValueError("File deletion patches not supported.")

            current_path = candidate
            i += 1
            continue

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


def _enforce_allowlist(file_patches: List[FilePatch]) -> None:
    """
    Refuse patches that touch files outside the patch-visible allowlist.
    """
    for fp in file_patches:
        if not _is_allowed_relpath(fp.path):
            raise ValueError(
                f"Refusing to patch non-allowed path: {fp.path}. "
                f"Allowed: main.py, core/**, dev/**, providers/**, tools/**"
            )


def _snapshot_backups(repo_root: str, diff_text: str) -> Dict[str, str]:
    """
    Best-effort backups: read the current content of every file touched by the diff.
    """
    root = Path(repo_root).resolve()
    file_patches = parse_unified_diff(diff_text)
    _enforce_allowlist(file_patches)

    backups: Dict[str, str] = {}
    for fp in file_patches:
        target = _safe_target_path(root, fp.path)
        if fp.is_new_file:
            backups[fp.path] = ""
        else:
            backups[fp.path] = target.read_text(encoding="utf-8") if target.exists() else ""
    return backups


def _try_git_apply(repo_root: str, diff_text: str) -> Tuple[bool, str]:
    """
    Try applying a patch using git (robust).
    Uses --3way to reduce mismatches but still fails if drift is too big.
    """
    try:
        result = subprocess.run(
            ["git", "apply", "--3way", "--whitespace=error", "-"],
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
    1) Parse diff and enforce allowlist
    2) Snapshot backups (best-effort)
    3) Try `git apply` first
    4) If git apply fails, fall back to strict applier
    """
    root = Path(repo_root).resolve()
    file_patches = parse_unified_diff(diff_text)
    _enforce_allowlist(file_patches)

    backups = _snapshot_backups(repo_root, diff_text)

    ok, msg = _try_git_apply(repo_root, diff_text)
    if ok:
        return backups

    # STRICT FALLBACK
    for fp in file_patches:
        target = _safe_target_path(root, fp.path)

        if fp.is_new_file:
            if target.exists():
                raise ValueError(f"Patch wants to create new file but it already exists: {fp.path}")
            new_content = _apply_to_lines([], fp.hunks)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(new_content) + ("\n" if new_content else ""), encoding="utf-8")
            continue

        if not target.exists():
            raise ValueError(f"Patch targets missing file: {fp.path}")

        old_text = target.read_text(encoding="utf-8")
        old_lines = old_text.splitlines()
        new_lines = _apply_to_lines(old_lines, fp.hunks)
        target.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return backups


def _apply_to_lines(old_lines: List[str], hunks: List[Tuple[int, int, int, int, List[str]]]) -> List[str]:
    """
    Strict hunk applier: context/removal must match exactly.
    """
    lines = old_lines[:]
    offset = 0

    for (old_start, old_len, new_start, new_len, hunk_lines) in hunks:
        idx = (old_start - 1) + offset

        new_chunk: List[str] = []
        consume_idx = idx

        for hl in hunk_lines:
            if hl == "":
                raise ValueError("Malformed hunk line (empty).")

            tag = hl[0]
            text = hl[1:]

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
