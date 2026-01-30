"""
dev/patch_apply.py
------------------
Applies unified diff patches to the local filesystem with safety checks.

This is intentionally conservative:
- If parsing fails, we refuse to apply.
- If a hunk doesn't match the current file, we refuse to apply.
- We create backups in memory (returned to caller) so you can rollback later.

Supported:
- Update existing text files
- Create new files (diff where original is /dev/null)
- Standard unified diff format with --- / +++ and @@ hunks

Note: This is a "good enough v0" applier for typical LLM diffs.
For complex patches, you can later switch to a robust patch library.
"""

from __future__ import annotations

import re
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
            patches.append(FilePatch(
                path=current_path,
                old_path=current_old,
                new_path=current_new,
                hunks=hunks,
                is_new_file=is_new_file
            ))
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
            # detect new file
            if current_old == "/dev/null":
                is_new_file = True
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("Invalid diff: expected '+++' after '---'")
            current_new = lines[i][4:].strip()

            # Determine the target path:
            # Common formats:
            # --- a/path
            # +++ b/path
            # or --- path
            # We'll prefer new path if present.
            candidate = current_new
            # Strip a/ or b/ prefixes if present
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            if candidate.startswith("a/"):
                candidate = candidate[2:]
            if candidate == "/dev/null":
                # If deleting a file, not supported in v0
                raise ValueError("File deletion patches not supported in v0.")
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
            # Hunk body lines start with ' ', '+', '-'
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


def apply_patches(repo_root: str, diff_text: str) -> Dict[str, str]:
    """
    Apply the diff to files under repo_root.

    Returns a dict of {path: old_content} for rollback.

    Raises ValueError on mismatch or parse errors.
    """
    root = Path(repo_root).resolve()
    file_patches = parse_unified_diff(diff_text)

    backups: Dict[str, str] = {}

    for fp in file_patches:
        target = (root / fp.path).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Refusing to write outside repo root: {fp.path}")

        if fp.is_new_file:
            if target.exists():
                raise ValueError(f"Patch wants to create new file but it already exists: {fp.path}")
            new_content = _apply_to_lines([], fp.hunks)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(new_content) + ("\n" if new_content and not new_content[-1].endswith("\n") else ""), encoding="utf-8")
            backups[fp.path] = ""  # new file backup is empty
            continue

        # Existing file update
        if not target.exists():
            raise ValueError(f"Patch targets missing file: {fp.path}")

        old_text = target.read_text(encoding="utf-8")
        backups[fp.path] = old_text

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

        # We'll walk through hunk lines and build the replacement chunk
        # while verifying context.
        new_chunk: List[str] = []
        consume_idx = idx

        for hl in hunk_lines:
            if not hl:
                # empty line can be context; unified diff represents it as " " + ""
                # but some generators might produce empty strings; treat as context mismatch
                raise ValueError("Malformed hunk line (empty).")

            tag = hl[0]
            text = hl[1:]  # rest of the line without prefix

            if tag == " ":
                # context: must match existing line
                if consume_idx >= len(lines) or lines[consume_idx] != text:
                    raise ValueError(f"Hunk context mismatch at line {consume_idx + 1}. Expected '{text}', found '{lines[consume_idx] if consume_idx < len(lines) else 'EOF'}'")
                new_chunk.append(text)
                consume_idx += 1

            elif tag == "-":
                # removal: must match existing line
                if consume_idx >= len(lines) or lines[consume_idx] != text:
                    raise ValueError(f"Hunk removal mismatch at line {consume_idx + 1}. Expected '{text}', found '{lines[consume_idx] if consume_idx < len(lines) else 'EOF'}'")
                # removed line is NOT added to new_chunk
                consume_idx += 1

            elif tag == "+":
                # addition: add new line
                new_chunk.append(text)

            else:
                raise ValueError(f"Unknown hunk tag '{tag}' in line: {hl}")

        # Replace the consumed range with new_chunk
        before = lines[:idx]
        after = lines[consume_idx:]
        lines = before + new_chunk + after

        # Update offset: new length - old length (approx)
        # consume_idx - idx is old consumed size
        consumed_old = consume_idx - idx
        offset += (len(new_chunk) - consumed_old)

    return lines
