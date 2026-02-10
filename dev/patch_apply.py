from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List


_INDEX_OK = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(?: \d{6})?$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_HUNK_PARSE_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
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


def _strip_markdown_fences(text: str) -> str:
    """Remove common accidental wrappers like ```diff fences or quote blocks."""
    t = (text or "").strip()

    # Common copy/paste quoting
    if t.startswith("> "):
        t = t[2:].lstrip()

    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_patch_text(text: str) -> str:
    """
    Normalize raw LLM output into a candidate patch string.

    Behavior:
    - If a Markdown/code fence (lines starting with ``` or ```diff) is present,
      extract the content from the first matching pair of fences.
    - After removing fences, locate the first literal occurrence of "diff --git"
      and return the substring starting at that position through the end.
    - If no "diff --git" exists, return the trimmed content.

    Keep this helper deliberately small so higher-level heuristics can still run.
    """
    if not text:
        return ""
    lines = text.splitlines()
    # Find first opening fence anywhere (line that starts with ```).
    start_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("```")), None)
    content = text
    if start_idx is not None:
        # Find closing fence after the opening.
        end_idx = next((i for i in range(start_idx + 1, len(lines)) if lines[i].strip().startswith("```")), None)
        if end_idx is not None:
            # Extract inner content between fences.
            inner_lines = lines[start_idx + 1 : end_idx]
            content = "\n".join(inner_lines)
        else:
            # No closing fence; take everything after the opening fence line.
            inner_lines = lines[start_idx + 1 :]
            content = "\n".join(inner_lines)
    # Normalize line endings and trim
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # If a diff header appears, return starting at the first 'diff --git'
    idx = content.find("diff --git")
    if idx != -1:
        return content[idx:].strip()
    # Otherwise return trimmed content (so existing validators get to decide)
    return content.strip()


def _normalize_unified_diff(raw_text: str) -> str:
    """Extract the diff portion from noisy model output.

    We accept that LLMs sometimes include code fences or a short prelude.
    The returned diff always ends with a newline.
    """
    t = _strip_markdown_fences(raw_text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    lines = (t or "").splitlines()
    first = next((i for i, l in enumerate(lines) if l.startswith("diff --git ")), None)
    if first is None:
        return (t or "").strip()

    # Drop leading junk.
    lines = lines[first:]

    # Drop trailing junk (keep only lines that plausibly belong to a diff).
    def _looks_like_diff_line(l: str) -> bool:
        if not l:
            return True
        if l.startswith((
            "diff --git ",
            "--- ",
            "+++ ",
            "@@ ",
            " ",
            "+",
            "-",
            "Binary files ",
            "GIT binary patch",
            "\\ No newline at end of file",
        )):
            return True
        if any(l.startswith(p) for p in _META_PREFIXES):
            return True
        return False

    last = None
    for i in range(len(lines) - 1, -1, -1):
        if _looks_like_diff_line(lines[i].rstrip("\n")):
            last = i
            break
    if last is not None:
        lines = lines[: last + 1]

    out = "\n".join(lines).strip("\n") + "\n"
    return out


def _validate_unified_diff(diff_text: str) -> None:
    # Normalize candidate text first (strip fences and find first diff --git)
    t = _normalize_patch_text(diff_text).strip()
    if not t:
        raise RuntimeError("Refusing to apply patch: empty diff.")

    lines = t.splitlines()
    first_diff_idx = next((i for i, l in enumerate(lines) if l.startswith("diff --git ")), None)
    if first_diff_idx is None:
        raise RuntimeError("Refusing to apply patch: missing 'diff --git' header.")

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
            in_hunk = False
            hunk_old_remaining = 0
            hunk_new_remaining = 0

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
                    m_hunk = _HUNK_PARSE_RE.match(l)
                    if not m_hunk:
                        raise RuntimeError(f"Refusing to apply patch: malformed hunk header '{l}'.")
                    old_count = int(m_hunk.group(2) or "1")
                    new_count = int(m_hunk.group(4) or "1")
                    hunk_old_remaining = old_count
                    hunk_new_remaining = new_count
                    in_hunk = True
                    saw_hunk = True
                elif in_hunk:
                    if l.startswith("\\ No newline at end of file"):
                        pass
                    elif l.startswith(" "):
                        hunk_old_remaining -= 1
                        hunk_new_remaining -= 1
                    elif l.startswith("-"):
                        hunk_old_remaining -= 1
                    elif l.startswith("+"):
                        hunk_new_remaining -= 1
                    else:
                        raise RuntimeError(f"Refusing to apply patch: unexpected line inside hunk '{l}'.")
                    if hunk_old_remaining < 0 or hunk_new_remaining < 0:
                        raise RuntimeError("Refusing to apply patch: hunk line counts do not match header.")
                    if hunk_old_remaining == 0 and hunk_new_remaining == 0:
                        in_hunk = False
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
            if in_hunk:
                raise RuntimeError("Refusing to apply patch: hunk line counts do not match header.")
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
    # Normalize candidate text first (strip fences and find first diff --git)
    diff_text = _normalize_patch_text(diff_text)
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
