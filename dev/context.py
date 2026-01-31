"""
dev/context.py
--------------
Collects local repository context for the Developer AI.

Correctness-first goals:
- Context MUST be reproducible from repo state.
- Context MUST NOT include runtime state, IDE state, venvs, caches, or binaries.
- Context MUST be bounded (size caps) so prompts don't explode.

We enforce a strict "patch-visible surface" allowlist:
  - main.py
  - core/**
  - dev/**
  - providers/**
  - tools/**

We explicitly exclude:
  - venv/.venv
  - .idea
  - __pycache__
  - memory/** (runtime state)
  - .env
  - binary / huge files
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------
# Patch-visible surface controls
# -----------------------------

ALLOWED_DIRS: Tuple[str, ...] = ("core", "dev", "providers", "tools")
ALLOWED_FILES: Tuple[str, ...] = ("main.py", ".gitignore")  # keep .gitignore (stable); exclude .env

DENY_DIR_NAMES: Set[str] = {"venv", ".venv", ".idea", "__pycache__", ".git"}
DENY_FILE_NAMES: Set[str] = {".env", "state.json"}  # state.json is runtime; block by name too (belt & suspenders)
DENY_DIR_PREFIXES: Tuple[str, ...] = ("memory",)  # exclude entire memory tree


# -----------------------------
# Helpers
# -----------------------------

def _is_under_denied_prefix(root: Path, p: Path) -> bool:
    """
    True if p is under a denied top-level directory prefix (e.g. memory/**).
    """
    try:
        rel = p.resolve().relative_to(root.resolve())
    except Exception:
        return True

    parts = rel.parts
    if not parts:
        return False
    top = parts[0]
    return top in DENY_DIR_PREFIXES


def _is_denied_path(root: Path, p: Path) -> bool:
    """
    Deny if any path segment is a deny-dir name, or file is denied, or under denied prefix.
    """
    if _is_under_denied_prefix(root, p):
        return True

    # deny any segment like venv/.idea/__pycache__ anywhere
    for part in p.parts:
        if part in DENY_DIR_NAMES:
            return True

    name = p.name
    if name in DENY_FILE_NAMES:
        return True

    return False


def _is_text_file(path: Path, max_probe_bytes: int = 4096) -> bool:
    """
    Heuristic: treat as text if it can be decoded as UTF-8 (strict) on a small probe.
    """
    try:
        data = path.read_bytes()[:max_probe_bytes]
    except Exception:
        return False
    try:
        data.decode("utf-8")
        return True
    except Exception:
        return False


def _read_text(path: Path, max_chars: int) -> str:
    """
    Read a file as UTF-8 text safely and cap its size.
    """
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Could not read file: {e}]"

    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[Truncated: file exceeded max_chars]"
    return txt


# -----------------------------
# Tree builder
# -----------------------------

def build_tree(root: Path, max_depth: int = 4) -> str:
    """
    Create a readable directory tree string (filtered to allowed surface).
    """
    lines: List[str] = []
    root = root.resolve()

    def walk_dir(p: Path, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            lines.append("  " * depth + "[PermissionError]")
            return

        for e in entries:
            # deny early
            if _is_denied_path(root, e):
                continue

            # only show allowed top-level dirs (and allowed files)
            if depth == 1:
                if e.is_dir() and e.name not in ALLOWED_DIRS:
                    continue
                if e.is_file() and e.name not in ALLOWED_FILES:
                    continue

            prefix = "  " * depth
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                walk_dir(e, depth + 1)
            else:
                lines.append(f"{prefix}{e.name}")

    lines.append(f"{root.name}/")
    walk_dir(root, 1)
    return "\n".join(lines)


# -----------------------------
# File selection
# -----------------------------

def _allowed_roots(root: Path) -> List[Path]:
    """
    Return list of allowed root paths to scan for context files.
    """
    allowed: List[Path] = []
    for d in ALLOWED_DIRS:
        p = root / d
        if p.exists() and p.is_dir():
            allowed.append(p)
    for f in ALLOWED_FILES:
        p = root / f
        if p.exists() and p.is_file():
            allowed.append(p)
    return allowed


def _extract_requested_paths(root: Path, request: str) -> List[Path]:
    """
    Conservative extraction: if the request contains a repo-relative path that exists,
    include it ONLY if it’s inside the allowed surface.
    Examples recognized: "dev/patch_apply.py", "core/agent.py", "main.py"
    """
    candidates: List[Path] = []

    # crude tokenization, but safe
    tokens = [t.strip(" ,.:;()[]{}<>\"'") for t in request.split()]
    for t in tokens:
        if not t or len(t) > 200:
            continue
        # normalize slashes
        t_norm = t.replace("\\", "/")
        # must look like a path we can resolve
        if "/" not in t_norm and t_norm not in ALLOWED_FILES:
            continue

        p = (root / t_norm).resolve()
        if not p.exists() or not p.is_file():
            continue

        # must not be denied
        if _is_denied_path(root, p):
            continue

        # must be within allowed surface
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in ALLOWED_DIRS or rel.name in ALLOWED_FILES:
            candidates.append(p)

    # dedupe preserve order
    seen: Set[str] = set()
    out: List[Path] = []
    for c in candidates:
        k = str(c)
        if k not in seen:
            out.append(c)
            seen.add(k)
    return out


def choose_relevant_files(
    root: Path,
    request: str,
    max_files: int = 40,
) -> List[Path]:
    """
    Select a bounded set of relevant files from the allowed surface.

    Strategy:
    - Always include main.py (if allowed) and dev/core/providers/tools python files.
    - Include a small set of stable config files (e.g. .gitignore).
    - If request mentions an allowed path, include it.
    - Never include memory/state.json or any denied content.
    """
    root = root.resolve()
    chosen: List[Path] = []

    # Requested explicit files (only if within allowed surface)
    chosen.extend(_extract_requested_paths(root, request))

    # Always include main.py if present
    main_py = root / "main.py"
    if main_py.exists() and not _is_denied_path(root, main_py):
        chosen.append(main_py)

    # Include python files in allowed dirs
    for folder in ALLOWED_DIRS:
        d = root / folder
        if not d.exists() or not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if _is_denied_path(root, p):
                continue
            chosen.append(p)

    # Include stable config file(s)
    gitignore = root / ".gitignore"
    if gitignore.exists() and not _is_denied_path(root, gitignore):
        chosen.append(gitignore)

    # Deduplicate while preserving order and cap count
    seen: Set[str] = set()
    uniq: List[Path] = []
    for f in chosen:
        fp = str(f.resolve())
        if fp not in seen:
            uniq.append(f)
            seen.add(fp)
        if len(uniq) >= max_files:
            break

    return uniq


# -----------------------------
# Context bundle
# -----------------------------

def build_context_bundle(
    repo_root: str,
    request: str,
    max_context_chars: int = 80_000,
    max_tree_depth: int = 4,
    max_file_chars: int = 12_000,
) -> Dict[str, str]:
    """
    Build a context bundle for the dev model.

    Returns:
    - "tree": filtered directory tree (allowed surface only)
    - "files": concatenated file contents (allowed + text only), with headers
    """
    root = Path(repo_root).resolve()

    tree = build_tree(root, max_depth=max_tree_depth)
    chosen = choose_relevant_files(root, request=request)

    parts: List[str] = []
    used = 0

    for f in chosen:
        if _is_denied_path(root, f):
            continue
        if not f.exists() or not f.is_file():
            continue

        # only include text files
        if not _is_text_file(f):
            continue

        rel = f.resolve().relative_to(root)
        header = f"\n\n===== FILE: {rel.as_posix()} =====\n"
        content = _read_text(f, max_chars=max_file_chars)
        block = header + content

        if used + len(block) > max_context_chars:
            parts.append("\n\n[Context truncated: max_context_chars reached]")
            break

        parts.append(block)
        used += len(block)

    return {
        "tree": tree,
        "files": "".join(parts).strip(),
    }
