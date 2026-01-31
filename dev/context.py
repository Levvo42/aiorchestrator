"""
dev/context.py
--------------
Collects local repository context for the Developer AI.

Goal:
- Provide the dev model enough context to propose a correct patch,
  without dumping your entire repo every time.

We keep this simple and safe:
- Collect a directory tree snapshot.
- Include the contents of a small set of relevant files (heuristics).
- Enforce a max character budget so prompts don't explode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def build_tree(root: Path, max_depth: int = 4) -> str:
    """
    Create a readable directory tree string.

    max_depth prevents huge output for larger repos.
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
            if e.name in ("venv", ".venv", ".idea", "__pycache__"):
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


def _read_text(path: Path) -> str:
    """
    Read a file as UTF-8 text safely.
    If unreadable, return a short note rather than crashing.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Could not read file: {e}]"


def choose_relevant_files(root: Path, request: str) -> List[Path]:
    """
    Heuristic: pick files likely relevant to a Dev request.

    Current strategy:
    - Always include main.py if it exists.
    - Always include core/*.py and dev/*.py (small projects benefit from this).
    - If the request mentions a filename/path that exists, include it.
    """
    root = root.resolve()
    files: List[Path] = []

    main_py = root / "main.py"
    if main_py.exists():
        files.append(main_py)

    # Include core + dev python files
    for folder in ("core", "dev"):
        d = root / folder
        if d.exists() and d.is_dir():
            files.extend(sorted(d.rglob("*.py")))

    # If request mentions a specific file name, try to include it
    tokens = [t.strip(" ,.:;()[]{}<>\"'") for t in request.split()]
    token_set = set(t for t in tokens if t)

    # Scan a small set of common files
    common = [
        root / "core" / "capabilities.json",
        root / ".gitignore",
    ]
    for c in common:
        if c.exists():
            files.append(c)

    # Include mentioned files if they exist anywhere (deterministic ordering)
    for candidate in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if candidate.is_file() and candidate.name in token_set:
            files.append(candidate)

    # Deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for f in files:
        fp = str(f.resolve())
        if fp not in seen:
            uniq.append(f)
            seen.add(fp)

    return uniq


def build_context_bundle(
    repo_root: str,
    request: str,
    max_context_chars: int = 80_000,
    max_tree_depth: int = 4
) -> Dict[str, str]:
    """
    Build a context bundle for the dev model.

    Returns a dict with:
    - "tree": directory tree
    - "files": concatenated file contents (with headers)
    """
    root = Path(repo_root).resolve()
    tree = build_tree(root, max_depth=max_tree_depth)

    chosen = choose_relevant_files(root, request=request)

    # Build a single string containing file contents with clear boundaries
    parts: List[str] = []
    used = 0

    for f in chosen:
        rel = f.resolve().relative_to(root)
        header = f"\n\n===== FILE: {rel.as_posix()} =====\n"
        content = _read_text(f)
        block = header + content

        if used + len(block) > max_context_chars:
            # Stop once we hit budget; this prevents huge prompts.
            parts.append("\n\n[Context truncated: max_context_chars reached]")
            break

        parts.append(block)
        used += len(block)

    return {
        "tree": tree,
        "files": "".join(parts).strip()
    }
