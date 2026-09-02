"""Deterministic workspace reading and file selection.

Public API:
    FileEntry
    files(root) -> list[FileEntry]
    context(root) -> str
    resolve(root, name) -> Path

All files are ordered by modification time, oldest first. Hidden metadata,
VCS directories, and binary files are omitted from the agent context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class FileEntry:
    """One readable workspace file and its deterministic metadata."""

    path: str
    mtime_ns: int
    content: str


def files(root: str | Path) -> list[FileEntry]:
    """Read every readable text file below *root*, oldest modification first."""
    root = Path(root).resolve()
    entries: list[FileEntry] = []
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in {".git", ".xg", "__pycache__", ".venv", "venv"} for part in path.relative_to(root).parts):
            continue
        candidates.append(path)
    ignored = _gitignored(root, [path.relative_to(root).as_posix() for path in candidates])
    for path in candidates:
        if path.relative_to(root).as_posix() in ignored:
            continue
        try:
            data = path.read_bytes()
            if b"\0" in data:
                continue
            text = data.decode("utf-8")
            stat = path.stat()
        except (OSError, UnicodeDecodeError):
            continue
        entries.append(FileEntry(path.relative_to(root).as_posix(), stat.st_mtime_ns, text))
    return sorted(entries, key=lambda item: (item.mtime_ns, item.path))


def _gitignored(root: Path, names: list[str]) -> set[str]:
    """Return paths excluded by the workspace's .gitignore rules."""
    if not names:
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-z", "--stdin"],
            cwd=root,
            input="\0".join(names), text=True, capture_output=True, check=False,
        )
    except OSError:
        return set()
    return {name for name in result.stdout.split("\0") if name}


def context(root: str | Path) -> str:
    """Build the complete deterministic forced-read context for *root*."""
    entries = files(root)
    if not entries:
        return "(workspace contains no readable files)"
    sections = []
    for entry in entries:
        sections.append(f"=== {entry.path} | mtime_ns: {entry.mtime_ns} ===\n{entry.content}")
    return "\n\n".join(sections)


def resolve(root: str | Path, name: str) -> Path:
    """Resolve a selected relative file and reject paths outside *root*."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("file path is outside the workspace")
    if not path.is_file():
        raise FileNotFoundError(name)
    return path
