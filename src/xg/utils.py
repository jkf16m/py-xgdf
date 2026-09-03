"""Workflow-authoring utilities.

Public API:
    read_workspace(session, root) -> str
    tool_patch(tool_state, name, arguments) -> str | None
    format_patch(patch, formatter=None) -> str

These are explicit steps a workflow calls when it wants their effect —
not hidden behavior inside cfg.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from xg.workspace import context as _workspace_context

# Dedupe marker: a window that already carries a read never gets another.
WORKSPACE_MARKER = "# xgdf workspace read"


def read_workspace(session, root: str | Path) -> str:
    """Append the deterministic whole-workspace read to a session window.

    Reads every readable text file below ``root`` (m-time ordered, hidden
    metadata and gitignored paths excluded) and appends the rendered
    context to ``session`` as one system message — exactly once per window.
    The intended use is the default workflow's start step; workflows that
    don't need file context simply never call this.

    Returns the rendered context string.
    """
    context = _workspace_context(root)
    if not any(WORKSPACE_MARKER in m.get("content", "") for m in session):
        session.add("system", f"{WORKSPACE_MARKER}\n{context}")
    return context


# ---- git patches for unapplied tool calls -------------------------------


def _git_patch(old: Path | None, new: Path, rel: str) -> str | None:
    """Render a git patch between two file states without applying it.

    ``git diff --no-index`` works outside any repository and emits real git
    patch text (diff --git headers, ---/+++ lines). ``old`` may be None for
    file creation (diffed against /dev/null, producing new-file headers).
    Temp paths are rewritten to a/<rel> and b/<rel> in the output.
    """
    command = ["git", "diff", "--no-index", "--binary"]
    try:
        if old is None:
            command += ["/dev/null", str(new)]
        else:
            command += [str(old), str(new)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    patch = result.stdout
    if not patch.strip():  # exit code 1 means "differ", which is expected
        return None
    if old is not None:
        # output is "a" + "/abs/path" (git adds the a/ b/ prefixes itself)
        patch = patch.replace(str(old), f"/{rel}")
    return patch.replace(str(new), f"/{rel}")


def _fallback_patch(old_text: str, new_text: str, a_name: str, b_name: str) -> str:
    """difflib fallback when git is unavailable — same unified format."""
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        fromfile=f"a/{a_name}", tofile=f"b/{b_name}",
    )
    return "".join(diff)


def tool_patch(tool_state, name: str, arguments: str | dict) -> str | None:
    """Render a git patch for an edit/new/delete tool call before it runs.

    The patch previews what the tool WILL do — nothing is applied. Returns
    None when the call produces no patch (unknown tool, bad arguments, or
    edit() with a non-unique old_text).
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return None
    root: Path = tool_state.root

    old_text = new_text = None
    rel = None
    if name == "edit":
        if tool_state.selected is None:
            return None
        try:
            current = tool_state.selected.read_text(encoding="utf-8")
            if current.count(arguments.get("old_text", "")) != 1:
                return None
        except OSError:
            return None
        rel = tool_state.selected.relative_to(root).as_posix()
        old_text, new_text = current, current.replace(arguments.get("old_text", ""), arguments.get("new_text", ""), 1)
    elif name == "new":
        rel = arguments.get("name", "")
        new_text = arguments.get("content", "")
        old_text = None  # /dev/null side: real new-file headers
    elif name == "delete":
        rel = arguments.get("path", "")
        try:
            old_text = (root / rel).read_text(encoding="utf-8")
        except OSError:
            return None
        new_text = ""
        # deleted files are best shown as pure removals
    else:
        return None

    with tempfile.TemporaryDirectory(prefix="xg-patch-") as directory:
        temp = Path(directory)
        if old_text is not None:
            old_path = temp / "a" / rel
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text(old_text, encoding="utf-8")
        else:
            old_path = None
        new_path = temp / "b" / rel
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(new_text or "", encoding="utf-8")
        patch = _git_patch(old_path, new_path, rel)
    return patch if patch is not None else _fallback_patch(old_text or "", new_text or "", rel, rel)


def _default_formatter() -> str | None:
    """External diff formatter from $XG_DIFF_FORMATTER or config.json."""
    from xg.config import load

    value = os.environ.get("XG_DIFF_FORMATTER")
    if value:
        return value
    return load(".").get("diff_formatter")


def format_patch(patch: str, formatter: str | None = None) -> str:
    """Color a git patch: via an external formatter subprocess if configured.

    ``formatter`` (or $XG_DIFF_FORMATTER, or the ``diff_formatter`` config
    key) is a shell command receiving the patch on stdin — e.g. ``delta``
    or ``diffr``. If the formatter fails, the plain patch is returned.
    """
    if not patch:
        return patch
    formatter = formatter or _default_formatter()
    if formatter:
        try:
            result = subprocess.run(
                formatter, shell=True, input=patch,
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except OSError:
            pass
    # Built-in ANSI coloring fallback.
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(f"\033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f"\033[31m{line}\033[0m")
        elif line.startswith("@@"):
            lines.append(f"\033[36m{line}\033[0m")
        else:
            lines.append(line)
    return "\n".join(lines)
