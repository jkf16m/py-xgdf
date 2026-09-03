"""Workflow-authoring utilities.

Public API:
    read_workspace(session, root) -> str

These are explicit steps a workflow calls when it wants their effect —
not hidden behavior inside cfg.
"""

from __future__ import annotations

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
