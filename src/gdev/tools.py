"""Stateful deterministic editing tools.

Public API:
    ToolState
    schemas() -> list[dict]
    ToolState.select(path) -> str
    ToolState.edit(old_text, new_text) -> str
    ToolState.new(name) -> str
    ToolState.content(content) -> str
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from gdev.workspace import resolve


class ToolState:
    """Tool state for one agent turn; selection is required before editing."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.selected: Path | None = None
        self.pending_new: Path | None = None
        # A select/new call starts a mandatory two-step workflow. Until the
        # second step completes, no other tool call is valid.
        self.expected: str | None = None
        self.workflow_started = False
        self.workflow_complete = False

    def allowed_tools(self) -> set[str]:
        """Return only tools valid for the current workflow state."""
        if self.workflow_complete:
            return set()
        if self.expected:
            return {self.expected}
        return {"select", "new"}

    def select(self, path: str) -> str:
        """Select an existing workspace file for the next edit."""
        if self.workflow_complete or self.workflow_started:
            raise ValueError("this turn already has a completed or active workflow")
        self.selected = resolve(self.root, path)
        self.pending_new = None
        self.expected = "edit"
        self.workflow_started = True
        return f"selected {self.selected.relative_to(self.root).as_posix()}; next call must be edit()"

    def preview_edit(self, old_text: str, new_text: str) -> str:
        """Return a colored unified diff without changing the selected file."""
        if self.selected is None:
            raise ValueError("select a file before edit")
        current = self.selected.read_text(encoding="utf-8")
        if current.count(old_text) != 1:
            raise ValueError("old_text must occur exactly once in selected file")
        updated = current.replace(old_text, new_text, 1)
        diff = difflib.unified_diff(
            current.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile=f"a/{self.selected.relative_to(self.root).as_posix()}",
            tofile=f"b/{self.selected.relative_to(self.root).as_posix()}",
        )
        lines = []
        for line in diff:
            color = "\\033[31m" if line.startswith("-") and not line.startswith("---") else "\\033[32m" if line.startswith("+") and not line.startswith("+++") else "\\033[90m"
            lines.append(f"{color}{line.rstrip()}\\033[0m")
        return "\\n".join(lines) or "(no changes)"

    def edit(self, old_text: str, new_text: str) -> str:
        """Replace exactly one occurrence in the previously selected file."""
        if self.expected != "edit":
            raise ValueError("edit() is not the required next workflow step")
        if self.selected is None:
            raise ValueError("select a file before edit")
        current = self.selected.read_text(encoding="utf-8")
        if current.count(old_text) != 1:
            raise ValueError("old_text must occur exactly once in selected file")
        self.selected.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
        self.expected = None
        self.workflow_complete = True
        return f"edited {self.selected.relative_to(self.root).as_posix()}"

    def new(self, name: str) -> str:
        """Choose a new relative file; content() must be called next."""
        if self.workflow_complete or self.workflow_started:
            raise ValueError("this turn already has a completed or active workflow")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path.exists():
            raise ValueError("new file must be a non-existing path inside the workspace")
        self.pending_new = path
        self.selected = None
        self.expected = "content"
        self.workflow_started = True
        return f"ready for content for {path.relative_to(self.root).as_posix()}; next call must be content()"

    def content(self, content: str) -> str:
        """Write content to the file selected by new()."""
        if self.pending_new is None:
            raise ValueError("call new(name) before content(content)")
        self.pending_new.parent.mkdir(parents=True, exist_ok=True)
        self.pending_new.write_text(content, encoding="utf-8")
        name = self.pending_new.relative_to(self.root).as_posix()
        self.pending_new = None
        self.expected = None
        self.workflow_complete = True
        return f"created {name}"


def schemas(allowed: set[str] | None = None) -> list[dict]:
    """Return schemas, narrowed to the currently valid workflow step."""
    all_schemas = [
        {"type": "function", "function": {"name": "select", "description": "Select an existing file before editing it.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit", "description": "Edit the file selected by select().", "parameters": {"type": "object", "properties": {"old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["old_text", "new_text"]}}},
        {"type": "function", "function": {"name": "new", "description": "Choose a new file, then call content().", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
        {"type": "function", "function": {"name": "content", "description": "Write content to the file chosen by new().", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}},
    ]
    if allowed is None:
        return all_schemas
    return [schema for schema in all_schemas if schema["function"]["name"] in allowed]


def dispatch(state: ToolState, name: str, arguments: str) -> str:
    """Dispatch one JSON tool call to the stateful protocol."""
    if name not in {"select", "edit", "new", "content"}:
        raise ValueError(f"unknown tool: {name}")
    if state.expected is not None and name != state.expected:
        raise ValueError(f"the next tool call must be {state.expected}()")
    args = json.loads(arguments or "{}")
    return str(getattr(state, name)(**args))
