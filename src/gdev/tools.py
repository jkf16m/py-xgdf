"""Stateful deterministic editing tools.

Public API:
    ToolState
    schemas() -> list[dict]
    ToolState.select(path) -> str
    ToolState.edit(old_text, new_text) -> str
    ToolState.close() -> str
    ToolState.new(name, content) -> str
    ToolState.delete(path) -> str

The default flow is deliberately constrained. There is no shell access.
Allowed transitions:
    select(path) -> edit(old_text, new_text) | close()
    new(name, content)            (single operation)
    delete(path)                  (single operation, asks for confirmation)
"""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import prompt as line_prompt

from gdev.workspace import resolve

BASE_TOOLS = {"select", "new", "delete"}
SELECTED_TOOLS = {"edit", "close"}


@dataclass(slots=True)
class ToolSpec:
    """A tool reference with optional metadata for the model.

    Only ``name`` is required (must match a built-in tool). A ``description``
    overrides the built-in schema description and ``parameters`` overrides
    the JSON schema — richer information makes the model more reliable, but
    execution still follows the built-in state machine.
    """

    name: str
    description: str | None = None
    parameters: dict | None = None

    @property
    def title(self) -> str:
        return self.name


def _confirm_terminal(message: str) -> bool:
    """Terminal confirmation prompt used for destructive operations."""
    if not sys.stdin.isatty():
        return False
    try:
        return line_prompt(message).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


class ToolState:
    """Tool state for one agent turn; selection is required before editing."""

    def __init__(self, root: str | Path, confirm=_confirm_terminal):
        self.root = Path(root).resolve()
        self.selected: Path | None = None
        self.confirm = confirm
        self.tool_set: set[str] | None = None  # optional restriction from workflow config

    def restrict(self, tools: list[str]) -> None:
        """Narrow valid tools to these names (workflow-provided subset)."""
        unknown = set(tools) - (BASE_TOOLS | SELECTED_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
        self.tool_set = set(tools)

    def allowed_tools(self) -> set[str]:
        """Return only tools valid for the current workflow state."""
        allowed = SELECTED_TOOLS if self.selected is not None else BASE_TOOLS
        if self.tool_set is not None:
            allowed = allowed & self.tool_set
            if self.selected is not None:
                # close() is always available once a file is selected; it is
                # the safety valve that un-stucks a restricted agent.
                allowed.add("close")
        return allowed

    def select(self, path: str) -> str:
        """Select an existing workspace file; only edit() or close() is valid next."""
        self.selected = resolve(self.root, path)
        return f"selected {self.selected.relative_to(self.root).as_posix()}; next call must be edit() or close()"

    def close(self) -> str:
        """Close the selected file and return to the previous step."""
        if self.selected is None:
            raise ValueError("no file is selected")
        name = self.selected.relative_to(self.root).as_posix()
        self.selected = None
        return f"closed {name}; back to {', '.join(sorted(BASE_TOOLS))}"

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
            color = "\033[31m" if line.startswith("-") and not line.startswith("---") else "\033[32m" if line.startswith("+") and not line.startswith("+++") else "\033[90m"
            lines.append(f"{color}{line.rstrip()}\033[0m")
        return "\n".join(lines) or "(no changes)"

    def edit(self, old_text: str, new_text: str) -> str:
        """Replace exactly one occurrence in the previously selected file."""
        if self.selected is None:
            raise ValueError("select a file before edit")
        current = self.selected.read_text(encoding="utf-8")
        if current.count(old_text) != 1:
            raise ValueError("old_text must occur exactly once in selected file")
        self.selected.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
        name = self.selected.relative_to(self.root).as_posix()
        self.selected = None
        return f"edited {name}"

    def new(self, name: str, content: str) -> str:
        """Create a new file with content; a single, non-composable operation."""
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path.exists():
            raise ValueError("new file must be a non-existing path inside the workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"created {path.relative_to(self.root).as_posix()}"

    def delete(self, path: str) -> str:
        """Delete an existing file after explicit user confirmation."""
        target = resolve(self.root, path)
        name = target.relative_to(self.root).as_posix()
        if not self.confirm(f"Delete {name}? [y/N] "):
            self.selected = None
            return f"deletion of {name} rejected"
        target.unlink()
        self.selected = None
        return f"deleted {name}"


def schemas(allowed: set[str] | None = None, overrides: dict[str, ToolSpec] | None = None) -> list[dict]:
    """Return schemas, narrowed to the currently valid workflow step.

    ``overrides`` applies per-tool description/parameter customizations from
    ToolSpec references onto the built-in schemas.
    """
    all_schemas = [
        {"type": "function", "function": {"name": "select", "description": "Select an existing file; then edit() it or close() to go back.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit", "description": "Edit the file selected by select().", "parameters": {"type": "object", "properties": {"old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["old_text", "new_text"]}}},
        {"type": "function", "function": {"name": "close", "description": "Close the selected file and return to the previous step.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "new", "description": "Create a new file with content in a single operation.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "content": {"type": "string"}}, "required": ["name", "content"]}}},
        {"type": "function", "function": {"name": "delete", "description": "Delete an existing file; the user is asked for confirmation.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    ]
    result = [schema for schema in all_schemas if schema["function"]["name"] in allowed]
    for schema in result:
        spec = (overrides or {}).get(schema["function"]["name"])
        if spec is not None:
            if spec.description:
                schema["function"]["description"] = spec.description
            if spec.parameters:
                schema["function"]["parameters"] = spec.parameters
    return result


def dispatch(state: ToolState, name: str, arguments: str) -> str:
    """Dispatch one JSON tool call to the stateful protocol."""
    if name not in BASE_TOOLS | SELECTED_TOOLS:
        raise ValueError(f"unknown tool: {name}")
    allowed = state.allowed_tools()
    if name not in allowed:
        raise ValueError(f"tool {name}() is not valid now; allowed: {', '.join(sorted(allowed))}")
    args = json.loads(arguments or "{}")
    return str(getattr(state, name)(**args))
