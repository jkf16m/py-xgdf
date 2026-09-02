"""Small SDK for defining deterministic xg tools.

Tool modules under ``.xg/tools`` can use this module without importing the CLI.
A tool is just a name, an OpenAI-compatible schema, and a Python callable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from xg.workspace import context as workspace_context


@dataclass
class ToolContext:
    """Execution context supplied to every registered tool."""

    root: Path
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Explicit result returned to the model and session recorder."""

    content: str
    ok: bool = True

    def __str__(self) -> str:
        return self.content


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    """Ordered registry of tools exposed to one agent turn."""

    def __init__(self, context: ToolContext):
        self.context = context
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def schemas(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if names is None else (
            self._tools[name] for name in self._tools if name in names
        )
        return [tool.schema() for tool in tools]

    def call(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            raise ValueError(f"unknown tool: {name}")
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = self._tools[name].function(self.context, **args)
        return result if isinstance(result, ToolResult) else ToolResult(str(result))

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(name: str, description: str, parameters: dict[str, Any]):
    """Decorator that turns a function into a :class:`Tool` definition."""
    def decorate(function: Callable[..., Any]) -> Tool:
        return Tool(name, description, parameters, function)
    return decorate


class Context:
    """Explicit context builder used by a profile program."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.parts: list[str] = []

    def read_workspace(self) -> str:
        """Add the deterministic workspace representation to this context."""
        value = workspace_context(self.root)
        self.parts.append(value)
        return value

    def add(self, value: str) -> None:
        self.parts.append(str(value))

    def render(self) -> str:
        return "\n\n".join(self.parts)


@dataclass
class CallResult:
    """Result of one concrete execution point in a profile."""

    name: str
    arguments: dict[str, Any]
    result: str


class ToolRef:
    """A per-execution-point tool object filled by ``agent.call``."""

    def __init__(self, name: str):
        self.name = name
        self.used = False
        self.arguments: dict[str, Any] = {}
        self.result: str | None = None

    def reset(self) -> "ToolRef":
        self.used = False
        self.arguments = {}
        self.result = None
        return self


# Convenience factories. A profile should create fresh instances per turn.
def select() -> ToolRef:
    return ToolRef("select")


def edit() -> ToolRef:
    return ToolRef("edit")


def new() -> ToolRef:
    return ToolRef("new")


def close() -> ToolRef:
    return ToolRef("close")


def delete() -> ToolRef:
    return ToolRef("delete")


class Agent:
    """Imperative profile runtime; each ``call`` is one model turn."""

    def __init__(self, root: str | Path, prompt: str, chat: Callable, system_prompt: str):
        self.root = Path(root).resolve()
        self.prompt = prompt
        self.chat = chat
        self.system_prompt = system_prompt
        self.context = Context(self.root)
        self.messages: list[dict[str, Any]] = []

    def tool(self, name: str) -> ToolRef:
        """Create a fresh tool execution-point object."""
        return ToolRef(name)

    def call(self, *tool_names: ToolRef) -> None:
        """Expose these tool objects and fill the one the model uses."""
        if not tool_names:
            raise ValueError("agent.call() requires at least one tool")
        from xg.tools import ToolState, dispatch, schemas
        from prompt_toolkit import prompt as line_prompt
        state = getattr(self, "_tool_state", None)
        if state is None:
            state = self._tool_state = ToolState(self.root)
        allowed = {tool.name for tool in tool_names}
        user = f"REQUEST:\n{self.prompt}\n\nCONTEXT:\n{self.context.render()}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        message = self.chat(messages, schemas(allowed))
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise RuntimeError("an execution point must produce exactly one tool call")
        call = calls[0]
        function = call.get("function") or {}
        name = function.get("name", "")
        if name not in allowed:
            raise RuntimeError(f"tool not allowed at this execution point: {name}")
        print(f"\nProposed tool call: {name}({function.get('arguments', '{}')})")
        try:
            accepted = line_prompt("Accept this tool call? [y/N] ").strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            accepted = False
        if not accepted:
            raise RuntimeError(f"tool call rejected: {name}")
        arguments = json.loads(function.get("arguments") or "{}")
        result = dispatch(state, name, function.get("arguments") or "{}")
        selected = next(tool for tool in tool_names if tool.name == name)
        selected.used = True
        selected.arguments = arguments
        selected.result = result


def profile(function: Callable[[Agent], Any]) -> Callable[[Agent], Any]:
    """Mark an ordinary Python function as a xg agent profile."""
    function.__xg_profile__ = True
    return function


def fake_tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    """Create an assistant tool call for deterministic SDK tests."""
    return {
        "id": "fake-call",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
