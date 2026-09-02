"""Near-deterministic workflow programs.

Public API:
    AgentConfig(model=None, session=None)
    WorkflowRuntime        # passed to workflows as ``gdev``
    load_workflow(name, cwd) -> callable
    run_workflow(name, cwd) -> int

A workflow is a Python file at ``.gdev/workflows/<name>.py`` exposing
``run(gdev)``. Inside a workflow the author decides deterministically when to
execute shell steps, when to request model inference, and when to yield
control to the user:

    def run(gdev):
        gdev.shell("ruff format .")
        gdev.agent("fix all lint issues", config=AgentConfig(model="org/big"))
        if gdev.prompt("describe the change"):
            gdev.agent(gdev.request, config=AgentConfig(session="main"))

Rules:
  * ``gdev.shell()`` is a trusted author step, never available to the model;
    the agent stays limited to the constrained file tools.
  * ``AgentConfig.model`` defaults to the configured gdev model.
  * ``AgentConfig.session`` names a context window that persists across
    ``agent()`` calls within the workflow; without it each agent call runs on
    a temporary context (the model works from its own memory alone).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gdev.config import gdev_directories, load
from gdev.llm import chat
from gdev.workspace import context as workspace_context


@dataclass
class ContextWindow:
    """Replaying message history for one named session."""

    name: str
    messages: list[dict] = field(default_factory=list)

    def extend(self, more: list[dict]) -> None:
        self.messages.extend(more)


@dataclass
class AgentConfig:
    """Optional inference settings for one agent() step."""

    model: str | None = None
    session: str | None = None


def default_model(cwd: str | Path = ".") -> str:
    """Resolve the gdev default model: config.json, env, then built-in."""
    configured = load(cwd).get("model")
    if configured:
        return str(configured)
    import os

    return os.environ.get("GDEV_MODEL", "@preset/mimo")


class WorkflowRuntime:
    """Primitives handed to a workflow program."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.request: str = ""
        self._windows: dict[str, ContextWindow] = {}

    def shell(self, command: str) -> int:
        """Run one trusted shell step as the workflow author, not the agent."""
        print(f"\n[workflow] $ {command}")
        return subprocess.run(command, shell=True, cwd=self.root).returncode

    def agent(self, prompt: str, config: AgentConfig | None = None) -> str:
        """Run one constrained agent turn with optional model/session."""
        config = config or AgentConfig()
        model = config.model or default_model(self.root)
        print(f"\n[workflow] agent step (model={model}, session={config.session or 'temporary'})")
        window = self._windows.setdefault(config.session or "__temporary__", ContextWindow(config.session or "__temporary__"))
        # Late import avoids the loader cycle: agent.run imports tools, and
        # workflows import neither at module load time.
        from gdev.agent import ToolRejected, run as agent_run

        # A named session replays its history as a persistent context window;
        # a temporary session starts empty every time.
        history = [] if config.session is None else list(window.messages)
        seen: list[dict] = []

        def llm(messages, tools=None):
            seen.clear()
            seen.extend(messages)
            return chat(messages, tools, model=model)

        try:
            result = agent_run(str(self.root), prompt, llm, history=history)
        except ToolRejected as exc:
            print(f"\n[workflow] tool call rejected: {exc}")
            result = f"tool call rejected: {exc}"
        # The loop mutates one list; the last observed snapshot is the whole
        # conversation including tool exchanges.
        if seen:
            window.messages = list(seen)
        return result

    def prompt(self, invitation: str = "describe what you want") -> bool:
        """Yield control to the user; returns False on empty input/EOF."""
        try:
            text = input(f"\n{invitation}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not text:
            return False
        self.request = text
        return True

    def workspace(self) -> str:
        """Render the deterministic whole-workspace context."""
        return workspace_context(self.root)


def _builtin_default(gdev: WorkflowRuntime) -> int:
    """The default workflow: the constrained file-editing flow."""
    gdev.workspace()  # forced deterministic read
    print("tools: select -> edit|close | new(name, content) | delete\n")
    if not gdev.prompt("your request"):
        return 0
    gdev.agent(gdev.request)
    return 0


def load_workflow(name: str, cwd: str | Path = ".") -> Callable[[WorkflowRuntime], int]:
    """Load a workflow program by name; ``default`` is built in."""
    if name == "default":
        return _builtin_default
    selected: Path | None = None
    for directory in gdev_directories(cwd):
        path = directory / "workflows" / f"{name}.py"
        if path.is_file():
            selected = path
    if selected is None:
        raise RuntimeError(f"workflow not found: {name}")
    module_name = f"gdev_workflow_{name}"
    spec = importlib.util.spec_from_file_location(module_name, selected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workflow: {selected}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError(f"workflow must define run(gdev): {selected}")
    return run


def run_workflow(name: str, cwd: str | Path = ".") -> int:
    """Execute one workflow in a fresh runtime."""
    program = load_workflow(name, cwd)
    runtime = WorkflowRuntime(cwd)
    return int(program(runtime) or 0)


def list_workflows(cwd: str | Path = ".") -> list[str]:
    """Return available workflow names, built-ins first."""
    names = ["default"]
    seen = set()
    for directory in gdev_directories(cwd):
        for path in sorted((directory / "workflows").glob("*.py")):
            if path.stem not in seen:
                seen.add(path.stem)
                names.append(path.stem)
    return names
