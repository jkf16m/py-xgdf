"""Near-deterministic workflow programs.

Public API:
    Session                 # a writable context window
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

        cfg = AgentConfig()                 # owns a session (context window)
        session = cfg.get_session()         # writable reference
        session.add("system", "answer in the user's language")
        if gdev.prompt("describe the change", session=session):
            gdev.agent(gdev.request, config=cfg)

Rules:
  * ``gdev.shell()`` is a trusted author step, never available to the model;
    the agent stays limited to the constrained file tools.
  * ``AgentConfig.model`` defaults to the configured gdev model.
  * ``AgentConfig()`` without an explicit session lazily creates one; keep
    the config around and every ``agent(config=cfg)`` call shares that
    context window. Without a session, each call runs on a temporary
    context (the model works from its own memory alone).
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
from gdev.tools import ToolSpec
from gdev.workspace import context as workspace_context


@dataclass(kw_only=True, slots=True)
class Session:
    """A writable context window shared between prompt() and agent() calls.

    Use ``add(role, content)`` or append dicts to ``messages`` directly;
    agent() replays the whole history on every call.
    """

    name: str = "session"
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """Append one message to the window."""
        self.messages.append({"role": role, "content": content})

    def extend(self, more: list[dict]) -> None:
        self.messages.extend(more)


@dataclass(kw_only=True, slots=True)
class AgentConfig:
    """Optional inference settings for one or more agent() steps.

    ``model`` defaults to the configured gdev model. ``session`` is a
    :class:`Session`; when omitted, one is created lazily by
    ``get_session()``. ``tools`` optionally narrows the agent's tool schema
    to these names only — plain strings are normalized to bare ToolSpecs
    internally (no description, no parameters); the internal state machine
    still applies. Passing
    the same config (or session) to several agent() calls makes them share
    one context window; use ``branch()`` or ``AgentConfig(**{**vars(cfg),
    "tools": [...]})`` (object destructuring) for per-step variations.
    """

    model: str | None = None
    session: Session | None = None
    tools: list[str | ToolSpec] | None = None

    _KNOWN_TOOLS = frozenset({"select", "edit", "close", "new", "delete"})

    def __post_init__(self) -> None:
        """Normalize plain strings to bare ToolSpecs and fail fast on
        unknown tool names at construction, not at run time."""
        if self.tools is not None:
            normalized: list[ToolSpec] = []
            for entry in self.tools:
                if isinstance(entry, str):
                    entry = ToolSpec(name=entry)
                normalized.append(entry)
            unknown = set(spec.name for spec in normalized) - self._KNOWN_TOOLS
            if unknown:
                raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
            self.tools = normalized

    def get_session(self) -> Session:
        """Return the config's session, creating it on first use."""
        if self.session is None:
            self.session = Session()
        return self.session

    def branch(self, model: str | None = None, session: Session | None = None,
               tools: list[str | ToolSpec] | None = None, **more) -> "AgentConfig":
        """Derive a config with only the given fields replaced.

        Fields left as None are inherited from this config; the session
        object is shared, so branched steps still write to the same window.
        ``more`` rejects typos instead of silently ignoring them.
        """
        if more:
            raise TypeError(f"unknown AgentConfig fields: {sorted(more)}")
        return AgentConfig(
            model=model if model is not None else self.model,
            session=session if session is not None else self.session,
            tools=tools if tools is not None else self.tools,
        )


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

    def shell(self, command: str) -> int:
        """Run one trusted shell step as the workflow author, not the agent."""
        print(f"\n[workflow] $ {command}")
        return subprocess.run(command, shell=True, cwd=self.root).returncode

    def agent(self, prompt: str, config: AgentConfig | None = None) -> str:
        """Run one constrained agent turn with optional model/session."""
        config = config or AgentConfig()
        model = config.model or default_model(self.root)
        session = config.session
        print(f"\n[workflow] agent step (model={model}, session={session.name if session else 'temporary'})")
        # Late import avoids the loader cycle: agent.run imports tools, and
        # workflows import neither at module load time.
        from gdev.agent import ToolRejected, run as agent_run

        # A provided session replays its history as a persistent context
        # window; a temporary session starts empty every time. If prompt()
        # already appended this exact request to the session, pop it:
        # agent_run() adds it back as this turn's user message, keeping the
        # history correct.
        history = [] if session is None else list(session.messages)
        if session is not None and history and history[-1] == {"role": "user", "content": prompt}:
            history.pop()
        seen: list[dict] = []

        def llm(messages, tools=None):
            seen.clear()
            seen.extend(messages)
            return chat(messages, tools, model=model)

        try:
            result = agent_run(str(self.root), prompt, llm, history=history, tools=config.tools)
        except ToolRejected as exc:
            print(f"\n[workflow] tool call rejected: {exc}")
            result = f"tool call rejected: {exc}"
        # The loop mutates one list; the last observed snapshot is the whole
        # conversation including tool exchanges.
        if seen and session is not None:
            session.messages = list(seen)
        return result

    def prompt(self, invitation: str = "describe what you want", session: Session | None = None) -> bool:
        """Yield control to the user; returns False on empty input/EOF.

        With ``session``, the user request is appended to that session's
        context window immediately, so the next agent() call with the same
        session sees it.
        """
        try:
            text = input(f"\n{invitation}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not text:
            return False
        self.request = text
        if session is not None:
            session.add("user", text)
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
