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
import json
import re
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
    """An ephemeral, disk-backed context window.

    Messages live in an append-only JSONL file under ``.gdev/sessions/``.
    Inference reads straight from that file: chat() walks it line-by-line to
    assemble the HTTP request body, so the window is never loaded into RAM.
    Writes append one JSON line at a time. Sessions are cheap to create —
    there is no lifecycle to manage.
    """

    name: str = "session"
    path: Path | None = None
    _buffer: list[dict] = field(default_factory=list)  # pre-bind writes

    def bind(self, root: str | Path) -> None:
        """Attach the session to a workspace. Idempotent.

        Existing files resume, so named sessions keep their history across
        agent() calls and runs.
        """
        if self.path is not None:
            return
        directory = Path(root).resolve() / ".gdev" / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-") or "session"
        self.path = directory / f"{slug}.jsonl"
        for message in self._buffer:
            self._write(message)
        self._buffer.clear()

    def _write(self, message: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    def add(self, role: str, content: str) -> None:
        """Append one message to the window."""
        self.append({"role": role, "content": content})

    def append(self, message: dict) -> None:
        """Append a raw message dict (roles like tool need extra keys)."""
        if self.path is not None:
            self._write(message)
        else:
            self._buffer.append(message)

    def extend(self, more) -> None:
        for message in more:
            self.append(message)

    def __iter__(self):
        """Walk the window oldest-first; one message in RAM at a time."""
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        yield from self._buffer

    def first(self) -> dict | None:
        for message in self:
            return message
        return None

    def last(self) -> dict | None:
        result = None
        for message in self:
            result = message
        return result

    def snapshot(self) -> list[dict]:
        """Materialize the window (debugging only; inference never does)."""
        return list(self)


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
        """Run one constrained agent turn on a session-backed context window.

        Without ``config.session`` an ephemeral session is created (unique
        name, removed afterwards); with one, the window resumes from disk.
        """
        import uuid

        config = config or AgentConfig()
        model = config.model or default_model(self.root)
        session = config.session
        ephemeral = session is None
        if ephemeral:
            session = Session(name=f"temp-{uuid.uuid4().hex[:8]}")
        session.bind(self.root)
        prompt_appended = (
            not ephemeral
            and session.last() == {"role": "user", "content": prompt}
        )
        print(f"\n[workflow] agent step (model={model}, session={session.name}{', ephemeral' if ephemeral else ''})")
        # Late import avoids the loader cycle: agent.run imports tools, and
        # workflows import neither at module load time.
        from gdev.agent import ToolRejected, run as agent_run

        try:
            return agent_run(
                str(self.root), prompt,
                lambda messages, tools=None: chat(messages, tools, model=model),
                tools=config.tools, session=session, prompt_appended=prompt_appended,
            )
        except ToolRejected as exc:
            print(f"\n[workflow] tool call rejected: {exc}")
            return f"tool call rejected: {exc}"
        finally:
            if ephemeral and session.path is not None and session.path.exists():
                session.path.unlink()

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
