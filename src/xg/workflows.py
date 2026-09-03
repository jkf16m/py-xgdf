"""Near-deterministic workflow programs.

Public API:
    Session                 # a writable context window
    AgentConfig(model=None, session=None)   # the ``cfg`` handed to workflows
    load_workflow(name, cwd) -> callable
    run_workflow(name, cwd, cfg=None) -> int

A workflow is a Python file at ``.xg/workflows/<name>.py`` exposing
``run(cfg)``. ``cfg`` is an :class:`AgentConfig`: it carries the inference
settings (model, session, tools) AND the runtime primitives. The xg runner
creates it for ``xg -w <name>``; a parent workflow passes its own cfg to a
child workflow, so the child inherits the model, session, and tools:

    def run(cfg):
        from xg.utils import read_workspace

        cfg.shell("ruff format .")
        cfg.agent("fix all lint issues", config=AgentConfig(model="org/big"))

        session = cfg.get_session()             # writable context window
        read_workspace(session, cfg.root)       # deterministic reads, once
        if cfg.prompt("describe the change"):
            cfg.agent(cfg.request)

        cfg.workflow("changelog")               # reuse another workflow

Rules:
  * ``cfg.shell()`` is a trusted author step, never available to the model;
    the agent stays limited to the constrained file tools.
  * ``AgentConfig.model`` defaults to the configured xg model.
  * ``AgentConfig()`` without an explicit session lazily creates one; keep
    the config around and every ``cfg.agent()`` call shares that context
    window. Without a session, each call runs on a temporary context.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from xg.config import xg_directories, load
from xg.docs import documentation_for
from xg.llm import chat
from xg.tools import ToolSpec

if TYPE_CHECKING:
    from xg.profiles import AgentProfile


@dataclass(kw_only=True, slots=True)
class Session:
    """An ephemeral, disk-backed context window.

    Messages live in an append-only JSONL file under ``.xg/sessions/``.
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
        agent() calls and runs. Named sessions also become the workspace's
        ``last session`` pointer (see last_session).
        """
        if self.path is not None:
            return
        directory = Path(root).resolve() / ".xg" / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-") or "session"
        self.path = directory / f"{slug}.jsonl"
        if not self.name.startswith("temp-"):
            (directory / ".last").write_text(slug, encoding="utf-8")
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

    def _raw(self):
        """Walk every stored line — messages AND xgdf step markers."""
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        yield from self._buffer

    def __iter__(self):
        """Walk the window oldest-first; one message in RAM at a time.

        xgdf metadata lines (step completion markers, recorded prompts) are
        resume bookkeeping; inference never sees them.
        """
        for entry in self._raw():
            if not any(key.startswith("xgdf-") for key in entry):
                yield entry

    def events(self):
        """Yield recorded resume events in order.

        Two event kinds:
        * ("prompt", text)         — a recorded prompt() answer
        * ("step", hash, segment)  — a completed agent turn; segment is the
          messages since the previous event boundary (request, context,
          assistant replies, tool results)
        """
        segment: list[dict] = []
        for entry in self._raw():
            if "xgdf-prompt" in entry:
                yield ("prompt", entry["xgdf-prompt"], None)
            elif "xgdf-step" in entry:
                yield ("step", entry["xgdf-step"].get("hash", ""), segment)
                segment = []
            else:
                segment.append(entry)

    def completed_step(self, step_hash: str) -> str | None:
        """Return the final assistant text of a completed step by hash."""
        for kind, value, segment in self.events():
            if kind == "step" and value == step_hash:
                for message in reversed(segment or []):
                    if message.get("role") == "assistant":
                        return message.get("content", "")
        return None

    def first(self) -> dict | None:
        for message in self:
            return message
        return None

    def last(self) -> dict | None:
        result = None
        for message in self:
            result = message
        return result

    def clone(self, name: str | None = None) -> "Session":
        """Return an independent copy of this window — a branch point.

        The clone shares nothing with the original: new appends go to the
        clone's file only. Without ``name`` the clone is named
        ``<original>-clone-<suffix>``. A bound session's file is copied
        (the clone is already bound to the same workspace); an unbound
        one copies its buffered writes.
        """
        import uuid

        if name is None:
            base = re.sub(r"-clone-[0-9a-f]{6}$", "", self.name)
            name = f"{base}-clone-{uuid.uuid4().hex[:6]}"
        clone = Session(name=name)
        if self.path is not None and self.path.exists():
            clone.bind(self.path.parent.parent.parent)  # .../.xg/sessions -> root
            shutil.copyfile(self.path, clone.path)      # plain byte copy
            for message in self._buffer:
                clone.append(message)
        else:
            clone._buffer = list(self._buffer)
        return clone

    def delete(self) -> None:
        """Remove this session's file (a temporary branch's cleanup).

        The session becomes empty and re-binds as a fresh file on next use.
        """
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self.path = None
        self._buffer = []

    def snapshot(self) -> list[dict]:
        """Materialize the window (debugging only; inference never does)."""
        return list(self)


@dataclass(kw_only=True, slots=True)
class AgentConfig:
    """Optional inference settings for one or more agent() steps.

    ``model`` defaults to the configured xg model. ``session`` is a
    :class:`Session`; when omitted, one is created lazily by
    ``get_session()``. ``tools`` optionally narrows the agent's tool schema
    to these names only — plain strings are normalized to bare ToolSpecs
    internally (no description, no parameters); the internal state machine
    still applies. Passing
    the same config (or session) to several agent() calls makes them share
    one context window; use ``branch()`` for per-step variations — dataclass
    slots mean ``**vars(cfg)`` destructuring is not available.
    """

    model: str | None = None
    session: Session | None = None
    tools: list[str | ToolSpec] | None = None
    profile: "AgentProfile | None" = None  # per-step profile override
    resume: bool = False  # replay completed agent steps instead of re-running
    _runtime: "WorkflowRuntime | None" = None  # attached by the runner

    _KNOWN_TOOLS = frozenset({"select", "edit", "close", "new", "delete", "cmd"})

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
               tools: list[str | ToolSpec] | None = None, resume: bool | None = None,
               profile: "AgentProfile | None" = None, **more) -> "AgentConfig":
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
            resume=resume if resume is not None else self.resume,
            profile=profile if profile is not None else self.profile,
            _runtime=self._runtime,
        )

    def fork(self, name: str | None = None) -> "AgentConfig":
        """Derive a config whose session is a clone of this one.

        A temporary branch: the forked config shares the model, tools, and
        runtime with this one, but its session is an independent copy —
        steps run on the fork never write to the original window. Call
        ``cfg.fork().session.delete()`` when the branch outlives its use,
        or ``cfg.fork(name="experiment")`` to pin the branch name.
        """
        session = self.get_session().clone(name)
        return AgentConfig(
            model=self.model,
            session=session,
            tools=self.tools,
            _runtime=self._runtime,
        )

    # ---- runtime primitives (delegated; cfg is the workflow's facade) ----

    def _rt(self) -> "WorkflowRuntime":
        if self._runtime is None:
            raise RuntimeError(
                "this config has no runtime attached: workflows receive one "
                "from the xg runner or the parent workflow; construct via "
                "run_workflow(name, cfg=...) instead of AgentConfig() directly"
            )
        return self._runtime

    def shell(self, command: str) -> int:
        """Run one trusted shell step as the workflow author, not the agent."""
        return self._rt().shell(command)

    def agent(self, prompt: str, config: "AgentConfig | None" = None) -> str:
        """Run one constrained agent turn (see WorkflowRuntime.agent)."""
        return self._rt().agent(prompt, config=config or self)

    def prompt(self, invitation: str = "describe what you want", session: Session | None = None) -> bool:
        """Yield control to the user; False on empty input/EOF."""
        return self._rt().prompt(invitation, session=session if session is not None else self.session,
                                 resume=self.resume)

    @property
    def request(self) -> str:
        """The text the user entered at the last prompt()."""
        return self._rt().request

    @property
    def root(self) -> "Path":
        """The workspace root this runtime executes in."""
        return self._rt().root

    @property
    def documentation(self) -> str:
        """The injected xgdf reference (see xg.docs)."""
        return self._rt().documentation

    def workflow(self, name: str) -> int:
        """Run another workflow (built-in or project) reusing this cfg.

        The child receives this config — same model, session, tools, and
        runtime root — so workflows compose like function calls.
        """
        return run_workflow(name, self._rt().root, cfg=self)


def default_model(cwd: str | Path = ".") -> str:
    """Resolve the xg default model: config.json, env, then built-in."""
    configured = load(cwd).get("model")
    if configured:
        return str(configured)
    import os

    return os.environ.get("XG_MODEL", "@preset/mimo")


class WorkflowRuntime:
    """Primitives handed to a workflow program."""

    def __init__(self, root: str | Path, request: str = ""):
        self.root = Path(root).resolve()
        self.request: str = request  # pre-loaded request (e.g. `xg "..."`)
        self.documentation: str = documentation_for(self.root)  # location-dependent, see docs.py
        # Resume playback: events from the recorded session, consumed in order.
        self._resume_session: Session | None = None
        self._resume_events: list = []
        self._resume_cursor: int = 0

    def _begin_resume(self, session: Session) -> None:
        """Start playback for a session (once): snapshot its event list."""
        if self._resume_session is not session:
            self._resume_session = session
            self._resume_events = list(session.events())
            self._resume_cursor = 0

    def _take_event(self, session: Session, kind: str, step_hash: str | None = None):
        """Consume the next recorded event for a resumed session.

        Returns the event payload if the next event matches (prompt text or
        final assistant text of the step); otherwise ends playback (cursor
        jumps to the end) and returns None — everything goes live.
        """
        self._begin_resume(session)
        if self._resume_cursor >= len(self._resume_events):
            return None  # current state reached: live from here
        event = self._resume_events[self._resume_cursor]
        if kind == "prompt" and event[0] == "prompt":
            self._resume_cursor += 1
            return event[1]
        if kind == "step" and event[0] == "step" and event[1] == step_hash:
            self._resume_cursor += 1
            for message in reversed(event[2]):
                if message.get("role") == "assistant":
                    return message.get("content", "")
            return ""
        # diverged (different event kind or hash): stop replaying
        self._resume_cursor = len(self._resume_events)
        return None

    def _complete_step(self, session: Session, step_hash: str) -> None:
        """Mark a step as completed (later marker wins in completed_step)."""
        if session.path is not None:
            session.append({"xgdf-step": {"hash": step_hash, "complete": True}})

    def _inject_documentation(self, session: Session) -> None:
        """Append the xgdf reference to a session window once.

        The window itself carries the framework docs — injected at session
        level, not per-turn — so any agent step reading the session (parent,
        fork, or a resumed run) sees them.
        """
        marker = "# xgdf reference"
        if not any(
            m.get("role") == "system" and marker in m.get("content", "")
            for m in session
        ):
            session.add("system", self.documentation)

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
        self._inject_documentation(session)   # the window itself carries the docs

        # Step resume: a completed step with the same hash is replayed from
        # the window instead of paying for inference again.
        import hashlib

        tool_names = "".join(sorted(
            spec.name if isinstance(spec, ToolSpec) else str(spec)
            for spec in (config.tools or [])
        ))
        step_hash = hashlib.sha1(
            f"{model}\0{tool_names}\0{prompt}".encode()
        ).hexdigest()[:12]
        if config.resume and not ephemeral:
            replayed = self._take_event(session, "step", step_hash)
            if replayed is not None:
                print(f"\n[workflow] agent step replayed from session ({session.name})")
                return replayed
        prompt_appended = (
            not ephemeral
            and session.last() == {"role": "user", "content": prompt}
        )
        print(f"\n[workflow] agent step (model={model}, session={session.name}{', ephemeral' if ephemeral else ''})")
        # Late import avoids the loader cycle: agent.run imports tools, and
        # workflows import neither at module load time.
        from xg.agent import ToolRejected, run as agent_run

        try:
            result = agent_run(
                str(self.root), prompt,
                lambda messages, tools=None: chat(messages, tools, model=model),
                profile=config.profile,
                tools=config.tools, session=session, prompt_appended=prompt_appended,
            )
            self._complete_step(session, step_hash)
            return result
        except ToolRejected as exc:
            print(f"\n[workflow] tool call rejected: {exc}")
            return f"tool call rejected: {exc}"
        finally:
            if ephemeral and session.path is not None and session.path.exists():
                session.path.unlink()

    def prompt(self, invitation: str = "describe what you want", session: Session | None = None,
               resume: bool = False) -> bool:
        """Yield control to the user; returns False on empty input/EOF.

        With ``session``, the user request is appended to that session's
        context window immediately, so the next agent() call with the same
        session sees it. With ``resume``, a recorded prompt answer is
        replayed instead of asking the user again.
        """
        if resume and session is not None:
            recorded = self._take_event(session, "prompt")
            if recorded is not None:
                print(f"\n\033[90m[resume] replaying recorded answer: {recorded}\033[0m")
                self.request = recorded
                return True
            # no more recorded prompts: ask for real (live from here)
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
            session.append({"xgdf-prompt": text})  # resume event
        return True


def _builtin_cmd(cfg: AgentConfig) -> int:
    """The built-in shell-command workflow (selected with ``-w xg-cmd``).

    Deliberately no workspace read: it needs one command, not file context.
    """
    from xg.profiles import command_profile

    session = cfg.get_session()
    if not cfg.request and not cfg.prompt("your command request", session=session):
        return 0
    command_cfg = cfg.branch(session=session, tools=["cmd"], profile=command_profile())
    cfg.agent(cfg.request, config=command_cfg)
    return 0


def _builtin_default(cfg: AgentConfig) -> int:
    """The default workflow: the constrained file-editing flow.

    Starts with the hardcoded deterministic workspace read (m-time ordered
    files, injected into the session once); subsequent agent turns never
    re-inject it.

    Uses a named (resumable) session so repeated `xg` invocations continue
    one JSONL window in the workspace.
    """
    session = cfg.get_session()
    from xg.utils import read_workspace
    read_workspace(session, cfg.root)  # deterministic reads, once, explicit
    print("tools: select -> edit|close | new(name, content) | delete\n")
    if not cfg.request and not cfg.prompt("your request", session=session):
        return 0
    cfg.agent(cfg.request)
    return 0


def last_session(root: str | Path) -> str | None:
    """Return the name of a workspace's most recently used session.

    Uses the ``.xg/sessions/.last`` pointer written by Session.bind(); falls
    back to the most recently modified session file (ephemeral sessions
    excluded — they are deleted after their run anyway).
    """
    directory = Path(root).resolve() / ".xg" / "sessions"
    pointer = directory / ".last"
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        if name and (directory / f"{name}.jsonl").is_file():
            return name
    candidates = [p for p in directory.glob("*.jsonl") if not p.name.startswith("temp-")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).removesuffix(".jsonl")


def _project_workflow(name: str, cwd: str | Path) -> Callable[[AgentConfig], int] | None:
    """Load a project workflow, searching from least to most specific."""
    selected: Path | None = None
    for directory in xg_directories(cwd):
        path = directory / "workflows" / f"{name}.py"
        if path.is_file():
            selected = path
    if selected is None:
        return None
    module_name = f"xg_workflow_{name}_{abs(hash(str(selected)))}"
    spec = importlib.util.spec_from_file_location(module_name, selected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workflow: {selected}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError(f"workflow must define run(cfg): {selected}")
    return run


def load_workflow(name: str, cwd: str | Path = ".") -> Callable[[AgentConfig], int]:
    """Load a workflow, resolving built-in aliases as one logical name.

    ``default`` and ``xg-default`` are aliases for the same workflow. A
    project-local ``default.py`` intentionally shadows that built-in for both
    names. The same rule applies to the ``cmd``/``xg-cmd`` aliases.
    """
    aliases = {
        "default": ("default", _builtin_default),
        "xg-default": ("default", _builtin_default),
        "cmd": ("cmd", _builtin_cmd),
        "xg-cmd": ("cmd", _builtin_cmd),
    }
    if name in aliases:
        local_name, builtin = aliases[name]
        local = _project_workflow(local_name, cwd)
        return local or builtin
    workflow = _project_workflow(name, cwd)
    if workflow is None:
        raise RuntimeError(f"workflow not found: {name}")
    return workflow


def run_workflow(name: str, cwd: str | Path = ".", cfg: AgentConfig | None = None) -> int:
    """Execute one workflow program.

    With ``cfg`` (a parent workflow reusing another), the child runs on the
    parent's config and runtime; otherwise a fresh one is built for ``cwd``
    — that fresh path is what ``xg -w <name>`` uses.
    """
    program = load_workflow(name, cwd)
    if cfg is None:
        cfg = AgentConfig(_runtime=WorkflowRuntime(cwd))
    elif cfg._runtime is None:
        import dataclasses

        cfg = dataclasses.replace(cfg, _runtime=WorkflowRuntime(cwd))
    return int(program(cfg) or 0)


def list_workflows(cwd: str | Path = ".") -> list[str]:
    """Return available workflow names, built-ins first."""
    # Packaged workflows are listed by their canonical xg-* names. Their
    # short names are invocation aliases, not additional workflows.
    names = ["xg-default", "xg-cmd"]
    seen = set()
    for directory in xg_directories(cwd):
        for path in sorted((directory / "workflows").glob("*.py")):
            if path.stem not in seen:
                seen.add(path.stem)
                names.append(path.stem)
    return names
