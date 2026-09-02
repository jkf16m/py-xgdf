"""Reusable built-in agent profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gdev.sdk import profile
from gdev.workspace import context


@dataclass(frozen=True)
class AgentProfile:
    """Code-defined policy for one agent turn."""

    name: str
    context: Callable[[str], str]
    system_prompt: str
    tools: str = "standard"
    program: Callable | None = None


@profile
def default_program(agent) -> None:
    """Reusable imperative default program."""
    agent.context.read_workspace()
    select_call = agent.tool("select")
    new_call = agent.tool("new")
    agent.call(select_call, new_call)
    if select_call.used:
        agent.call(agent.tool("edit"))
    elif new_call.used:
        agent.call(agent.tool("content"))


def default_profile() -> AgentProfile:
    """The standard gdev agent: deterministic reads of the whole workspace."""
    return AgentProfile(
        name="default",
        context=lambda root: context(root),
        system_prompt=(
            "You are a deterministic coding agent. The complete workspace was "
            "read before this turn and is below. Use select(path) before edit; "
            "use new(name) followed immediately by content(content) for new files. "
            "Do not invent paths. Return a concise final report when finished."
        ),
        program=default_program,
    )


def empty_profile(name: str = "empty") -> AgentProfile:
    """Profile base with no automatic context; useful for custom agents."""
    return AgentProfile(
        name=name,
        context=lambda root: "(profile supplied no automatic context)",
        system_prompt="You are a deterministic agent. Use only the tools exposed to you.",
    )
