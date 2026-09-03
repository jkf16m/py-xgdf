"""Reusable built-in agent profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from xg.sdk import profile
from xg.workspace import context


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
    delete_call = agent.tool("delete")
    agent.call(select_call, new_call, delete_call)
    if select_call.used:
        # After a select, only edit() or close() is valid.
        edit_call = agent.tool("edit")
        close_call = agent.tool("close")
        agent.call(edit_call, close_call)
        if close_call.used:
            agent.call(agent.tool("select"), agent.tool("new"), agent.tool("delete"))


def default_profile() -> AgentProfile:
    """The standard xg agent: deterministic reads of the whole workspace."""
    return AgentProfile(
        name="default",
        context=lambda root: context(root),
        system_prompt=(
            "You are a deterministic coding agent. The complete workspace was "
            "read before this turn and is below. Use select(path) then edit(); "
            "close() unselects and returns to the previous step. Use new(name, "
            "content) to create a file in one operation. Use delete(path) to "
            "remove a file (the user confirms). There is no shell access. "
            "Do not invent paths. If no file operation is needed, simply "
            "reply in text (one concise answer, no tool call). "
            "Return a concise final report when finished."
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
