"""Single-prompt agent loop over the deterministic workspace protocol.

Public API:
    run(root, prompt, chat) -> str
"""

from __future__ import annotations

import json
import sys
from typing import Callable

from prompt_toolkit import prompt as line_prompt

from xg.profiles import AgentProfile, command_profile, default_profile, empty_profile
from xg.sdk import Agent
from xg.tools import ToolState, ToolSpec, dispatch, schemas
from xg.utils import format_patch, tool_patch


class ToolRejected(Exception):
    """Raised when the user rejects a proposed tool call."""


def run(root: str, prompt: str, chat: Callable, profile: AgentProfile | None = None, history: list[dict] | None = None, tools: list[str | ToolSpec] | None = None, session=None, prompt_appended: bool = False) -> str:
    """Run one prompt using a reusable, code-defined agent profile.

    ``session`` (a xg.graphs.Session) makes the conversation log itself
    the context window: every message is appended to its JSONL file and each
    model request is assembled by walking that file — the history is never
    loaded into RAM. ``prompt_appended`` marks that prompt() already wrote
    the request into the session, so only the profile context is added.
    The legacy ``history`` list path and imperative profiles ignore sessions.
    """
    tool_names: list[str] = []
    tool_specs: dict[str, ToolSpec] = {}
    for entry in tools or []:
        if isinstance(entry, ToolSpec):
            tool_names.append(entry.name)
            tool_specs[entry.name] = entry
        else:
            tool_names.append(entry)
    if session is not None:
        # The session-logged loop replaces imperative programs: the file is
        # the conversation, so the message loop below is the only valid path.
        profile = profile or default_profile()
        state = ToolState(root)
        if tools:
            state.restrict(tool_names)
        return _run_logged(root, prompt, chat, profile, state, tool_specs, session, prompt_appended)
    profile = profile or (empty_profile("workflow") if history is not None else default_profile())
    if profile.program is not None:
        program = Agent(root, prompt, chat, profile.system_prompt)
        profile.program(program)
        return ""
    user_content = f"REQUEST:\n{prompt}\n\nPROFILE CONTEXT:\n{profile.context(root)}"
    state = ToolState(root)
    if tools:
        state.restrict(tool_names)
    messages = [
        {"role": "system", "content": profile.system_prompt},
        *(history or []),
        {"role": "user", "content": user_content},
    ]
    while True:
        # Tool exposure is recomputed before every model request. Once a
        # workflow step starts, the model literally cannot see the other tools.
        message = chat(messages, schemas(state.allowed_tools(), overrides=tool_specs))
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return str(message.get("content") or "")
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            print(f"\nProposed tool call: {name}({fn.get('arguments', '{}')})")
            patch = tool_patch(state, name, fn.get("arguments", "{}"))
            if patch:
                print(format_patch(patch))
            if not _accept_tool():
                raise ToolRejected(name)
            try:
                result = dispatch(state, fn.get("name", ""), fn.get("arguments", "{}"))
            except Exception as exc:
                result = f"tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})


def _run_logged(root, prompt, chat, profile, state, tool_specs, session, prompt_appended) -> str:
    """Agent loop whose conversation log IS the session file.

    Each model request is built by streaming the session file line-by-line;
    every assistant reply and tool result is appended before the next
    request. RAM holds one message at a time.
    """
    profile = profile or default_profile()
    if not any(m.get("role") == "system" for m in session):  # O(1) RAM walk
        session.add("system", profile.system_prompt)
    # The workspace context is NOT injected here: deterministic file reads
    # are a one-time step of the workflow that needs them (xg-default calls
    # xg.utils.read_workspace(session, root) at its start), never a per-turn
    # injection.
    if not prompt_appended:
        session.add("user", prompt)
    while True:
        message = chat(session, schemas(state.allowed_tools(), overrides=tool_specs))
        session.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return str(message.get("content") or "")
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            print(f"\nProposed tool call: {name}({fn.get('arguments', '{}')})")
            patch = tool_patch(state, name, fn.get("arguments", "{}"))
            if patch:
                print(format_patch(patch))
            if not _accept_tool():
                session.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                "content": f"user rejected the tool call: {name}"})
                raise ToolRejected(name)
            try:
                result = dispatch(state, fn.get("name", ""), fn.get("arguments", "{}"))
            except Exception as exc:
                result = f"tool error: {exc}"
            session.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})


def _accept_tool() -> bool:
    """Ask the terminal user to accept or reject one tool call."""
    if not sys.stdin.isatty():
        raise ToolRejected("non-interactive tool call")
    try:
        return line_prompt("Accept this tool call? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False
