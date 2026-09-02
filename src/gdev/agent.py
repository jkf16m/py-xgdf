"""Single-prompt agent loop over the deterministic workspace protocol.

Public API:
    run(root, prompt, chat) -> str
"""

from __future__ import annotations

import json
import sys
from typing import Callable

from prompt_toolkit import prompt as line_prompt

from gdev.profiles import AgentProfile, default_profile
from gdev.sdk import Agent
from gdev.tools import ToolState, dispatch, schemas


class ToolRejected(Exception):
    """Raised when the user rejects a proposed tool call."""


def run(root: str, prompt: str, chat: Callable, profile: AgentProfile | None = None) -> str:
    """Run one prompt using a reusable, code-defined agent profile."""
    profile = profile or default_profile()
    if profile.program is not None:
        program = Agent(root, prompt, chat, profile.system_prompt)
        profile.program(program)
        return ""
    user_content = f"REQUEST:\n{prompt}\n\nPROFILE CONTEXT:\n{profile.context(root)}"
    state = ToolState(root)
    messages = [
        {"role": "system", "content": profile.system_prompt},
        {"role": "user", "content": user_content},
    ]
    while True:
        # Tool exposure is recomputed before every model request. Once a
        # workflow step starts, the model literally cannot see the other tools.
        message = chat(messages, schemas(state.allowed_tools()))
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return str(message.get("content") or "")
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            print(f"\nProposed tool call: {name}({fn.get('arguments', '{}')})")
            if name == "edit":
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    print(state.preview_edit(args.get("old_text", ""), args.get("new_text", "")))
                except Exception as exc:
                    print(f"preview unavailable: {exc}")
            if not _accept_tool():
                raise ToolRejected(name)
            try:
                result = dispatch(state, fn.get("name", ""), fn.get("arguments", "{}"))
            except Exception as exc:
                result = f"tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})


def _accept_tool() -> bool:
    """Ask the terminal user to accept or reject one tool call."""
    if not sys.stdin.isatty():
        raise ToolRejected("non-interactive tool call")
    try:
        return line_prompt("Accept this tool call? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False
