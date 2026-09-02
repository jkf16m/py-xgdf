#!/usr/bin/env python3
"""gdev: a single-prompt deterministic coding agent with staged sessions.

Commands:
    gdev --pty       launch an interactive shell behind a pseudo-terminal
    gdev PROMPT      create a new inactive coding-agent session
    gdev c [TEXT]    append input to the latest inactive session
    gdev r|run       execute the latest inactive session
    gdev cmd [PROMPT]  propose a shell command and invoke it (or inject it into
                     the active --pty shell prompt); without PROMPT, use TTY input
"""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_toolkit import prompt as line_prompt
from prompt_toolkit.history import FileHistory

from gdev.agent import ToolRejected, run
from gdev.init import initialize
from gdev.llm import chat
from gdev.pty import launch, propose
from gdev.profile_loader import load_profile
from gdev.profiles import AgentProfile
from gdev.session import SessionStore
from gdev.workspace import files


def _input_text(arguments: list[str]) -> str:
    """Combine command arguments and piped stdin into one request fragment."""
    pieces = [" ".join(arguments).strip()]
    if not sys.stdin.isatty():
        pieces.append(sys.stdin.read().strip())
    return "\n\n".join(piece for piece in pieces if piece)


def _run_session(store: SessionStore, session, profile: AgentProfile | None = None) -> int:
    """Run and permanently close one inactive session."""
    store.begin(session)
    print(f"running session {session.id}")
    rejected = False
    try:
        count = len(files("."))
        print(f"forced reads: {count} files, oldest to newest by mtime")
        print("tools: select -> edit | new -> content\n")
        run(".", session.prompt, chat, profile=profile)
    except ToolRejected as exc:
        rejected = True
        print(f"\nRejected tool call: {exc}")
    finally:
        store.close(session)
        print(f"\nclosed session {session.id}; start a new session with `gdev \"request\"`")
    if rejected:
        # Remove the streamed answer and tool proposal before returning to input.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print("Tool call rejected. The previous attempt was discarded.")
        return _interactive(store, profile)
    return 0


def _interactive_cmd() -> int:
    """Collect a command request interactively, like bare ``gdev``."""
    Path(".gdev").mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(Path(".gdev") / "cmd-history"))
    context_lines: list[str] = []
    print("\033[90mType a command request. Submit an empty line to ask the agent; Ctrl-C exits.\033[0m")
    try:
        while True:
            value = line_prompt("cmd >> ", history=history).strip()
            if not value:
                # Remove the submission prompt before rendering the command
                # approval UI, keeping the operator display compact.
                sys.stdout.write("\033[1A\033[2K\r")
                sys.stdout.flush()
                if not context_lines:
                    print("\033[90mAdd a request first, or press Ctrl-C to exit.\033[0m")
                    continue
                prompt = "\n\n".join(context_lines)
                history.append_string(prompt)
                return propose(prompt, chat)
            context_lines.append(value)
    except (EOFError, KeyboardInterrupt):
        print("\ncommand request cancelled")
        return 130


def _interactive(store: SessionStore, profile: AgentProfile | None = None) -> int:
    """Collect context until an empty request starts inference."""
    history = FileHistory(str(Path(".gdev") / "history"))
    session = store.create("")
    print("\033[90mType context. Submit an empty request to start inference; Ctrl-C exits and saves it.\033[0m")
    try:
        while True:
            value = line_prompt(">> ", history=history).strip()
            if not value:
                if not session.prompt.strip():
                    print("\033[90mAdd context first, or press Ctrl-C to save and exit.\033[0m")
                    continue
                history.append_string(session.prompt)
                return _run_session(store, session, profile)
            session.prompt = f"{session.prompt}\n\n{value}" if session.prompt else value
            store._save(session)
    except (EOFError, KeyboardInterrupt):
        if session.prompt.strip():
            history.append_string(session.prompt)
        print("\ncontext saved; continue it with `gdev c`")
        return 130


def main() -> int:
    """Create, continue, or run a deterministic request session."""
    argv = sys.argv[1:]
    profile = None
    if len(argv) >= 2 and argv[0] in {"-a", "--agent"}:
        try:
            profile = load_profile(argv[1], ".")
        except RuntimeError as exc:
            print(f"gdev: {exc}", file=sys.stderr)
            return 2
        argv = argv[2:]
    if argv and argv[0].lower() == "init":
        if len(argv) != 1:
            print("usage: gdev init", file=sys.stderr)
            return 2
        return initialize(".")

    if argv and argv[0] == "--pty":
        if len(argv) != 1:
            print("gdev: --pty does not accept arguments", file=sys.stderr)
            return 2
        return launch()

    if argv and argv[0].lower() == "cmd":
        prompt = _input_text(argv[1:])
        if not prompt:
            if sys.stdin.isatty():
                return _interactive_cmd()
            print("usage: gdev cmd PROMPT", file=sys.stderr)
            return 2
        return propose(prompt, chat)

    store = SessionStore(".")
    command = argv[0].lower() if argv and argv[0].lower() in {"c", "r", "run"} else None

    if command == "c":
        text = _input_text(argv[1:])
        try:
            session = store.latest_open()
            if not text and sys.stdin.isatty():
                if session is None:
                    raise RuntimeError("no inactive session to inspect")
                print(session.prompt or "(no context added yet)")
                return 0
            session = store.continue_latest(text)
        except RuntimeError as exc:
            print(f"gdev: {exc}", file=sys.stderr)
            return 1
        print(f"continued session {session.id}; run `gdev r` when ready")
        return 0

    if command in {"r", "run"}:
        session = store.latest_open()
        if session is None:
            print("gdev: no inactive session; start one with `gdev \"request\"`", file=sys.stderr)
            return 1
        return _run_session(store, session, profile)

    if not argv and sys.stdin.isatty():
        return _interactive(store, profile)

    prompt = _input_text(argv)
    if not prompt:
        return 0
    session = store.create(prompt)
    FileHistory(str(Path(".gdev") / "history")).append_string(prompt)
    print(f"created session {session.id}; run `gdev r` when ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
