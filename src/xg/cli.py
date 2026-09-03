#!/usr/bin/env python3
"""xg: the Generative Development Framework CLI.

Commands:
    xg              run the default workflow (interactive)
    xg PROMPT       run the default workflow with a request
    xg -w NAME      run a named workflow; -w alone lists them
    xg init         scaffold .xg/ in the current project
    xg --pty        launch an interactive shell behind a pseudo-terminal
    xg cmd [PROMPT] propose a shell command and invoke it (or inject it into
                    the active --pty shell prompt); without PROMPT, use TTY input
"""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_toolkit import prompt as line_prompt
from prompt_toolkit.history import FileHistory

from xg.init import initialize
from xg.llm import chat
from xg.pty import launch, propose


def _input_text(arguments: list[str]) -> str:
    """Combine command arguments and piped stdin into one request fragment."""
    pieces = [" ".join(arguments).strip()]
    if not sys.stdin.isatty():
        pieces.append(sys.stdin.read().strip())
    return "\n\n".join(piece for piece in pieces if piece)


def _interactive_cmd() -> int:
    """Collect a command request interactively, like bare ``xg``."""
    Path(".xg").mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(Path(".xg") / "cmd-history"))
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


def main() -> int:
    """Run the xgdf runtime: everything routes through a workflow."""
    argv = sys.argv[1:]

    if argv and argv[0].lower() == "init":
        if len(argv) != 1:
            print("usage: xg init", file=sys.stderr)
            return 2
        return initialize(".")

    if argv and argv[0] == "--pty":
        if len(argv) != 1:
            print("xg: --pty does not accept arguments", file=sys.stderr)
            return 2
        return launch()

    if argv and argv[0] in {"-w", "--workflow"}:
        argv = argv[1:]
        if not argv or argv[0] in {"ls", "list"}:
            if argv and len(argv) != 1:
                print("usage: xg --workflow [NAME|list]", file=sys.stderr)
                return 2
            from xg.workflows import list_workflows

            print("available workflows:")
            for workflow_name in list_workflows("."):
                print(f"  {workflow_name}")
            return 0
        if len(argv) != 1:
            print("usage: xg --workflow [NAME|list]", file=sys.stderr)
            return 2
        try:
            from xg.workflows import run_workflow

            return run_workflow(argv[0], ".")
        except RuntimeError as exc:
            print(f"xg: {exc}", file=sys.stderr)
            return 2

    if argv and argv[0].lower() == "cmd":
        prompt = _input_text(argv[1:])
        if not prompt:
            if sys.stdin.isatty():
                return _interactive_cmd()
            print("usage: xg cmd PROMPT", file=sys.stderr)
            return 2
        return propose(prompt, chat)

    # Everything else routes through the default workflow: bare `xg` asks
    # for a request interactively; `xg PROMPT...` runs with the request
    # pre-loaded. Same workflow, same session schema as `xg -w default`.
    from xg.workflows import AgentConfig, WorkflowRuntime, run_workflow

    prompt = _input_text(argv)
    runtime = WorkflowRuntime(".", request=prompt)
    try:
        return run_workflow("default", ".", cfg=AgentConfig(_runtime=runtime))
    except RuntimeError as exc:
        print(f"xg: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
