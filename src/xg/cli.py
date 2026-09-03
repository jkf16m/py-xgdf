#!/usr/bin/env python3
"""xg: the Generative Development Framework CLI.

Commands:
    xg              run the default workflow (interactive)
    xg PROMPT       run the default workflow with a request
    xg -w NAME      run a named workflow; -w alone lists them
    xg --resume [PATH] [SESSION]
                    resume a recorded session: prompts and agent turns are
                    replayed from the JSONL until current state, then live.
                    No argument resumes the last session of the current
                    path; a directory argument resumes that path's last
                    session; PATH SESSION picks a specific one
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


def _resume_config(resume_args: list[str], request: str):
    """Build a resume cfg from optional PATH and SESSION arguments."""
    from xg.workflows import AgentConfig, Session, WorkflowRuntime, last_session

    path = Path(resume_args[0]) if resume_args else None
    if path is not None and path.is_dir():
        root = path.resolve()
        name = resume_args[1] if len(resume_args) > 1 else last_session(root)
        if name is None:
            raise RuntimeError(f"no session to resume in {root}")
    else:
        root = Path(".").resolve()
        name = resume_args[0] if resume_args else last_session(root)
        if name is None:
            raise RuntimeError("no session to resume in the current path")
    runtime = WorkflowRuntime(root, request=request)
    return AgentConfig(session=Session(name=name), resume=True, _runtime=runtime)


def main() -> int:
    """Run the xgdf runtime: everything routes through a workflow."""
    argv = sys.argv[1:]

    # `--resume [PATH] [SESSION]` can combine with `-w NAME` or a request;
    # pull it out first so the remaining argv keeps its old shape. No argument
    # means the last session of the current path; a directory means that
    # path's last session; PATH SESSION picks a specific one.
    resume_args: list[str] | None = None
    if "--resume" in argv:
        index = argv.index("--resume")
        argv.pop(index)
        resume_args = []
        while index < len(argv) and not argv[index].startswith("-") and len(resume_args) < 2:
            resume_args.append(argv.pop(index))

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
            from xg.workflows import AgentConfig, WorkflowRuntime, run_workflow

            cfg = (_resume_config(resume_args, "") if resume_args is not None
                   else AgentConfig())
            return run_workflow(argv[0], ".", cfg=cfg)
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
    try:
        cfg = (_resume_config(resume_args, prompt) if resume_args is not None
               else AgentConfig(_runtime=WorkflowRuntime(".", request=prompt)))
        return run_workflow("default", ".", cfg=cfg)
    except RuntimeError as exc:
        print(f"xg: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
