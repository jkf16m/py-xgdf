#!/usr/bin/env python3
"""xg: the Generative Development Framework CLI.

Commands (see `xg --help`):
    xg [run] [PROMPT...] [--resume [PATH] [SESSION]] [-w NAME]
                    run the default workflow (interactive without a PROMPT);
                    --resume replays a recorded session until its current
                    state, then goes live
    xg workflow [NAME|list]
                    run a named workflow; without a name, list them
    xg cmd [PROMPT] propose a shell command and invoke it (or inject it into
                    the active --pty shell prompt); without PROMPT, use TTY input
    xg pty          launch an interactive shell behind a pseudo-terminal
    xg init         scaffold .xg/ in the current project

`xg -w NAME`, `xg --pty` and bare `xg PROMPT...` keep working: arguments
that start with a known command word route to that command, anything else
routes to `run`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import prompt as line_prompt
from prompt_toolkit.history import FileHistory

from xg.init import initialize
from xg.llm import chat
from xg.pty import launch, propose

_COMMANDS = ("run", "workflow", "cmd", "pty", "init")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("py-xgdf")
    except Exception:
        return "unknown"


def _build_parser(resume_default=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xg", description="the Generative Development Framework: a "
        "deterministic coding agent driven by Python workflows.")
    parser.add_argument("--version", action="version", version=f"xg {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_run = sub.add_parser("run", help="run the default workflow "
                           "(the implicit command for `xg [PROMPT...]`)")
    p_run.add_argument("prompt", nargs="*", help="pre-loaded request text")
    p_run.add_argument("--resume", default=resume_default, action="store_true",
                       help="replay a recorded session until its current "
                       "state, then go live: no argument resumes the last "
                       "session of the current path; PATH resumes another "
                       "workspace's last session; PATH SESSION picks one "
                       "(arguments are consumed from right after the flag; "
                       "put the prompt before --resume to combine both)")
    p_run.add_argument("-w", "--workflow", default=None, metavar="NAME",
                       help="run a named workflow instead of `default`")

    p_wf = sub.add_parser("workflow", help="run a named workflow "
                          "(`workflow list` lists them)")
    p_wf.add_argument("name", nargs="?", default="list",
                      help="workflow name, or `list` (default: list)")

    p_cmd = sub.add_parser("cmd", help="propose a shell command and invoke it")
    p_cmd.add_argument("prompt", nargs="*", help="command request")

    sub.add_parser("pty", help="launch an interactive shell behind a "
                   "pseudo-terminal")
    sub.add_parser("init", help="scaffold .xg/ in the current project")
    return parser


def _route(argv: list[str]) -> list[str]:
    """Prepend the implicit `run` command for legacy/loose invocations.

    `xg PROMPT...`, `xg -w NAME`, `xg --pty`, `xg --resume ...` all keep
    working by routing to the matching real command.
    """
    if not argv or argv[0] not in _COMMANDS:
        if set(argv) <= {"-h", "--help", "--version"}:
            return argv  # top-level help/version belongs to the main parser
        if argv and argv[0] == "--pty":
            return ["pty"]
        if argv and argv[0] in {"-w", "--workflow"}:
            return ["workflow", *argv[1:]]
        return ["run", *argv]
    return argv


def _extract_resume(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Pop ``--resume`` plus up to two PATH/SESSION words from argv.

    argparse can't cap an optional's nargs, so this runs before parsing.
    Returns (remaining argv, resume args or None if the flag is absent).
    """
    if "--resume" not in argv:
        return argv, None
    index = argv.index("--resume")
    argv = argv[:index] + argv[index + 1:]
    args: list[str] = []
    while (index < len(argv) and not argv[index].startswith("-")
           and len(args) < 2):
        args.append(argv.pop(index))
    return argv, args


def _input_text(arguments: list[str]) -> str:
    """Combine command arguments and piped stdin into one request fragment."""
    pieces = [" ".join(arguments).strip()]
    if not sys.stdin.isatty():
        pieces.append(sys.stdin.read().strip())
    return "\n\n".join(piece for piece in pieces if piece)


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


def _run_command(args) -> int:
    """`xg [run]`: the default workflow, optionally with --resume."""
    from xg.workflows import AgentConfig, WorkflowRuntime, run_workflow

    prompt = _input_text(args.prompt)
    try:
        if args.resume is not None:
            cfg = _resume_config(args.resume, prompt)
        else:
            cfg = AgentConfig(_runtime=WorkflowRuntime(".", request=prompt))
        return run_workflow(args.workflow or "default", ".", cfg=cfg)
    except RuntimeError as exc:
        print(f"xg: {exc}", file=sys.stderr)
        return 2


def _workflow_command(args) -> int:
    """`xg workflow [NAME|list]`."""
    from xg.workflows import list_workflows, run_workflow

    if args.name in {"ls", "list"}:
        print("available workflows:")
        for workflow_name in list_workflows("."):
            print(f"  {workflow_name}")
        return 0
    try:
        return run_workflow(args.name, ".")
    except RuntimeError as exc:
        print(f"xg: {exc}", file=sys.stderr)
        return 2


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


def _cmd_command(args) -> int:
    prompt = _input_text(args.prompt)
    if not prompt:
        if sys.stdin.isatty():
            return _interactive_cmd()
        print("usage: xg cmd PROMPT", file=sys.stderr)
        return 2
    return propose(prompt, chat)


def main() -> int:
    """Run the xgdf runtime: everything routes through a workflow."""
    argv = _route(sys.argv[1:])
    argv, resume_args = _extract_resume(argv)
    parser = _build_parser(resume_default=resume_args)
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "workflow":
        return _workflow_command(args)
    if args.command == "cmd":
        return _cmd_command(args)
    if args.command == "pty":
        return launch()
    if args.command == "init":
        if len(sys.argv[1:]) != 1:
            print("usage: xg init", file=sys.stderr)
            return 2
        return initialize(".")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
