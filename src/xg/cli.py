#!/usr/bin/env python3
"""xg: the Generative Development Framework CLI.

    xg                     run the default workflow (interactive)
    xg PROMPT...           run the default workflow with a request
    xg --resume [PATH] [SESSION]
                           replay a recorded session until its current state
    xg -w [NAME]           run (or list) a named workflow
    xg workflow [NAME|list]
                           same as -w, as an explicit command
    xg cmd [PROMPT]        propose a shell command and invoke it
    xg init                scaffold .xg/ in the current project

Every form has `--help` (`xg --help`, `xg workflow --help`, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import prompt as line_prompt
from prompt_toolkit.history import FileHistory

from xg.init import initialize
from xg.llm import chat
from xg.pty import propose


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("py-xgdf")
    except Exception:
        return "unknown"


def _main_parser(resume_default=None) -> argparse.ArgumentParser:
    """The bare-`xg` parser: default workflow + resume + named workflow."""
    parser = argparse.ArgumentParser(
        prog="xg",
        description="the Generative Development Framework: a deterministic "
                    "coding agent driven by Python workflows.",
        epilog="commands:\n"
               "  workflow  run or list a named workflow  (xg workflow --help)\n"
               "  cmd       propose a shell command         (xg cmd --help)\n"
               "  init      scaffold .xg/ here             (xg init --help)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"xg {_version()}")
    parser.add_argument("prompt", nargs="*",
                        help="request text for the default (or named) workflow")
    parser.add_argument("--resume", action="store_true", default=resume_default,
                        help="replay a recorded session until its current "
                             "state, then go live: no arguments resume the "
                             "last session of the current path; PATH resumes "
                             "another workspace's last session; PATH SESSION "
                             "picks one (arguments are consumed from right "
                             "after the flag; put the prompt before --resume "
                             "to combine both)")
    parser.add_argument("-w", "--workflow", nargs="?", const="list", metavar="NAME",
                        help="run a named workflow (`-w` alone lists them)")
    return parser


def _workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xg workflow",
        description="run a named workflow, or list the available ones")
    parser.add_argument("name", nargs="?", default="list",
                        help="workflow name, or `list` (default: list)")
    return parser


def _cmd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xg cmd",
        description="propose a shell command and invoke it; without a "
                    "PROMPT, collect the request interactively")
    parser.add_argument("prompt", nargs="*", help="command request")
    return parser


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xg init",
        description="scaffold .xg/ (config, workflows, session store) in "
                    "the current project")
    return parser


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


def _default_workflow(args) -> int:
    """Bare `xg`: the default (or named) workflow, optionally resumed."""
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


def _named_workflow(name: str) -> int:
    """`xg -w NAME` / `xg workflow NAME`."""
    from xg.workflows import list_workflows, run_workflow

    if name in {"ls", "list"}:
        print("available workflows:")
        for workflow_name in list_workflows("."):
            print(f"  {workflow_name}")
        return 0
    try:
        return run_workflow(name, ".")
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
    argv = sys.argv[1:]

    # Explicit commands dispatch to their own parsers (each with --help).
    if argv and argv[0] == "workflow":
        return _named_workflow(_workflow_parser().parse_args(argv[1:]).name)
    if argv and argv[0] == "cmd":
        return _cmd_command(_cmd_parser().parse_args(argv[1:]))
    if argv and argv[0] == "init":
        _init_parser().parse_args(argv[1:])
        return initialize(".")

    # Everything else is the bare-`xg` default workflow.
    argv, resume_args = _extract_resume(argv)
    parser = _main_parser(resume_default=resume_args)
    args = parser.parse_args(argv)
    if args.workflow:  # `-w` alone lists, `-w NAME` runs that workflow
        return _named_workflow(args.workflow)
    return _default_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
