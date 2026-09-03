#!/usr/bin/env python3
"""xg: the Generative Development Framework CLI.

    xg                     run the default graph (interactive)
    xg PROMPT...           run the default graph with a request
    xg --resume [PATH] [SESSION]
                           replay a recorded session until its current state
    xg -g [NAME]           run a graph (default: xg-default)
    xg init                scaffold .xg/ in the current project

Built-in graph names are `xg-default` (also `default`) and `xg-cmd`
(also `cmd`). `xg-cmd` proposes and runs shell commands.

Every form has `--help` (`xg --help`, `xg init --help`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xg.init import initialize


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("py-xgdf")
    except Exception:
        return "unknown"


def _main_parser(resume_default=None) -> argparse.ArgumentParser:
    """The bare-`xg` parser: default graph + resume + named graph."""
    parser = argparse.ArgumentParser(
        prog="xg",
        description="the Generative Development Framework: a deterministic "
                    "coding agent driven by Python graphs.",
        epilog="built-in graphs:\n"
               "  xg-default  constrained coding graph (also: default)\n"
               "  xg-cmd      shell command graph (also: cmd)\n\n"
               "commands:\n"
               "  init        scaffold .xg/ here (xg init --help)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"xg {_version()}")
    parser.add_argument("prompt", nargs="*",
                        help="request text for the default (or named) graph")
    parser.add_argument("-r", "--resume", action="store_true", default=resume_default,
                        help="replay a recorded session until its current "
                             "state, then go live: no arguments resume the "
                             "last session of the current path; PATH resumes "
                             "another workspace's last session; PATH SESSION "
                             "picks one (arguments are consumed from right "
                             "after the flag; put the prompt before --resume "
                             "to combine both)")
    parser.add_argument("-g", "--graph", nargs="?", const="list", metavar="NAME",
                        help="list graphs when used alone; otherwise run NAME "
                             "(default: xg-default; aliases: default, cmd)")
    return parser


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xg init",
        description="scaffold .xg/ (config, graphs, session store) in "
                    "the current project")
    return parser


def _extract_resume(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Pop ``--resume`` (or ``-r``) plus up to two PATH/SESSION words.

    argparse can't cap an optional's nargs, so this runs before parsing.
    Returns (remaining argv, resume args or None if the flag is absent).
    """
    flag = None
    for candidate in ("--resume", "-r"):
        if candidate in argv:
            flag = candidate
            break
    if flag is None:
        return argv, None
    index = argv.index(flag)
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
    from xg.graphs import AgentConfig, Session, WorkflowRuntime, last_session

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
    session = Session(name=name)
    session.bind(root)  # Load the existing session file
    return AgentConfig(session=session, resume=True, _runtime=runtime)


def _list_graphs() -> int:
    """Print every built-in and project graph."""
    from xg.graphs import list_graphs

    print("available graphs:")
    for graph_name in list_graphs("."):
        print(f"  {graph_name}")
    return 0


def _default_graph(args) -> int:
    """Bare `xg`: the default (or named) graph, optionally resumed."""
    from xg.graphs import AgentConfig, WorkflowRuntime, run_graph

    prompt = _input_text(args.prompt)
    try:
        if args.resume is not None:
            cfg = _resume_config(args.resume, prompt)
        else:
            cfg = AgentConfig(_runtime=WorkflowRuntime(".", request=prompt))
        return run_graph(args.graph or "xg-default", ".", cfg=cfg)
    except RuntimeError as exc:
        print(f"xg: {exc}", file=sys.stderr)
        return 2



def main() -> int:
    """Run the xgdf runtime: everything routes through a graph."""
    argv = sys.argv[1:]

    # `init` remains the only explicit CLI command. Graphs, including
    # xg-cmd, are selected with `-w` and therefore share the normal runtime.
    if argv and argv[0] == "init":
        _init_parser().parse_args(argv[1:])
        return initialize(".")

    # Everything else is the bare-`xg` default graph.
    argv, resume_args = _extract_resume(argv)
    parser = _main_parser(resume_default=resume_args)
    args = parser.parse_args(argv)
    if args.graph in {"list", "ls"}:
        return _list_graphs()
    return _default_graph(args)


if __name__ == "__main__":
    raise SystemExit(main())
