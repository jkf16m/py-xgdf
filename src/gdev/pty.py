"""PTY launcher and command injection support for :command:`gdev --pty`."""

from __future__ import annotations

import json
import os
import pty
import shutil
import subprocess
import sys
import termios
import struct
import fcntl
import socket
import tempfile
import threading
from pathlib import Path

from prompt_toolkit import prompt as line_prompt
from prompt_toolkit.completion import WordCompleter


_MASTER_FD: int | None = None


def _operator(path: str) -> None:
    """Receive shell input requests from gdev and write them to the PTY master."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(path)
        server.listen(4)
        while True:
            connection, _ = server.accept()
            with connection:
                payload = connection.recv(65536)
            if not payload:
                continue
            submit = payload[:1] == b"y"
            command = payload[1:]
            if _MASTER_FD is not None:
                os.write(_MASTER_FD, command + (b"\n" if submit else b""))
    except OSError:
        return
    finally:
        server.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def launch() -> int:
    """Run the user's shell behind a real pseudo-terminal."""
    shell = os.environ.get("SHELL") or "/bin/sh"
    if not Path(shell).exists():
        shell = shutil.which(shell) or "/bin/sh"

    old = os.environ.get("GDEV_PTY")
    socket_path = tempfile.mktemp(prefix="gdev-pty-", dir="/tmp")
    operator = threading.Thread(target=_operator, args=(socket_path,), daemon=True)
    operator.start()
    os.environ["GDEV_PTY"] = "1"
    os.environ["GDEV_PTY_SOCKET"] = socket_path
    try:
        # pty.spawn handles raw mode, terminal copying, and restoring the
        # caller's terminal even when the shell exits via Ctrl-D.
        status = pty.spawn([shell, "-i"], master_read=_master_read)
        return os.waitstatus_to_exitcode(status)
    finally:
        if old is None:
            os.environ.pop("GDEV_PTY", None)
        else:
            os.environ["GDEV_PTY"] = old
        os.environ.pop("GDEV_PTY_SOCKET", None)


def _master_read(fd: int) -> bytes:
    """Copy PTY output and keep its window size in sync with the terminal."""
    global _MASTER_FD
    _MASTER_FD = fd
    try:
        size = shutil.get_terminal_size()
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", size.lines, size.columns, 0, 0))
    except OSError:
        pass
    return os.read(fd, 1024)


def _command_from_answer(answer: str) -> str:
    """Turn the command-only model response into one shell command."""
    text = answer.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"sh", "bash", "shell", "zsh"}:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("command:"):
            text = line.split(":", 1)[1].strip()
            break
    # The command proposal protocol is deliberately single-command. This
    # prevents an accidental explanatory paragraph becoming shell input.
    command = next((line.strip().lstrip("$").strip() for line in text.splitlines() if line.strip()), "")
    if not command or "\x00" in command:
        raise ValueError("the agent did not return a shell command")
    return command


def _cmd_schema() -> list[dict]:
    """Expose only the command tool during a ``gdev cmd`` turn."""
    return [{
        "type": "function",
        "function": {
            "name": "cmd",
            "description": "Propose one shell command to accomplish the request.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }]


def _executables() -> list[str]:
    """Return executable names used for command-name completion."""
    names: set[str] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            for entry in os.scandir(directory or "."):
                if entry.is_file() and os.access(entry.path, os.X_OK):
                    names.add(entry.name)
        except OSError:
            continue
    return sorted(names)


def _clear_lines(count: int) -> None:
    """Erase the most recent terminal lines and return to their start."""
    if count:
        sys.stdout.write(f"\033[{count}A")
    for index in range(count):
        sys.stdout.write("\033[2K\r")
        if index + 1 < count:
            sys.stdout.write("\033[1B")
    if count:
        sys.stdout.write(f"\033[{count - 1}A" if count > 1 else "")
    sys.stdout.flush()


def _inject_pty(command: str, submit: bool) -> str | None:
    """Inject keyboard input into the controlling PTY, optionally Enter."""
    socket_path = os.environ.get("GDEV_PTY_SOCKET")
    if socket_path:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(socket_path)
                connection.sendall((b"y" if submit else b"e") + command.encode())
            return None
        except OSError as exc:
            return f"could not send command to PTY operator: {exc}"
    return "GDEV_PTY_SOCKET is not available"


def _invoke(command: str) -> str:
    """Approve a cmd call and put it into the real PTY shell prompt."""
    if os.environ.get("GDEV_PTY"):
        print("[y/N/e]")
        print(f"\033[2;32m$ {command}\033[0m")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "n"
        _clear_lines(3)
        if choice not in {"y", "yes", "e", "edit"}:
            print("command rejected")
            return "rejected by user"
        execute = choice in {"y", "yes"}
        if execute:
            error = _inject_pty(command, submit=True)
            if error:
                return error
            print(f"\033[2;32m$ {command}\033[0m")
            print("-----")
            return "command inserted into the PTY prompt and submitted for execution"
        # Return the command without injecting it. propose() ends the model
        # turn first, then performs the PTY handoff as its final operation.
        return "__GDEV_PTY_EDIT_HANDOFF__" + command

    # Without a controlling gdev PTY, provide a local editable fallback.
    print("\nProposed command (edit it, Tab autocompletes command names):")
    try:
        command = line_prompt(">> ", default=command, completer=WordCompleter(_executables(), sentence=True)).strip()
    except (EOFError, KeyboardInterrupt):
        command = ""
    if not command:
        print("command rejected")
        return "rejected by user"

    result = subprocess.run(
        [os.environ.get("SHELL", "/bin/sh"), "-lc", command],
        capture_output=True, text=True,
    )
    output = (result.stdout + result.stderr).strip()
    return f"exit {result.returncode}\n{output}" if output else f"exit {result.returncode}"


def propose(prompt: str, chat) -> int:
    """Run one agent turn with the command tool and no standard tools."""
    messages = [
        {"role": "system", "content": (
            "You are a shell command agent. You have exactly one tool: cmd. "
            "Use cmd to propose commands; do not call any other tool. "
            "Propose safe, useful commands and use one command per call."
        )},
        {"role": "user", "content": prompt},
    ]
    while True:
        message = chat(messages, _cmd_schema())
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return 0
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != "cmd":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                command = _command_from_answer(arguments.get("command", ""))
                result = _invoke(command)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                result = f"invalid cmd tool arguments: {exc}"
            if result.startswith("__GDEV_PTY_EDIT_HANDOFF__"):
                command = result.removeprefix("__GDEV_PTY_EDIT_HANDOFF__")
                error = _inject_pty(command, submit=False)
                if error:
                    print(f"gdev cmd: {error}", file=sys.stderr)
                    return 1
                return 0
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": result,
            })
