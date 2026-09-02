"""Initialization of a local .gdev subproject."""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path


def _ensure_ignored(project: Path) -> None:
    """Ensure the local gdev environment is not committed."""
    path = project / ".gitignore"
    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if any(line.strip().rstrip("/") == ".gdev/.venv" for line in content.splitlines()):
        return
    prefix = content if not content or content.endswith("\n") else content + "\n"
    separator = "" if not prefix or prefix.endswith("\n\n") else "\n"
    try:
        path.write_text(prefix + separator + ".gdev/.venv/\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"could not update {path}: {exc}") from exc


def initialize(cwd: str | Path = ".") -> int:
    """Create ``.gdev` config and a private environment containing gdev."""
    project = Path(cwd).resolve()
    try:
        _ensure_ignored(project)
    except RuntimeError as exc:
        print(f"gdev init: {exc}", file=sys.stderr)
        return 1
    directory = project / ".gdev"
    environment = directory / ".venv"
    config = directory / "config.json"
    directory.mkdir(parents=True, exist_ok=True)

    if config.exists():
        try:
            value = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"gdev init: invalid existing config.json: {exc}", file=sys.stderr)
            return 1
        if not isinstance(value, dict):
            print("gdev init: config.json must contain an object", file=sys.stderr)
            return 1
    else:
        config.write_text("{}\n", encoding="utf-8")

    if not environment.exists():
        print(f"creating {environment}")
        venv.EnvBuilder(with_pip=True, clear=False).create(environment)

    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    package_root = Path(__file__).resolve().parents[1]
    if (package_root / "pyproject.toml").is_file():
        package = str(package_root)
    else:
        package = "gdev"
    print("installing gdev into the local environment")
    try:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", package], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"gdev init: could not install gdev: {exc}", file=sys.stderr)
        return exc.returncode or 1

    print(f"initialized {directory}")
    print(f"python: {python}")
    print(f"config: {config}")
    return 0
