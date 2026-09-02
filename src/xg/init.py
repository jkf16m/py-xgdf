"""Initialization of a local .xg subproject."""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path


def _ensure_ignored(project: Path) -> None:
    """Ensure the local xg environment is not committed."""
    path = project / ".gitignore"
    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if any(line.strip().rstrip("/") == ".xg/.venv" for line in content.splitlines()):
        return
    prefix = content if not content or content.endswith("\n") else content + "\n"
    separator = "" if not prefix or prefix.endswith("\n\n") else "\n"
    try:
        path.write_text(prefix + separator + ".xg/.venv/\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"could not update {path}: {exc}") from exc


def _package_root() -> Path | None:
    """Repo root containing this checkout's pyproject.toml, if any."""
    for parent in Path(__file__).resolve().parents:
        if not (parent / "pyproject.toml").is_file():
            continue
        if (parent / "src" / "xg" / "init.py").is_file() or (parent / "xg" / "init.py").is_file():
            return parent
    return None


def initialize(cwd: str | Path = ".") -> int:
    """Create ``.xg` config and a private environment containing xg."""
    project = Path(cwd).resolve()
    try:
        _ensure_ignored(project)
    except RuntimeError as exc:
        print(f"xg init: {exc}", file=sys.stderr)
        return 1
    directory = project / ".xg"
    environment = directory / ".venv"
    config = directory / "config.json"
    directory.mkdir(parents=True, exist_ok=True)

    if config.exists():
        try:
            value = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"xg init: invalid existing config.json: {exc}", file=sys.stderr)
            return 1
        if not isinstance(value, dict):
            print("xg init: config.json must contain an object", file=sys.stderr)
            return 1
    else:
        config.write_text("{}\n", encoding="utf-8")

    if not environment.exists():
        print(f"creating {environment}")
        venv.EnvBuilder(with_pip=True, clear=False).create(environment)

    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    package_root = _package_root()
    package = str(package_root) if package_root else "xg"
    print("installing xg into the local environment")
    try:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", package], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"xg init: could not install xg: {exc}", file=sys.stderr)
        return exc.returncode or 1

    print(f"initialized {directory}")
    print(f"python: {python}")
    print(f"config: {config}")
    return 0
