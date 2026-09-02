"""Load trusted Python tool modules from layered .gdev directories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gdev.config import gdev_directories
from gdev.sdk import ToolRegistry


def load_tools(cwd: str | Path, registry: ToolRegistry) -> None:
    """Load ``.gdev/tools/*.py`` from least to most specific directories.

    A script must expose ``register(registry)``. Loading is explicit and
    ordered; Python files are trusted code, not data files.
    """
    for index, directory in enumerate(gdev_directories(cwd)):
        tools_dir = directory / "tools"
        for path in sorted(tools_dir.glob("*.py")):
            module_name = f"gdev_user_tool_{index}_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load gdev tool: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if not callable(register):
                raise RuntimeError(f"gdev tool must define register(registry): {path}")
            register(registry)
