"""Load code-defined profiles from layered .xg/agent directories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from xg.config import xg_directories
from xg.profiles import AgentProfile, default_profile, empty_profile


def load_profile(name: str, cwd: str | Path = ".") -> AgentProfile:
    """Load the nearest code profile named ``name``.

    A profile is a Python file at ``.xg/agents/<name>.py`` and must expose
    ``profile`` containing an :class:`AgentProfile`.
    """
    if name == "default":
        return default_profile()
    selected: Path | None = None
    for directory in xg_directories(cwd):
        path = directory / "agents" / f"{name}.py"
        if path.is_file():
            selected = path
    if selected is None:
        raise RuntimeError(f"agent profile not found: {name}")
    module_name = f"xg_agent_profile_{name}"
    spec = importlib.util.spec_from_file_location(module_name, selected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agent profile: {selected}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = getattr(module, "profile", None)
    if isinstance(profile, AgentProfile):
        return profile
    if callable(profile) and getattr(profile, "__xg_profile__", False):
        base = empty_profile(name)
        return AgentProfile(
            name=name,
            context=base.context,
            system_prompt=base.system_prompt,
            program=profile,
        )
    raise RuntimeError(f"profile must expose a decorated profile function or AgentProfile: {selected}")
