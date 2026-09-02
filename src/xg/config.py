"""Deterministic layered discovery of xg configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def xg_directories(cwd: str | Path = ".") -> list[Path]:
    """Return configuration directories from least to most specific."""
    cwd = Path(cwd).resolve()
    candidates: list[Path] = [Path.home() / ".xg"]
    ancestors = list(cwd.parents)[::-1] + [cwd]
    for directory in ancestors:
        candidate = directory / ".xg"
        if candidate not in candidates:
            candidates.append(candidate)
    return [path for path in candidates if path.is_dir()]


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            # Lists and scalar values are replacement values. This keeps
            # precedence deterministic and avoids surprising list unions.
            result[key] = value
    return result


def load(cwd: str | Path = ".") -> dict[str, Any]:
    """Load ``config.json`` from all applicable .xg directories."""
    config: dict[str, Any] = {}
    for directory in xg_directories(cwd):
        path = directory / "config.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid xg configuration: {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"xg configuration must be an object: {path}")
        config = _merge(config, value)
    return config


def load_agent(name: str, cwd: str | Path = ".") -> dict[str, Any]:
    """Load one agent profile with the nearest profile taking precedence."""
    profile: dict[str, Any] = {}
    found = False
    for directory in xg_directories(cwd):
        path = directory / "agents" / f"{name}.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid xg agent profile: {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"xg agent profile must be an object: {path}")
        profile = _merge(profile, value)
        found = True
    if not found:
        raise RuntimeError(f"agent profile not found: {name}")
    return profile
