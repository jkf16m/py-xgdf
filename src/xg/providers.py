"""Provider registry: OpenAI-compatible endpoints and API-key resolution.

Every supported provider is reached through its OpenAI-compatible
chat-completions endpoint, so one streaming request builder serves all.

API keys resolve in this order:

1. ``providers.<name>.api_key`` from config — which may be:

   - a bash command in ``$(...)`` form, e.g. ``$(pass show openai/xg)``
   - ``env:NAME`` — an environment variable reference
   - a literal key (not recommended)

2. the provider's standard environment variable (``OPENAI_API_KEY``, ...)
3. the provider's built-in default command, where one exists

Bash commands run once per process per provider; the resolved key is cached.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    host: str
    path: str
    env_var: str
    default_command: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()


REGISTRY: dict[str, Provider] = {
    "openrouter": Provider(
        name="openrouter",
        host="openrouter.ai",
        path="/api/v1/chat/completions",
        env_var="OPENROUTER_API_KEY",
        default_command="pass show pi/openrouter",
    ),
    "openai": Provider(
        name="openai",
        host="api.openai.com",
        path="/v1/chat/completions",
        env_var="OPENAI_API_KEY",
    ),
    "anthropic": Provider(
        name="anthropic",
        host="api.anthropic.com",
        path="/v1/chat/completions",  # their OpenAI-compatible endpoint
        env_var="ANTHROPIC_API_KEY",
        extra_headers=(("anthropic-version", "2023-06-01"),),
    ),
    "xai": Provider(
        name="xai",
        host="api.x.ai",
        path="/v1/chat/completions",
        env_var="XAI_API_KEY",
    ),
    "gemini": Provider(
        name="gemini",
        host="generativelanguage.googleapis.com",
        path="/v1beta/openai/chat/completions",
        env_var="GEMINI_API_KEY",
    ),
}

DEFAULT_PROVIDER = "openrouter"

_resolved: dict[str, str] = {}


def _run_command(command: str) -> str:
    """Run a key-sourcing command via bash and return its first stdout line."""
    try:
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"could not run API-key command {command!r}: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"API-key command failed ({result.returncode}): {command!r}: "
            f"{result.stderr.strip()[:200]}"
        )
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"API-key command returned nothing: {command!r}")
    return stdout.splitlines()[0].strip()


def _resolve_raw(value: str) -> str:
    """Resolve one api_key value: $(command), env:NAME, or literal."""
    value = value.strip()
    if value.startswith("$(") and value.endswith(")"):
        return _run_command(value[2:-1])
    if value.startswith("env:"):
        name = value[4:]
        key = os.environ.get(name)
        if not key:
            raise RuntimeError(f"environment variable {name!r} is not set (api_key: env:{name})")
        return key.strip()
    return value


def resolve_key(provider: Provider, override: str | None = None) -> str:
    """Return the API key for a provider, resolving and caching as needed."""
    cache_key = f"{provider.name}:{override!r}"
    if cache_key in _resolved:
        return _resolved[cache_key]
    if override is not None:
        key = _resolve_raw(override)
    elif os.environ.get(provider.env_var):
        key = os.environ[provider.env_var].strip()
    elif provider.default_command:
        key = _run_command(provider.default_command)
    else:
        raise RuntimeError(
            f"no API key for provider {provider.name!r}: set providers.{provider.name}"
            f".api_key in config (e.g. \"$(pass show ...)\" or \"env:{provider.env_var}\")"
        )
    if not key:
        raise RuntimeError(f"provider {provider.name!r} resolved an empty API key")
    _resolved[cache_key] = key
    return key


def for_model(model: str, default_provider: str = DEFAULT_PROVIDER) -> tuple[Provider, str]:
    """Split ``provider/model-id`` into (provider, model-id).

    ``@preset/...`` models belong to openrouter; an unknown prefix is left
    in the model id and the default provider is used.
    """
    if "/" in model:
        head, rest = model.split("/", 1)
        if head in REGISTRY:
            return REGISTRY[head], rest
    provider = REGISTRY.get(default_provider) or REGISTRY[DEFAULT_PROVIDER]
    return provider, model
