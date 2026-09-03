"""Streaming OpenAI-compatible chat client with terminal presentation.

Public API:
    chat(messages, tools=None) -> dict
"""

import http.client
import json
import os
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from xg.config import load
from xg.providers import for_model, resolve_key


def _resolve_model_and_provider(model: str | None):
    """Resolve the model string and provider from args, env, and config."""
    config = load() if os.environ.get("XG_NO_CONFIG") != "1" else {}
    if not model:
        model = config.get("model") or os.environ.get("XG_MODEL", "@preset/mimo")
    provider, model_id = for_model(model, config.get("default_provider", "openrouter"))
    overrides = config.get("providers", {}).get(provider.name, {})
    return provider, model_id, overrides.get("api_key")


class _MarkdownStream:
    """Continuously re-render partial Markdown with Rich."""

    def __init__(self):
        self.text = ""
        self.last_arrival = time.monotonic()
        self.live = Live(Markdown(""), console=Console(), refresh_per_second=30, transient=False)
        self.live.start()

    def feed(self, chunk: str) -> None:
        """Add a response chunk and refresh the Markdown rendering."""
        now = time.monotonic()
        interval = now - self.last_arrival
        self.last_arrival = now
        delay = min(0.025, max(0.001, interval / max(len(chunk), 1)))
        for char in chunk:
            self.text += char
            self.live.update(Markdown(self.text), refresh=True)
            time.sleep(delay)

    def close(self) -> None:
        """Finish the live display and leave the final rendered Markdown."""
        self.live.update(Markdown(self.text), refresh=True)
        self.live.stop()


def _request_chunks(source, model: str, tools):
    """Yield the JSON request body one chunk at a time.

    ``source`` may be any iterable of message dicts — including a Session,
    whose iterator walks its JSONL file on disk. Nothing beyond one message
    is ever materialized here.
    """
    yield b'{"model":' + json.dumps(model).encode() + b',"messages":['
    separator = b""
    for message in source:
        yield separator + json.dumps(message, ensure_ascii=False).encode()
        separator = b","
    yield b"]"
    if tools:
        yield b',"tools":' + json.dumps(tools, ensure_ascii=False).encode()
    yield b',"stream":true}'


def chat(messages, tools=None, model: str | None = None) -> dict:
    """Stream one chat request and return its reconstructed assistant message.

    ``messages`` may be a plain list or any iterable of message dicts (e.g. a
    session backed by a file); the request body is generated and sent with
    chunked transfer encoding, so the conversation is never materialized.
    """
    provider, model_id, api_key_override = _resolve_model_and_provider(model)
    connection = http.client.HTTPSConnection(provider.host, timeout=300)
    try:
        connection.request(
            "POST", provider.path,
            body=_request_chunks(messages, model_id, tools),
            headers={
                "Authorization": f"Bearer {resolve_key(provider, api_key_override)}",
                "Content-Type": "application/json",
                **dict(provider.extra_headers),
            },
            encode_chunked=True,
        )
        response = connection.getresponse()
        if response.status != 200:
            error = response.read().decode(errors="replace")
            raise RuntimeError(f"chat request failed (HTTP {response.status}): {error[:300]}")
        printer = _MarkdownStream()
        content, calls = [], {}
        try:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
                text = delta.get("content") or ""
                if text:
                    content.append(text)
                    printer.feed(text)
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    item = calls.setdefault(index, {"id": call.get("id", ""), "type": "function", "function": {"name": "", "arguments": ""}})
                    item["id"] = item["id"] or call.get("id", "")
                    function = call.get("function") or {}
                    item["function"]["name"] += function.get("name", "")
                    item["function"]["arguments"] += function.get("arguments", "")
        finally:
            printer.close()
    finally:
        connection.close()
    if content:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return {"role": "assistant", "content": "".join(content), "tool_calls": list(calls.values())}
