"""Streaming OpenAI-compatible chat client with terminal presentation.

Public API:
    chat(messages, tools=None) -> dict
"""

import http.client
import json
import os
import subprocess
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


def _api_key() -> str:
    """Read the OpenRouter key from the configured pass entry."""
    try:
        result = subprocess.run(["pass", "show", "pi/openrouter"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not read API key with `pass show pi/openrouter`") from exc
    key = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not key:
        raise RuntimeError("`pass show pi/openrouter` returned an empty API key")
    return key


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
    import os

    model = model or os.environ.get("GDEV_MODEL", "@preset/mimo")
    connection = http.client.HTTPSConnection("openrouter.ai", timeout=300)
    try:
        connection.request(
            "POST", "/api/v1/chat/completions",
            body=_request_chunks(messages, model, tools),
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
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
