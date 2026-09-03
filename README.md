# xg — Generative Development Framework

A deterministic coding agent with an interactive PTY shell and a Python
tool SDK.

## Installation (inside your development environment)

`xg` is deliberately scoped to the Python environment where it is
installed — it is not available globally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .          # development install of this repo
# or, once published:     pip install py-xgdf
```

## Usage

```bash
xg --pty         # interactive shell behind a pseudo-terminal
xg PROMPT        # create a new inactive coding-agent session
xg c [TEXT]      # append input to the latest inactive session
xg r | xg run    # execute the latest inactive session
xg -w            # list workflows; xg -w NAME runs one
```

## Workflows

The agent's behavior lives in `.xg/workflows/<name>.py`, exposing
`run(gdev)` — ordinary Python that decides where inference happens:

```python
import xg.workflows as wf

def run(gdev):
    cfg = wf.AgentConfig()
    session = cfg.get_session()               # disk-backed context window
    if gdev.prompt("describe the change", session=session):
        gdev.agent(gdev.request, config=cfg)  # one inference turn
    return 0
```

## Configuration

`xg` composes `config.json` from `~/.xg/` and every `<ancestor>/.xg/`
up to the working directory, least specific first. A layer only sets the
keys it declares; dicts merge recursively, scalars and lists replace.

```json
{
  "model": "anthropic/claude-sonnet-4-5",
  "default_provider": "openrouter",
  "providers": {
    "anthropic": {"api_key": "$(pass show anthropic/xg)"}
  }
}
```

## Providers

Models are `provider/model-id`. Built-in providers (all used via their
OpenAI-compatible chat-completions endpoints): `openrouter`, `openai`,
`anthropic`, `xai`, `gemini`. A model without a known prefix goes to
`default_provider` (default: `openrouter`).

API keys resolve in order:

1. `providers.<name>.api_key` from config:
   - `$(command …)` — run via bash, first stdout line is the key
     (e.g. `"$(pass show openai/xg)"`)
   - `env:NAME` — read environment variable `NAME`
   - any other value — literal key
2. the provider's environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …)
3. the provider's built-in default (openrouter: `pass show pi/openrouter`)

## Layout

```
src/xg/        import name: xg (CLI in cli.py, SDK in sdk.py)
examples/        example workspaces
```

## Names

| Kind | Name |
|---|---|
| Distribution (PyPI) | `py-xgdf` |
| Import (library) | `xg` |
| CLI | `xg` |
