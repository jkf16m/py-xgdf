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
`run(cfg)`. `cfg` is an `AgentConfig` handed to it either by the xg runner
(`xg -w <name>`) or by a parent workflow — it carries the inference settings
(model, session, tools) **and** the runtime primitives:

```python
def run(cfg):
    session = cfg.get_session()             # writable context window
    if cfg.prompt("describe the change"):
        cfg.agent(cfg.request)              # one inference turn
        cfg.workflow("changelog")           # reuse another workflow
    return 0
```

A child workflow receives the parent's cfg — same model, session, and
tools — so workflows compose like function calls. Built-in workflows
(`default`, …) are usable the same way, and from the SDK:

```python
from xg import run_workflow, AgentConfig

run_workflow("default")                     # fresh runtime
cfg = AgentConfig(model="anthropic/claude-sonnet-4-5")
run_workflow("my-flow", cfg=cfg)            # your settings
```

## Injected documentation

Executing inside a workspace, the framework reference (`xg.docs`) is
automatically appended to the system message of every agent turn and is
available as `cfg.documentation` for custom prompts.

## Session branches

Sessions clone like branches: `cfg.fork()` derives a config whose session
is an independent copy of the current window — model, tools, and runtime
are shared, but forked steps never write to the original:

```python
branch = cfg.fork(name="experiment")
branch.agent("try the risky refactor")     # its own window copy
branch.session.delete()                     # discard the branch
```

`Session.clone(name=None)` (auto-names `<original>-clone-<suffix>`) and
`Session.delete()` work on sessions directly too.

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
