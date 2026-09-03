# xg — Generative Development Framework

A deterministic coding agent driven by Python workflows: every edit is a
small, human-approved tool call, every run is a scripted `run(cfg)` function,
and every session is a resumable event log. Constrained file tools, a Python
tool SDK, no hidden shell access.

- **PyPI:** `py-xgdf`
- **Import:** `xg`
- **CLI:** `xg`
- **Requires:** Python 3.11+, Linux or macOS (uses `pty`/`termios`)
- **License:** MIT

## Installation

`xg` is deliberately scoped to the Python environment where it is installed —
it is not available globally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .          # development install of this repo
# or:                     pip install py-xgdf   (once published to PyPI)
```

## Quick start

```bash
xg init               # scaffold .xg/ (config, workflows, session store)
xg                    # interactive: type your request, agent edits files
```

## Security model

* The default workflow performs a deterministic **workspace read**: every text file is injected as one simulated `read` tool call (with a deterministic `XG_{ID}` tool-call id) plus its tool result, exactly once per session window
* The default workflow reads your **entire workspace** (text files, oldest
  first, gitignored paths excluded) and sends it to the configured model
  provider. Do not run xg on workspaces you would not share with that
  provider.
* The agent can only edit files via a constrained tool state machine —
  every edit is shown as a git patch and needs your explicit approval.
  `new`/`delete` and the `cmd` workflow are also human-confirmed.
* There is no shell tool for the agent in the default workflow; `cfg.shell()`
  is a trusted workflow-author step only. The `xg-cmd` workflow runs shell
  commands with your confirmation.
* API keys never enter the workspace context; they resolve from config,
  environment variables, or a command such as `pass show`.

## Usage

```bash
xg                     # xg-default workflow, interactive request
xg PROMPT              # xg-default with a pre-loaded request
xg -w                  # list all available workflows
xg -w xg-default       # explicit default workflow (also: default)
xg -w xg-cmd PROMPT    # shell-command workflow (also: cmd)
xg init                # scaffold .xg/ in the current project
xg --resume [PATH] [SESSION]
                       # replay a recorded session until its current state
xg --help              # full CLI reference
```

Workflows are selected with `-w`; `xg -w` or `xg --workflow` lists all
available workflows. There are no `workflow`, `cmd`, `run`, or `pty` CLI
commands. Packaged workflows use canonical `xg-*` names: `xg-default`
(alias `default`) and `xg-cmd` (alias `cmd`). Both are ordinary workflow
implementations: `xg-cmd` uses `cfg.agent()` with its restricted `cmd` tool,
so prompts, model replies, tool calls, command results, and resume behavior
use the same session machinery as `xg-default`. Additional workflows can be
placed in `.xg/workflows/<name>.py`.

Run `xg --help` for every option, including resume combinations.

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

The workflow runtime reads `.xg/` from the current directory, `~/.xg/`, and
all ancestors of `.xg/` (least specific first); `.xg/` in the working
directory wins. Inside `.xg/`:

* `workflows/<name>.py` — Python workflows exposing `run(cfg)`
* `sessions/` — JSONL session logs and the `.last` pointer
* `config.json` — configuration (see Configuration below)
* `.venv/` — the private environment created by `xg init`

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

Injection is location-dependent and happens at runtime, before any
workflow code runs: running inside `.xg/workflows` injects the full
workflow-authoring reference (tool state machine, `run(cfg)` API);
anywhere else in a workspace injects documentation about `config.json`
composition plus a superficial workflows overview. The reference lands
in every session window (once, deduped) and is available as
`cfg.documentation`.

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

## Resuming sessions

A session is an event log. While a workflow runs, the runtime records two
kinds of events into the JSONL: every `cfg.prompt()` answer and a completion
marker for every finished agent turn. Metadata lines (`xgdf-*`) are invisible
to the model — they never enter a request body.

Because workflows are deterministic, resuming means **replaying**:

```
xg --resume                        # last session of the current path
xg --resume mysession              # a named session in the current path
xg --resume ~/other/project        # that path's last session
xg --resume ~/other/project fix    # a named session in another path
xg --resume mysession -w repair    # combine with a named workflow
```

Every path keeps its own sessions: sessions live in `<path>/.xg/sessions/`,
and each workspace records a `.last` pointer (updated whenever a named
session is used). Resuming another path runs the workflow with that path as
the runtime root — same history, same workspace.

The workflow re-executes; each `prompt()` and `agent()` call consumes the
next recorded event instead of asking you or paying for inference. When the
cursor reaches the session's current state — or the recording diverges (you
changed the workflow) — everything goes live again. Interrupted turns have
no completion marker, so they simply re-run.

Programmatically: `AgentConfig(resume=True, session=Session(name=...))`.

## Configuration

`xg` composes `config.json` from `~/.xg/` and every `<ancestor>/.xg/`
up to the working directory, least specific first. A layer only sets the
keys it declares; dicts merge recursively, scalars and lists replace.

```json
{
  "model": "anthropic/claude-sonnet-4-5",
  "default_provider": "openrouter",
  "diff_formatter": "delta",
  "providers": {
    "anthropic": {"api_key": "$(pass show anthropic/xg)"}
  }
}
```

`diff_formatter` (or the `XG_DIFF_FORMATTER` env var) names a shell command
that receives proposed tool-call patches on stdin — e.g. `delta` or
`diffr` — and prints them with nicer colors. Without it, xg applies simple
built-in ANSI coloring.

## Providers

Models are `provider/model-id`. Built-in providers (all used via their
OpenAI-compatible chat-completions endpoints): `openrouter`, `openai`,
`anthropic`, `xai`, `gemini`. A model without a known prefix goes to
`default_provider` (default: `openrouter`). Example:

```bash
export ANTHROPIC_API_KEY=...      # or configure via config.json
xg "explain the structure of this repo"
```

API keys resolve in order:

1. `providers.<name>.api_key` from config:
   - `$(command …)` — run via bash, first stdout line is the key
     (e.g. `"$(pass show anthropic/xg)"`)
   - `env:NAME` — read environment variable `NAME`
   - any other value — literal key
2. the provider's environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …)
3. the provider's built-in default (openrouter only: `pass show pi/openrouter`;
   this is a developer convenience and can be overridden by config).

## Layout

```
src/xg/        import name: xg (CLI in cli.py, SDK in sdk.py)
examples/        example workspaces
```

`src/xg/` modules: `cli.py` (entry point), `workflows.py` (Session,
AgentConfig, runtime), `agent.py` (agent loop), `tools.py` (tool state
machine), `sdk.py` (tool/profile SDK), `providers.py` + `llm.py` (model
providers and streaming chat), `config.py` (layered config), `init.py`
(`xg init`).

## Contributing

Issues and pull requests are welcome. Please keep changes consistent with
the deterministic-workflow philosophy: no hidden side effects, tool calls
remain human-approved.

## License

MIT — see `LICENSE` in the repository root.

## Names

| Kind | Name |
|---|---|
| Distribution (PyPI) | `py-xgdf` |
| Import (library) | `xg` |
| CLI | `xg` |
