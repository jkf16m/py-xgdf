"""Injected xgdf documentation — location-dependent.

The runtime computes what to inject from *where execution happens*, before
any workflow runs (this is part of the xgdf runtime, not workflow code):

* cwd inside ``.xg/workflows``  → the full workflow-authoring reference
  (the ``run(cfg)`` API, the tool state machine, the no-shell rule).
* anywhere else in a workspace  → documentation about ``config.json``
  composition (layers, providers, API keys) plus a superficial overview of
  workflows.

The chosen reference lands in every session window created by the runtime
(once per window, deduped), and is available as ``cfg.documentation``.
"""

from __future__ import annotations

from pathlib import Path

WORKFLOW_REFERENCE = """\
# xgdf reference — workflows

You are running as the agent step of an xgdf workflow. You are NOT a general
shell agent: your only capabilities are the tools listed below. There is no
bash/tool-execution tool available to you, ever.

File tools (enforced as a state machine):
* `select(path)` — open a file for editing. Selecting enables `edit`/`close`.
* `edit(old_text, new_text)` — exact-match replacement in the selected file.
* `close()` — release the selected file.
* `new(name, content)` — create one file (single operation).
* `delete(path)` — remove a file (human confirm required).

Rules of the environment:
* One tool call per turn; the human accepts or rejects each call.
* When a file is selected, `close` is always allowed (safety valve).
* Tool calls that violate the state machine are rejected before execution.

Workflows around you are deterministic Python (`run(cfg)`); inference is a
deliberate step. Be concise and prefer small, reviewable operations.\
"""

WORKFLOW_OVERVIEW = """\
Workflows are Python files under `.xg/workflows/<name>.py` exposing
`run(cfg)`; the xg runner (or a parent workflow) passes the cfg. List them
with `xg -w`, run one with `xg -w NAME`.\
"""

CONFIG_REFERENCE = """\
# xgdf reference — configuration

`config.json` composes by location, least specific first:
* `~/.xg/config.json` — user-level defaults
* `<any ancestor>/.xg/config.json` — intermediate layers
* `.xg/config.json` (cwd) — project settings, wins

A layer only sets the keys it declares; dicts merge recursively, scalars and
lists replace. Relevant keys:
* `model` — e.g. "openrouter/@preset/mimo" or "anthropic/claude-..."
* `default_provider` — when a model has no known provider prefix
* `providers.<name>.api_key` — a bash command "$(pass show ...)", an
  "env:VAR" reference, or a literal key

{overview}\
"""


def documentation_for(path: str | Path) -> str:
    """Return the reference matching where execution happens."""
    resolved = Path(path).resolve()
    parts = resolved.parts
    if parts[-1] == "workflows" and ".xg" in parts:
        return WORKFLOW_REFERENCE
    return CONFIG_REFERENCE.format(overview=WORKFLOW_OVERVIEW)


def documentation() -> str:
    """Back-compat alias: general (workspace-level) documentation."""
    return CONFIG_REFERENCE.format(overview=WORKFLOW_OVERVIEW)
