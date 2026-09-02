# gdev — Generative Development Framework

A deterministic coding agent with an interactive PTY shell and a Python
tool SDK. Renamed and restructured from the `gq` prototype.

## Installation (inside your development environment)

`gdev` is deliberately scoped to the Python environment where it is
installed — it is not available globally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .          # development install of this repo
# or, once published:     pip install pygdev
```

## Usage

```bash
gdev --pty         # interactive shell behind a pseudo-terminal
gdev PROMPT        # create a new inactive coding-agent session
gdev c [TEXT]      # append input to the latest inactive session
gdev r | gdev run  # execute the latest inactive session
gd ...             # short alias for gdev
```

## SDK (library use)

The same install exposes the SDK for import:

```python
from gdev import Agent, Tool, ToolRegistry
```

## Layout

```
src/gdev/        import name: gdev (CLI in cli.py, SDK in sdk.py)
examples/        example workspaces
```

## Names

| Kind | Name |
|---|---|
| Distribution (PyPI) | `pygdev` |
| Import (library) | `gdev` |
| CLI | `gdev`, alias `gd` |
