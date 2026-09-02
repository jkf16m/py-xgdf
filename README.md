# xg — Generative Development Framework

A deterministic coding agent with an interactive PTY shell and a Python
tool SDK. Renamed and restructured from the `gq` prototype.

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
xg r | xg run  # execute the latest inactive session
gd ...             # short alias for xg
```

## SDK (library use)

The same install exposes the SDK for import:

```python
from xg import Agent, Tool, ToolRegistry
```

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
| CLI | `xg`, alias `gd` |
