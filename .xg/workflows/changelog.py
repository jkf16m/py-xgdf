"""Summarize the latest commits into a changelog blurb."""

import subprocess


def run(cfg):
    log = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        capture_output=True, text=True, check=True, cwd=cfg.root,
    ).stdout
    session = cfg.get_session()
    if cfg.prompt("what should the changelog cover"):
        cfg.agent(f"Summarize these commits into 3 bullet points:\n{log}")
    return 0
