"""Summarize the latest commits into a changelog blurb."""

import subprocess

import xg.workflows as wf


def run(gdev):
    log = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        capture_output=True, text=True, check=True,
    ).stdout
    cfg = wf.AgentConfig()
    session = cfg.get_session()
    if gdev.prompt("what should the changelog cover", session=session):
        gdev.agent(
            f"Summarize these commits into 3 bullet points:\n{log}",
            config=cfg,
        )
    return 0
