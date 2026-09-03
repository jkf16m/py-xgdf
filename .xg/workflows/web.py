"""Investigate a topic via the DuckDuckGo Instant Answer API."""

import json
import subprocess


def run(cfg):
    topic = cfg.prompt("what topic should the web investigation cover")
    if not topic:
        return 1

    raw = subprocess.run(
        [
            "curl", "-s",
            "https://api.duckduckgo.com/",
            "--get",
            "--data-urlencode", f"q={topic}",
            "--data-urlencode", "format=json",
            "--data-urlencode", "no_html=1",
        ],
        capture_output=True, text=True, check=True,
    ).stdout

    data = json.loads(raw)
    # Keep the payload small and structured for the agent.
    highlights = {
        "query": data.get("Heading") or topic,
        "abstract": data.get("AbstractText") or "",
        "abstract_url": data.get("AbstractURL") or "",
        "related": [
            {"text": r.get("Text"), "url": r.get("FirstURL")}
            for r in (data.get("RelatedTopics") or [])[:10]
        ],
    }

    cfg.agent(
        "Investigate this web research result and give a concise summary "
        f"with key takeaways and source links:\n{json.dumps(highlights, indent=2)}"
    )
    return 0
