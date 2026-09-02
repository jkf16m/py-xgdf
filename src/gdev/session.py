"""Persistent, pre-run request sessions.

Public API:
    SessionStore(root)
    SessionStore.create(prompt) -> Session
    SessionStore.continue_latest(text) -> Session
    SessionStore.latest_open() -> Session | None
    SessionStore.run_latest() -> Session
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Session:
    """A request that may collect input until explicitly run."""

    id: str
    prompt: str
    status: str = "open"  # open, running, closed
    activity: bool = False


class SessionStore:
    """Store request sessions in the workspace's ignored .gdev directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.directory = self.root / ".gdev" / "sessions"
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, prompt: str) -> Session:
        """Create and persist a fresh open session."""
        session = Session(f"{time.time_ns()}-{uuid.uuid4().hex[:8]}", prompt)
        self._save(session)
        return session

    def latest_open(self) -> Session | None:
        """Return the newest session that has not started activity."""
        for path in sorted(self.directory.glob("*.json"), reverse=True):
            session = self._load(path)
            if session and session.status == "open" and not session.activity:
                return session
        return None

    def continue_latest(self, text: str) -> Session:
        """Append input to the latest inactive session, or fail clearly."""
        session = self.latest_open()
        if session is None:
            raise RuntimeError("no inactive session to continue; start a new session with gdev")
        if text:
            session.prompt = f"{session.prompt}\n\n{text}" if session.prompt else text
            self._save(session)
        return session

    def begin(self, session: Session) -> Session:
        """Mark a session active before any model or tool activity occurs."""
        session.status = "running"
        session.activity = True
        self._save(session)
        return session

    def close(self, session: Session) -> Session:
        """Close a completed session permanently against resume."""
        session.status = "closed"
        self._save(session)
        return session

    def _save(self, session: Session) -> None:
        """Persist one session atomically enough for CLI-sized writes."""
        path = self.directory / f"{session.id}.json"
        path.write_text(json.dumps(session.__dict__, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _load(path: Path) -> Session | None:
        """Load one session file, ignoring malformed files."""
        try:
            data = json.loads(path.read_text())
            return Session(**data)
        except (OSError, ValueError, TypeError):
            return None
