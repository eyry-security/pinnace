"""File-based session store. Transcripts persist as JSONL so a run can be
resumed later — by you, by the CLI, or by Quarterdeck.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.load import dumpd, loads
from langchain_core.messages import BaseMessage

_ALLOWED = "messages"  # loads() is beta; restrict to chat messages


class SessionStore:
    """Sessions live under <root>/<session_id>/transcript.jsonl."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        base = root or os.environ.get("PINNACE_HOME", "~/.pinnace")
        self.root = Path(base).expanduser() / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session_id)
        if not safe:
            raise ValueError("empty session id")
        return self.root / safe / "transcript.jsonl"

    def save(self, session_id: str, messages: list[BaseMessage]) -> Path:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(dumpd(m)) + "\n")
        return path

    def load(self, session_id: str) -> list[BaseMessage] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        messages: list[BaseMessage] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(loads(line, allowed_objects=_ALLOWED))
        return messages

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True
