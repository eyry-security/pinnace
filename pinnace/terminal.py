"""TTY-aware rendering for Pinnace CLI narration.

The agent's programmatic ``log=`` contract stays plain text.  The CLI routes
those messages through :class:`TerminalRenderer` to add lightweight progress
markers and ANSI color without introducing a terminal UI dependency.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

_RESET = "\033[0m"
_STYLES = {
    "turn": "\033[1;36m",
    "tool": "\033[33m",
    "compact": "\033[35m",
    "success": "\033[1;32m",
    "error": "\033[1;31m",
    "session": "\033[34m",
    "info": "\033[2m",
}


class TerminalRenderer:
    """Render one narration event per line to a terminal stream.

    ANSI color is enabled only for TTY streams and is disabled when
    ``NO_COLOR`` is present. Progress symbols remain in redirected output so
    logs are still easy to scan without escape sequences.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    @property
    def color_enabled(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty() and os.environ.get("NO_COLOR") is None)

    def render(self, message: str) -> str:
        """Return a formatted narration line without its trailing newline."""
        body = str(message)
        if body.startswith("[pinnace] "):
            body = body[len("[pinnace] "):]

        symbol, style = self._appearance(body)
        marker = symbol
        if self.color_enabled:
            marker = f"{_STYLES[style]}{symbol}{_RESET}"

        # Keep multiline tool arguments and errors visually attached to the
        # event marker instead of letting continuation lines look unrelated.
        body = body.replace("\n", "\n  ")
        return f"{marker} {body}"

    def __call__(self, message: str) -> None:
        self.stream.write(self.render(message) + "\n")
        self.stream.flush()

    @staticmethod
    def _appearance(body: str) -> tuple[str, str]:
        lowered = body.lower()
        if lowered.startswith("turn "):
            return "◆", "turn"
        if lowered.startswith("tool:"):
            return "→", "tool"
        if lowered.startswith("compacted context"):
            return "↻", "compact"
        if lowered.startswith(("finish()", "no tool calls", "done in")):
            return "✓", "success"
        if lowered.startswith(("error:", "hit max_turns")):
            return "✗", "error"
        if lowered.startswith(("resumed session", "session saved")):
            return "●", "session"
        return "•", "info"
