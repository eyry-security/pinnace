"""Configuration values for constructing a Pinnace agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool

from .sandbox import Sandbox
from .session import SessionStore

DEFAULT_MODEL = "anthropic:claude-opus-4-6"
DEFAULT_SYSTEM = """You are Pinnace, an autonomous agent running inside a sandbox.
Work the task step by step. Prefer running commands and reading files over guessing.
When the job is done, call finish() with a JSON summary of the result instead of
just stopping. Keep tool output in mind: it's truncated at 200k chars, so page
through big outputs rather than dumping them whole."""


@dataclass
class AgentConfig:
    """Reusable values accepted by :class:`~pinnace.PinnaceAgent`.

    ``model`` may be a ``provider:model`` string or an already-built LangChain
    chat model. Call :meth:`resolve` to apply ``PINNACE_MODEL`` and the default
    model before inspecting the configuration; direct agent construction keeps
    its existing lazy resolution behavior.
    """

    model: Any = None
    sandbox: Sandbox | None = None
    tools: list[BaseTool] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM
    max_turns: int = 30
    compaction_tokens: int = 100_000
    compaction_keep_last: int = 8
    session_store: SessionStore | None = None
    session_id: str | None = None
    log: Callable[[str], None] | None = None
    prompt_caching: bool = True

    @classmethod
    def resolve(cls, model: Any = None, **values: Any) -> "AgentConfig":
        """Build a config, resolving an omitted model from the environment."""
        resolved_model = (
            os.environ.get("PINNACE_MODEL", DEFAULT_MODEL)
            if model is None else model
        )
        return cls(model=resolved_model, **values)

    def to_kwargs(self) -> dict[str, Any]:
        """Return an isolated argument mapping for ``PinnaceAgent``."""
        return {
            "model": self.model,
            "sandbox": self.sandbox,
            "tools": list(self.tools),
            "system_prompt": self.system_prompt,
            "max_turns": self.max_turns,
            "compaction_tokens": self.compaction_tokens,
            "compaction_keep_last": self.compaction_keep_last,
            "session_store": self.session_store,
            "session_id": self.session_id,
            "prompt_caching": self.prompt_caching,
            "log": self.log,
        }
