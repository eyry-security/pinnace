"""Agent configuration.

Everything needed to stand up and run an agent: which model to talk to, how
to manage the context window, and which Docker image to sandbox tool execution
inside. Values come from explicit arguments, falling back to environment
variables, falling back to sane defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # LLM
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096

    # Context window management
    max_context_messages: int = 80
    compaction_threshold: int = 60
    compaction_keep_recent: int = 10

    # Sandbox
    sandbox_image: str = "python:3.12-slim"
    sandbox_timeout: int = 120          # seconds per command
    sandbox_mem_limit: str = "512m"
    sandbox_network: bool = False       # network access inside sandbox

    # Agent behaviour
    max_turns: int = 30                 # hard stop to prevent runaway loops
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)  # tool names to enable

    @classmethod
    def resolve(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        sandbox_image: str | None = None,
        max_turns: int | None = None,
        system_prompt: str | None = None,
    ) -> "AgentConfig":
        return cls(
            model=model or os.environ.get("PINNACE_MODEL", "gpt-4o"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", ""),
            sandbox_image=sandbox_image or os.environ.get("PINNACE_SANDBOX_IMAGE", "python:3.12-slim"),
            max_turns=max_turns or int(os.environ.get("PINNACE_MAX_TURNS", "30")),
            system_prompt=system_prompt or "",
        )
