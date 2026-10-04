"""Compaction controls: transparent, configurable context compaction."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, SystemMessage

from .compaction import (
    SUMMARIZE_PROMPT,
    compact_messages,
    estimate_tokens,
    needs_compaction,
)

__all__ = [
    "CompactionConfig",
    "CompactionReport",
    "compact_with_config",
    "needs_compaction",
    "estimate_tokens",
]


@dataclass
class CompactionConfig:
    """Knobs for when and how compaction happens."""

    max_tokens: int = 100_000
    """Tripwire: compact when the estimated context reaches this."""
    keep_last: int = 8
    """Recent messages kept verbatim (not summarized)."""
    summarize_prompt: str = SUMMARIZE_PROMPT
    """Instruction given to the summarizer model."""


@dataclass
class CompactionReport:
    """What a compaction pass did — for logs and user visibility."""

    messages_summarized: int
    tokens_before: int
    tokens_after: int
    summary_chars: int

    def one_liner(self) -> str:
        return (
            f"compacted {self.messages_summarized} messages "
            f"({self.tokens_before}→{self.tokens_after} est. tokens)"
        )


def compact_with_config(
    model,
    messages: list[BaseMessage],
    config: CompactionConfig | None = None,
) -> tuple[list[BaseMessage], CompactionReport | None]:
    """Compact like compact_messages, but configurable and reporting.

    Returns (messages, report). Report is None when nothing needed compacting
    (message list returned unchanged).
    """
    cfg = config or CompactionConfig()
    before = estimate_tokens(messages)
    if not needs_compaction(messages, cfg.max_tokens):
        return messages, None

    head = messages[0] if isinstance(messages[0], SystemMessage) else None
    body_start = 1 if head else 0
    keep = cfg.keep_last
    to_summarize = messages[body_start:-keep] if keep else messages[body_start:]
    tail = messages[-keep:] if keep else []

    summary = model.invoke(
        [SystemMessage(content=cfg.summarize_prompt), *to_summarize]
    ).content
    if not isinstance(summary, str):
        summary = str(summary)

    out: list[BaseMessage] = []
    if head:
        out.append(head)
    out.append(
        SystemMessage(
            content=(
                "Conversation so far (compacted — details summarized, "
                "tail kept verbatim):\n" + summary
            )
        )
    )
    out.extend(tail)

    report = CompactionReport(
        messages_summarized=len(to_summarize),
        tokens_before=before,
        tokens_after=estimate_tokens(out),
        summary_chars=len(summary),
    )
    return out, report
