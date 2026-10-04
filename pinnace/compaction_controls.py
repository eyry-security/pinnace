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

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self.keep_last < 0:
            raise ValueError("keep_last must be non-negative")


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

    def to_dict(self) -> dict[str, int]:
        return {
            "messages_summarized": self.messages_summarized,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "summary_chars": self.summary_chars,
        }


def compact_with_config(
    model,
    messages: list[BaseMessage],
    config: CompactionConfig | None = None,
    *,
    force: bool = False,
    on_response=None,
) -> tuple[list[BaseMessage], CompactionReport | None]:
    """Compact context with configurable controls and a transparent report.

    ``force=True`` bypasses the token tripwire for an explicit manual trigger.
    The operation remains a no-op when no messages precede the retained tail.
    ``on_response`` observes the raw summarizer response (for usage metering).
    """
    cfg = config or CompactionConfig()
    before = estimate_tokens(messages)
    if not messages or (not force and not needs_compaction(messages, cfg.max_tokens)):
        return messages, None

    head = messages[0] if isinstance(messages[0], SystemMessage) else None
    body_start = 1 if head else 0
    keep = cfg.keep_last
    to_summarize = messages[body_start:-keep] if keep else messages[body_start:]
    if not to_summarize:
        return messages, None
    tail = messages[-keep:] if keep else []

    response = model.invoke(
        [SystemMessage(content=cfg.summarize_prompt), *to_summarize]
    )
    if on_response is not None:
        on_response(response)
    summary = response.content
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
