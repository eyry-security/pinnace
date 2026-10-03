"""Context compaction: keep long runs inside the window.

Token counts here are rough (~4 chars/token). That's fine — compaction just
needs a tripwire, not an audit.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

CHARS_PER_TOKEN = 4

SUMMARIZE_PROMPT = """Summarize this agent conversation so far. Keep everything the next agent
needs to continue the work without re-doing it:

- the original goal and any constraints
- what was tried, in order, and what each attempt produced
- files created or changed, and where
- open questions and the current plan

Be dense. No preamble, no advice, just the facts."""


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate over message contents and tool calls."""
    chars = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    chars += len(block["text"])
        for call in getattr(m, "tool_calls", None) or []:
            chars += len(str(call.get("args", "")))
            chars += len(call.get("name", ""))
    return chars // CHARS_PER_TOKEN


def needs_compaction(messages: list[BaseMessage], max_tokens: int) -> bool:
    return estimate_tokens(messages) >= max_tokens


def compact_messages(model, messages: list[BaseMessage], keep_last: int = 8) -> list[BaseMessage]:
    """Summarize everything but the tail into one system message.

    The original system prompt (messages[0], if it's a SystemMessage) is kept
    verbatim; the summary slots in right after it, then the recent tail.
    """
    if len(messages) <= keep_last + 1:
        return messages
    head = messages[0] if isinstance(messages[0], SystemMessage) else None
    body_start = 1 if head else 0
    to_summarize = messages[body_start:-keep_last] if keep_last else messages[body_start:]
    tail = messages[-keep_last:] if keep_last else []

    summary = model.invoke(
        [SystemMessage(content=SUMMARIZE_PROMPT), *to_summarize]
    ).content
    if not isinstance(summary, str):
        summary = str(summary)

    out: list[BaseMessage] = []
    if head:
        out.append(head)
    out.append(SystemMessage(
        content="Conversation so far (compacted — details summarized, tail kept verbatim):\n" + summary
    ))
    out.extend(tail)
    return out
