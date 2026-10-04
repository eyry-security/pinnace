"""Context window management with compaction.

The agent keeps a list of messages (system, user, assistant, tool). When the
list exceeds ``compaction_threshold``, older messages are compressed into a
single summary message by asking the LLM itself to summarise the conversation
so far. The most recent ``keep_recent`` turns are always preserved verbatim so
the agent doesn't lose its immediate working memory.

This is a simple, effective strategy: summarise the old, keep the new.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """One message in the conversation."""
    role: str                              # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict] | None = None   # assistant messages with tool calls
    tool_call_id: str | None = None        # tool result messages
    name: str | None = None                # tool name on tool result messages

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


class Context:
    """The agent's message history with compaction support.

    Parameters
    ----------
    system_prompt : str
        Always the first message. Survives compaction.
    compaction_threshold : int
        When ``len(messages)`` exceeds this, compact.
    keep_recent : int
        Number of most-recent messages to preserve verbatim during compaction.
    """

    def __init__(
        self,
        system_prompt: str = "",
        compaction_threshold: int = 60,
        keep_recent: int = 10,
    ) -> None:
        self.system_prompt = system_prompt
        self.compaction_threshold = compaction_threshold
        self.keep_recent = keep_recent
        self.messages: list[Message] = []
        self._compaction_count: int = 0

        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def add_user(self, content: str) -> None:
        self.append(Message(role="user", content=content))

    def add_assistant(self, content: str | None = None,
                      tool_calls: list[dict] | None = None) -> None:
        self.append(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.append(Message(role="tool", content=content,
                            tool_call_id=tool_call_id, name=name))

    def needs_compaction(self) -> bool:
        return len(self.messages) > self.compaction_threshold

    def compact(self, summary: str) -> None:
        """Replace old messages with a summary, keeping the system prompt and
        the most recent ``keep_recent`` messages."""
        if len(self.messages) <= self.keep_recent + 1:
            return  # nothing worth compacting

        system = self.messages[0] if self.messages and self.messages[0].role == "system" else None
        recent = self.messages[-self.keep_recent:]

        compacted: list[Message] = []
        if system:
            compacted.append(system)
        compacted.append(Message(
            role="system",
            content=f"[conversation summary — compaction #{self._compaction_count + 1}]\n{summary}",
        ))
        compacted.extend(recent)
        self.messages = compacted
        self._compaction_count += 1

    def compaction_prompt(self) -> list[dict]:
        """Build a one-shot prompt asking the LLM to summarise the conversation.

        We send the messages that are *about to be dropped* and ask for a
        concise summary that preserves key facts, decisions, and any file
        paths or data the agent has been working with.
        """
        # Everything except the system prompt and the recent tail.
        start = 1 if self.messages and self.messages[0].role == "system" else 0
        end = max(start, len(self.messages) - self.keep_recent)
        to_summarise = self.messages[start:end]
        if not to_summarise:
            return []

        text_parts = []
        for m in to_summarise:
            prefix = m.role.upper()
            body = m.content or "(tool call)" if m.tool_calls is None else "(tool call)"
            text_parts.append(f"[{prefix}] {body}")
        transcript = "\n".join(text_parts)

        return [
            {"role": "system", "content": (
                "Summarise the following agent conversation transcript. Preserve: "
                "key facts, file paths, hostnames, decisions made, current task state, "
                "and any data the agent produced. Be concise but complete."
            )},
            {"role": "user", "content": transcript},
        ]

    def to_dicts(self) -> list[dict]:
        """Serialise for the OpenAI chat completions API."""
        return [m.to_dict() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)
