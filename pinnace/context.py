"""Public message type used by transcripts, compaction, and sessions."""

from __future__ import annotations

from typing import TypeAlias

from langchain_core.messages import BaseMessage

# Pinnace deliberately keeps LangChain messages as its runtime representation.
# This preserves normalized tool calls, structured content, and SessionStore's
# dump/load round trip while giving downstream code one stable type to import.
Message: TypeAlias = BaseMessage

__all__ = ["Message"]
