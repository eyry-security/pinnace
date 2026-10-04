"""The agent loop. Hand it a prompt or a seed and it runs: multi-turn,
tool-calling, context-compacting, sandboxed.

This is deliberately a hand-rolled loop, not the prebuilt ReAct agent — the
compaction hook and the finish() convention need control over every turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from .compaction import compact_messages, estimate_tokens, needs_compaction
from .config import AgentConfig, DEFAULT_MODEL, DEFAULT_SYSTEM
from .sandbox import DockerSandbox, Sandbox, SandboxError
from .session import SessionStore
from .tools import FINISH_PREFIX, builtin_tools, parse_finish


class PinnaceError(RuntimeError):
    """Anything the agent runtime itself gets wrong."""


def _make_model(model_ref: str):
    """Build a chat model from a "provider:model" ref via init_chat_model."""
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:
        raise PinnaceError(
            "building a chat model needs the 'langchain' package plus a provider, "
            "e.g. pip install 'pinnace[anthropic]'. "
            "Or pass an already-built model= to PinnaceAgent."
        ) from e
    provider, _, name = model_ref.partition(":")
    if not name:
        raise PinnaceError(
            f"bad model ref {model_ref!r}: want 'provider:model', e.g. 'anthropic:claude-opus-4-6'"
        )
    return init_chat_model(name, model_provider=provider)


@dataclass
class AgentResult:
    """What a run produced."""

    final: str | None
    """The agent's last message text (None if it ended via finish() with no text)."""
    structured: dict | str | None
    """Payload from finish(), if it called it."""
    turns: int
    compacted: int
    """How many times the context was compacted mid-run."""
    session_id: str | None
    transcript: list[dict] = field(default_factory=list)
    """Serialized messages, oldest first."""

    @property
    def finished(self) -> bool:
        return self.structured is not None

    def to_dict(self) -> dict:
        return {
            "final": self.final,
            "structured": self.structured,
            "turns": self.turns,
            "compacted": self.compacted,
            "session_id": self.session_id,
            "transcript": self.transcript,
        }


class PinnaceAgent:
    """A durable multi-turn agent run.

    Args:
        model: a langchain chat model, or a "provider:model" ref string
            (default: $PINNACE_MODEL or "anthropic:claude-opus-4-6").
        sandbox: where tools execute. Defaults to a DockerSandbox (raises a
            clear error when Docker isn't there).
        tools: extra langchain tools appended after the built-ins.
        system_prompt: prepended to every run.
        max_turns: hard stop on the loop.
        compaction_tokens: rough token tripwire for context compaction.
        compaction_keep_last: recent messages kept verbatim when compacting.
        session_store / session_id: persist and resume transcripts.
        log: callable taking a string; turn-by-turn narration goes here.
    """

    def __init__(
        self,
        model=None,
        sandbox: Sandbox | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM,
        max_turns: int = 30,
        compaction_tokens: int = 100_000,
        compaction_keep_last: int = 8,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        log=None,
    ) -> None:
        if model is None:
            model = AgentConfig.resolve().model
        self.model = _make_model(model) if isinstance(model, str) else model
        if sandbox is None:
            try:
                sandbox = DockerSandbox()
            except SandboxError as e:
                raise PinnaceError(
                    f"{e} Pass sandbox= explicitly (e.g. LocalSandbox for dev)."
                ) from e
        self.sandbox = sandbox
        self.tools = builtin_tools(sandbox) + list(tools or [])
        self._tools_by_name = {t.name: t for t in self.tools}
        self.bound = self.model.bind_tools(self.tools)
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.compaction_tokens = compaction_tokens
        self.compaction_keep_last = compaction_keep_last
        self.session_store = session_store or SessionStore()
        self.session_id = session_id
        self.log = log or (lambda *a: None)

    @classmethod
    def from_config(cls, config: AgentConfig) -> "PinnaceAgent":
        """Construct an agent from a reusable configuration value object."""
        return cls(**config.to_kwargs())

    def _say(self, msg: str) -> None:
        self.log(msg)

    def run(self, prompt: str) -> AgentResult:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        if self.session_id:
            history = self.session_store.load(self.session_id)
            if history:
                # History already carries its own system prompt; don't double it.
                messages = [m for m in history if not isinstance(m, SystemMessage)]
                if self.system_prompt:
                    messages.insert(0, SystemMessage(content=self.system_prompt))
                self._say(f"[pinnace] resumed session {self.session_id!r} ({len(history)} messages)")
        messages.append(HumanMessage(content=prompt))

        turns = 0
        compacted = 0
        structured = None
        final: str | None = None

        while turns < self.max_turns:
            turns += 1
            if needs_compaction(messages, self.compaction_tokens):
                before = estimate_tokens(messages)
                messages = compact_messages(self.model, messages, self.compaction_keep_last)
                compacted += 1
                self._say(f"[pinnace] compacted context ({before}→{estimate_tokens(messages)} est. tokens)")

            self._say(f"[pinnace] turn {turns}/{self.max_turns}, calling model…")
            resp: AIMessage = self.bound.invoke(messages)
            messages.append(resp)

            calls = resp.tool_calls or []
            if not calls:
                final = resp.content if isinstance(resp.content, str) else str(resp.content)
                self._say("[pinnace] no tool calls — done")
                break

            done = False
            for call in calls:
                name = call.get("name", "")
                call_id = call.get("id", "")
                args = call.get("args", {}) or {}
                tool = self._tools_by_name.get(name)
                if tool is None:
                    out = f"error: unknown tool {name!r}"
                else:
                    self._say(f"[pinnace] tool: {name}({str(args)[:120]})")
                    try:
                        out = str(tool.invoke(args))
                    except Exception as e:  # noqa: BLE001 - the model should see failures
                        out = f"error: {e}"
                messages.append(ToolMessage(content=out, tool_call_id=call_id, name=name))
                payload = parse_finish(out)
                if payload is not None:
                    structured = payload
                    done = True
                    self._say("[pinnace] finish() called — done")
                    break
            if done:
                break
        else:
            self._say(f"[pinnace] hit max_turns={self.max_turns}")

        if self.session_id:
            path = self.session_store.save(self.session_id, messages)
            self._say(f"[pinnace] session saved to {path}")

        from langchain_core.load import dumpd

        return AgentResult(
            final=final,
            structured=structured,
            turns=turns,
            compacted=compacted,
            session_id=self.session_id,
            transcript=[dumpd(m) for m in messages],
        )
