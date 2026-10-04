"""The agent loop. Hand it a prompt or a seed and it runs: multi-turn,
tool-calling, context-compacting, sandboxed.

This is deliberately a hand-rolled loop, not the prebuilt ReAct agent — the
compaction hook and the finish() convention need control over every turn.
"""

from __future__ import annotations

import os
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
from .usage import CostBasis, UsageMeter, response_model


class PinnaceError(RuntimeError):
    """Anything the agent runtime itself gets wrong."""


THINKING_BUDGET_TOKENS = 4000
"""Extended-thinking token budget wired in when thinking=True."""

THINKING_MAX_TOKENS = 8000
"""max_tokens used with thinking=True; Anthropic requires max_tokens > budget_tokens."""

THINKING_LOG_MAX_CHARS = 2000
"""Cap on thinking text echoed into the run narration per turn."""


def extract_thinking(resp: AIMessage) -> str:
    """Pull thinking/reasoning text out of a langchain model response.

    Handles Anthropic extended-thinking content blocks (dicts or objects),
    redacted-thinking blocks, and reasoning stashed in ``additional_kwargs``
    by other providers. Returns "" when the response carries no thinking.
    """
    parts: list[str] = []
    blocks = resp.content if isinstance(resp.content, list) else []
    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type")
            text = block.get("thinking")
        else:
            btype = getattr(block, "type", None)
            text = getattr(block, "thinking", None)
        if btype == "thinking":
            if isinstance(text, str) and text:
                parts.append(text)
        elif btype == "redacted_thinking":
            parts.append("[redacted thinking]")
    for key in ("reasoning", "reasoning_content", "thinking"):
        extra = resp.additional_kwargs.get(key)
        if isinstance(extra, str) and extra:
            parts.append(extra)
    return "\n".join(parts).strip()


def _make_model(model_ref: str, thinking: bool = False):
    """Build a chat model from a "provider:model" ref via init_chat_model.

    When thinking=True and the provider supports it (currently anthropic),
    the model's extended-thinking mode is enabled so run() can surface the
    reasoning in the narration. Other providers ignore the flag.
    """
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
    kwargs: dict = {}
    if thinking and provider == "anthropic":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}
        kwargs["max_tokens"] = THINKING_MAX_TOKENS
    return init_chat_model(name, model_provider=provider, **kwargs)


def _model_reference(model) -> str | None:
    if isinstance(model, str):
        return model
    for name in ("model_name", "model"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value:
            return value
    return None


# Anthropic prompt-caching breakpoint. Applied to the system prompt — the
# large static prefix of every request — so multi-turn runs pay cache-read
# rates (~10% of input price) instead of full input price on repeated turns.
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


def _supports_prompt_caching(model) -> bool:
    """True for LangChain Anthropic chat models (cache_control blocks)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return False
    return isinstance(model, ChatAnthropic)


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
    usage: list[dict] = field(default_factory=list)
    """Exact per-inference usage records generated during this run."""

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
            "usage": self.usage,
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
        usage_meter: durable JSONL meter (defaults to $PINNACE_USAGE_LOG or
            ~/.pinnace/usage.jsonl).
        cost_basis: explicit rates; known direct-provider defaults are used
            only for exact provider/model refs.
        agent_id / customer_id: attribution copied into every usage record.
        thinking: opt-in extended thinking for providers that support it
            (currently anthropic). Enables thinking mode when the model is
            built from a "provider:model" ref; has no effect on an
            already-built model passed in directly. Default off.
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
        usage_meter: UsageMeter | None = None,
        cost_basis: CostBasis | None = None,
        agent_id: str | None = None,
        customer_id: str | None = None,
        thinking: bool = False,
        prompt_caching: bool = True,
    ) -> None:
        if model is None:
            model = AgentConfig.resolve().model
        self.model_ref = _model_reference(model)
        self.thinking = thinking
        self.model = _make_model(model, thinking=thinking) if isinstance(model, str) else model
        self.prompt_caching = prompt_caching
        self._cache_system = prompt_caching and _supports_prompt_caching(self.model)
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
        if self._cache_system:
            self._add_tools_cache_breakpoint()
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.compaction_tokens = compaction_tokens
        self.compaction_keep_last = compaction_keep_last
        self.session_store = session_store or SessionStore()
        self.session_id = session_id
        self.log = log or (lambda *a: None)
        self.usage_meter = usage_meter or UsageMeter()
        self.cost_basis = cost_basis
        self.agent_id = (
            agent_id
            or os.environ.get("PINNACE_AGENT_ID")
            or session_id
            or "pinnace"
        )
        self.customer_id = customer_id or os.environ.get("PINNACE_CUSTOMER_ID")

    @classmethod
    def from_config(cls, config: AgentConfig) -> "PinnaceAgent":
        """Construct an agent from a reusable configuration value object."""
        return cls(**config.to_kwargs())

    def _say(self, msg: str) -> None:
        self.log(msg)

    def _system_message(self):
        """System prompt, with a prompt-caching breakpoint when supported."""
        if self._cache_system:
            return SystemMessage(content=[
                {"type": "text", "text": self.system_prompt,
                 "cache_control": dict(CACHE_CONTROL_EPHEMERAL)},
            ])
        return SystemMessage(content=self.system_prompt)

    def _add_tools_cache_breakpoint(self) -> None:
        """Mark the last bound tool definition cacheable (Anthropic)."""
        bound_kwargs = getattr(self.bound, "kwargs", None)
        if not isinstance(bound_kwargs, dict):
            return
        tools = bound_kwargs.get("tools")
        if not tools:
            return
        last = tools[-1]
        if isinstance(last, dict):
            last["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)

    def _record_usage(
        self,
        response: AIMessage,
        call_kind: str,
        records: list[dict],
    ) -> None:
        model = response_model(response, self.model_ref)
        record = self.usage_meter.record(
            response,
            model=model,
            model_ref=self.model_ref,
            call_kind=call_kind,
            agent_id=self.agent_id,
            customer_id=self.customer_id,
            session_id=self.session_id,
            cost_basis=self.cost_basis,
        )
        records.append(record)
        tokens = record["usage"]
        amount = record["cost"]["amount"]
        self._say(
            f"[pinnace] usage: model={model} input={tokens['input_tokens']} "
            f"output={tokens['output_tokens']} cache_read={tokens['cache_read_tokens']} "
            f"cost_usd={amount} status={record['cost']['status']} "
            f"agent={self.agent_id}"
        )

    def run(self, prompt: str) -> AgentResult:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(self._system_message())
        if self.session_id:
            history = self.session_store.load(self.session_id)
            if history:
                # History already carries its own system prompt; don't double it.
                messages = [m for m in history if not isinstance(m, SystemMessage)]
                if self.system_prompt:
                    messages.insert(0, self._system_message())
                self._say(f"[pinnace] resumed session {self.session_id!r} ({len(history)} messages)")
        if self._cache_system:
            self._say("[pinnace] prompt caching on (system prompt)")
        messages.append(HumanMessage(content=prompt))

        turns = 0
        compacted = 0
        structured = None
        final: str | None = None
        usage_records: list[dict] = []

        while turns < self.max_turns:
            turns += 1
            if needs_compaction(messages, self.compaction_tokens):
                before = estimate_tokens(messages)
                messages = compact_messages(
                    self.model,
                    messages,
                    self.compaction_keep_last,
                    on_response=lambda response: self._record_usage(
                        response, "compaction", usage_records
                    ),
                )
                compacted += 1
                self._say(f"[pinnace] compacted context ({before}→{estimate_tokens(messages)} est. tokens)")

            self._say(f"[pinnace] turn {turns}/{self.max_turns}, calling model…")
            resp: AIMessage = self.bound.invoke(messages)
            self._record_usage(resp, "agent_turn", usage_records)
            messages.append(resp)

            thinking = extract_thinking(resp)
            if thinking:
                if len(thinking) > THINKING_LOG_MAX_CHARS:
                    thinking = thinking[:THINKING_LOG_MAX_CHARS] + "…"
                self._say(f"[pinnace] thinking: {thinking}")

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
            usage=usage_records,
        )
