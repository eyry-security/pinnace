"""The multi-turn agent loop.

Hand it a prompt (or seed message) and it runs: sends the conversation to the
LLM, dispatches any tool calls, appends results, and repeats until the LLM
responds with plain text (no tool calls) or we hit the turn limit.

Context compaction happens automatically when the message list grows past the
threshold — the agent asks the LLM to summarise the conversation so far,
replaces the old messages with that summary, and keeps going.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .config import AgentConfig
from .context import Context, Message
from .sandbox import Sandbox
from .tools import Tool, ToolRegistry, ToolResult, default_tools

log = logging.getLogger(__name__)


class Agent:
    """A multi-turn agent that talks to an OpenAI-compatible API and dispatches
    tool calls into a sandboxed Docker container."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.config = config or AgentConfig.resolve()
        self.registry = ToolRegistry()
        for tool in (tools or default_tools()):
            self.registry.register(tool)
        self.context: Context | None = None
        self.sandbox: Sandbox | None = None
        self._client: Any = None  # openai.OpenAI instance

    def _init_client(self) -> Any:
        import openai  # late import
        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """One round-trip to the LLM."""
        client = self._client or self._init_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs)

    def _compact_if_needed(self) -> None:
        assert self.context is not None
        if not self.context.needs_compaction():
            return
        prompt = self.context.compaction_prompt()
        if not prompt:
            return
        log.info("compacting context (messages=%d, compaction #%d)",
                 len(self.context), self.context.compaction_count + 1)
        resp = self._chat(prompt)
        summary = resp.choices[0].message.content or ""
        self.context.compact(summary)
        log.info("compacted to %d messages", len(self.context))

    def _dispatch_tool_call(self, call: Any) -> ToolResult:
        """Execute a single tool call and return the result."""
        name = call.function.name
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            return ToolResult(f"invalid JSON arguments: {call.function.arguments}", ok=False)

        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(f"unknown tool: {name}", ok=False)

        log.info("tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])
        try:
            result = tool.execute(**args)
        except Exception as exc:
            result = ToolResult(f"{type(exc).__name__}: {exc}", ok=False)
        log.info("tool result: ok=%s len=%d", result.ok, len(result.output))
        return result

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        on_message: Any = None,
    ) -> str:
        """Run the agent loop to completion and return the final response.

        Parameters
        ----------
        prompt : str
            The user's initial message / seed.
        system_prompt : str, optional
            Override the config's system prompt.
        on_message : callable, optional
            ``on_message(role, content)`` called after each assistant or tool
            message, for streaming output to a terminal.
        """
        sys_prompt = system_prompt or self.config.system_prompt or (
            "You are Pinnace, an autonomous agent with access to a sandboxed "
            "Linux environment. Use the provided tools to accomplish the user's "
            "task. Think step by step, use tools as needed, and report your "
            "results clearly when done."
        )

        self.context = Context(
            system_prompt=sys_prompt,
            compaction_threshold=self.config.compaction_threshold,
            keep_recent=self.config.compaction_keep_recent,
        )
        self.context.add_user(prompt)

        # Bind sandbox to built-in tools.
        if self.sandbox is not None:
            for name in ("shell", "read_file", "write_file"):
                tool = self.registry.get(name)
                if tool is not None and hasattr(tool, "bind"):
                    tool.bind(self.sandbox)

        tool_schemas = self.registry.openai_schemas() or None
        last_text = ""

        for turn in range(self.config.max_turns):
            self._compact_if_needed()
            resp = self._chat(self.context.to_dicts(), tools=tool_schemas)
            choice = resp.choices[0]
            msg = choice.message

            # Plain text response — the agent is done (or pausing for user input).
            if not msg.tool_calls:
                text = msg.content or ""
                self.context.add_assistant(content=text)
                if on_message:
                    on_message("assistant", text)
                last_text = text
                break

            # Tool calls — record the assistant message, execute, and loop.
            self.context.add_assistant(
                content=msg.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            )
            if on_message and msg.content:
                on_message("assistant", msg.content)

            for tc in msg.tool_calls:
                result = self._dispatch_tool_call(tc)
                self.context.add_tool_result(tc.id, tc.function.name, result.to_message())
                if on_message:
                    on_message("tool", f"[{tc.function.name}] {result.to_message()[:500]}")
        else:
            last_text = f"[pinnace] hit turn limit ({self.config.max_turns})"
            if on_message:
                on_message("system", last_text)

        return last_text

    def run_sandboxed(
        self,
        prompt: str,
        system_prompt: str | None = None,
        on_message: Any = None,
    ) -> str:
        """Run with an auto-managed sandbox: start before, tear down after."""
        self.sandbox = Sandbox(
            image=self.config.sandbox_image,
            timeout=self.config.sandbox_timeout,
            mem_limit=self.config.sandbox_mem_limit,
            network=self.config.sandbox_network,
        )
        try:
            self.sandbox.start()
            return self.run(prompt, system_prompt=system_prompt, on_message=on_message)
        finally:
            self.sandbox.close()
            self.sandbox = None
