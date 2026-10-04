"""The agent loop, driven by a scripted fake model — no API keys."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from pinnace.agent import (
    PinnaceAgent,
    _make_model,
    _supports_prompt_caching,
    extract_thinking,
)
from pinnace.config import AgentConfig
from pinnace.sandbox import LocalSandbox
from pinnace.session import SessionStore
import pinnace.agent as agent_mod


def _tc(name, args, i):
    return {"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}


class FakeModel:
    """Replays a script of AIMessages, one per invoke."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.script[min(self.calls - 1, len(self.script) - 1)]


def _agent(tmp_path, script, **kw):
    kw.setdefault("session_store", SessionStore(tmp_path / "sessions"))
    return PinnaceAgent(
        model=FakeModel(script),
        sandbox=LocalSandbox(str(tmp_path / "work"), unsafe_ok=True),
        log=lambda *a: None,
        **kw,
    )


def test_run_shell_then_finish(tmp_path):
    agent = _agent(tmp_path, [
        AIMessage(content="", tool_calls=[_tc("shell", {"command": "echo hi"}, 1)]),
        AIMessage(content="", tool_calls=[_tc("finish", {"result": '{"done": true}'}, 2)]),
    ])
    result = agent.run("do the thing")
    assert result.turns == 2
    assert result.structured == {"done": True}
    assert result.finished
    # The shell output made it into the transcript as a ToolMessage.
    def _kind(m):
        return m.get("id", [""])[-1]

    tool_msgs = [m for m in result.transcript if _kind(m) == "ToolMessage"]
    assert any("hi" in m["kwargs"]["content"] for m in tool_msgs)


def test_run_no_tools_returns_final(tmp_path):
    agent = _agent(tmp_path, [AIMessage(content="all done, nothing to do")])
    result = agent.run("quick question")
    assert result.turns == 1
    assert result.final == "all done, nothing to do"
    assert not result.finished


def test_max_turns_stops_loop(tmp_path):
    script = [AIMessage(content="", tool_calls=[_tc("shell", {"command": "echo x"}, i)])
              for i in range(10)]
    agent = _agent(tmp_path, script, max_turns=3)
    result = agent.run("loop forever")
    assert result.turns == 3
    assert result.final is None


def test_unknown_tool_reported_not_crash(tmp_path):
    agent = _agent(tmp_path, [
        AIMessage(content="", tool_calls=[_tc("nope", {}, 1)]),
        AIMessage(content="ok nevermind"),
    ])
    result = agent.run("try unknown")
    assert result.final == "ok nevermind"
    tool_msgs = [m for m in result.transcript if m.get("id", [""])[-1] == "ToolMessage"]
    assert "unknown tool" in tool_msgs[0]["kwargs"]["content"]


def test_session_persists_and_resumes(tmp_path):
    store_path = tmp_path / "sessions"
    a1 = _agent(tmp_path, [AIMessage(content="first run done")],
                session_store=SessionStore(store_path), session_id="s1")
    r1 = a1.run("task one")
    assert r1.session_id == "s1"

    a2 = _agent(tmp_path, [AIMessage(content="second run done")],
                session_store=SessionStore(store_path), session_id="s1")
    r2 = a2.run("task two")
    # Resumed transcript contains both user prompts.
    humans = [m for m in r2.transcript if m.get("id", [""])[-1] == "HumanMessage"]
    assert [m["kwargs"]["content"] for m in humans] == ["task one", "task two"]


def _thinking_msg(thinking, text="final answer"):
    return AIMessage(
        content=[
            {"type": "thinking", "thinking": thinking, "signature": "sig"},
            {"type": "text", "text": text},
        ]
    )


def test_extract_thinking_dict_blocks():
    resp = _thinking_msg("consider the evidence carefully")
    assert extract_thinking(resp) == "consider the evidence carefully"


def test_extract_thinking_object_blocks():
    # Some providers return content blocks as objects; assign post-construction
    # since AIMessage validation only accepts str/dict content.
    resp = AIMessage(content="placeholder")
    resp.content = [
        SimpleNamespace(type="thinking", thinking="object-style block"),
        SimpleNamespace(type="text", text="answer"),
    ]
    assert extract_thinking(resp) == "object-style block"


def test_extract_thinking_redacted():
    resp = AIMessage(content=[
        {"type": "redacted_thinking", "data": "encrypted"},
        {"type": "text", "text": "answer"},
    ])
    assert extract_thinking(resp) == "[redacted thinking]"


def test_extract_thinking_additional_kwargs():
    resp = AIMessage(
        content="answer",
        additional_kwargs={"reasoning_content": "provider-side reasoning"},
    )
    assert extract_thinking(resp) == "provider-side reasoning"


def test_extract_thinking_empty_when_none():
    assert extract_thinking(AIMessage(content="just text")) == ""
    assert extract_thinking(AIMessage(content="")) == ""
    assert extract_thinking(AIMessage(content=[{"type": "text", "text": "x"}])) == ""


def _capture_agent(tmp_path, script, **kw):
    logged = []
    agent = PinnaceAgent(
        model=FakeModel(script),
        sandbox=LocalSandbox(str(tmp_path / "work"), unsafe_ok=True),
        session_store=SessionStore(tmp_path / "sessions"),
        log=logged.append,
        **kw,
    )
    return agent, logged


def test_run_emits_thinking_line(tmp_path):
    script = [
        AIMessage(
            content=[{"type": "thinking", "thinking": "check files first", "signature": "s"}],
            tool_calls=[_tc("shell", {"command": "echo hi"}, 1)],
        ),
        AIMessage(content="done"),
    ]
    agent, logged = _capture_agent(tmp_path, script)
    result = agent.run("do the thing")
    assert result.turns == 2
    lines = [m for m in logged if m.startswith("[pinnace] thinking:")]
    assert len(lines) == 1
    assert "check files first" in lines[0]


def test_run_thinking_truncated(tmp_path):
    long_thinking = "x" * 5000
    agent, logged = _capture_agent(tmp_path, [_thinking_msg(long_thinking)])
    agent.run("quick question")
    lines = [m for m in logged if m.startswith("[pinnace] thinking:")]
    assert len(lines) == 1
    assert len(lines[0]) <= len("[pinnace] thinking: ") + 2000 + 1
    assert lines[0].endswith("…")


def test_run_no_thinking_no_line(tmp_path):
    agent, logged = _capture_agent(tmp_path, [AIMessage(content="plain")])
    agent.run("quick question")
    assert not [m for m in logged if m.startswith("[pinnace] thinking:")]


def test_thinking_flag_stored_on_agent(tmp_path):
    agent, _ = _capture_agent(tmp_path, [AIMessage(content="x")], thinking=True)
    assert agent.thinking is True
    agent2, _ = _capture_agent(tmp_path, [AIMessage(content="x")])
    assert agent2.thinking is False


def _patched_init(monkeypatch):
    import langchain.chat_models

    seen = {}

    def fake_init(name, model_provider=None, **kwargs):
        seen["name"] = name
        seen["model_provider"] = model_provider
        seen["kwargs"] = kwargs
        return "fake-model"

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init)
    return seen


def test_make_model_thinking_enables_anthropic_thinking(monkeypatch):
    seen = _patched_init(monkeypatch)
    assert _make_model("anthropic:claude-opus-4-6", thinking=True) == "fake-model"
    assert seen["kwargs"]["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    # Anthropic requires max_tokens > budget_tokens, so it is raised too.
    assert seen["kwargs"]["max_tokens"] == 8000


def test_make_model_thinking_off_sends_no_thinking_kwargs(monkeypatch):
    seen = _patched_init(monkeypatch)
    _make_model("anthropic:claude-opus-4-6", thinking=False)
    assert "thinking" not in seen["kwargs"]
    assert "max_tokens" not in seen["kwargs"]


def test_make_model_thinking_ignored_for_other_providers(monkeypatch):
    seen = _patched_init(monkeypatch)
    _make_model("openai:gpt-5", thinking=True)
    assert "thinking" not in seen["kwargs"]
def _cached_agent(tmp_path, **kw):
    """Agent with prompt-caching support forced on (no API key needed)."""
    orig = agent_mod._supports_prompt_caching
    agent_mod._supports_prompt_caching = lambda m: True
    try:
        return _agent(tmp_path, [AIMessage(content="done")], **kw)
    finally:
        agent_mod._supports_prompt_caching = orig


def test_system_message_cache_blocks_when_supported(tmp_path):
    agent = _cached_agent(tmp_path)
    assert agent._cache_system is True
    m = agent._system_message()
    assert isinstance(m.content, list)
    block = m.content[0]
    assert block["type"] == "text"
    assert block["text"] == agent.system_prompt
    assert block["cache_control"] == {"type": "ephemeral"}


def test_prompt_caching_opt_out(tmp_path):
    agent = _cached_agent(tmp_path, prompt_caching=False)
    assert agent._cache_system is False
    assert isinstance(agent._system_message().content, str)


def test_run_emits_caching_log_line(tmp_path):
    lines = []
    agent = _cached_agent(tmp_path)
    agent.log = lambda s: lines.append(s)
    agent.run("hi")
    assert any("prompt caching" in line for line in lines)


def test_run_uses_cached_system_message_in_transcript(tmp_path):
    agent = _cached_agent(tmp_path)
    result = agent.run("hi")
    first = result.transcript[0]
    assert first["id"][-1] == "SystemMessage"
    assert isinstance(first["kwargs"]["content"], list)


def test_supports_prompt_caching_rejects_other_models():
    class FakeModel:
        pass

    assert _supports_prompt_caching(FakeModel()) is False
    assert _supports_prompt_caching(object()) is False


def test_config_prompt_caching_roundtrip():
    assert AgentConfig().to_kwargs()["prompt_caching"] is True
    assert AgentConfig(prompt_caching=False).to_kwargs()["prompt_caching"] is False


def test_tools_cache_breakpoint_on_last_tool(tmp_path):
    """The last bound tool definition carries the cache breakpoint."""
    from langchain_anthropic import ChatAnthropic

    from pinnace.sandbox import LocalSandbox
    from pinnace.session import SessionStore

    agent = PinnaceAgent(
        model=ChatAnthropic(model="claude-opus-4-6"),
        sandbox=LocalSandbox(str(tmp_path / "work"), unsafe_ok=True),
        session_store=SessionStore(tmp_path / "sessions"),
        log=lambda *a: None,
    )
    assert agent._cache_system is True
    tools = agent.bound.kwargs["tools"]
    assert len(tools) == 6
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in t for t in tools[:-1])


def test_tools_cache_breakpoint_opt_out(tmp_path):
    from langchain_anthropic import ChatAnthropic

    from pinnace.sandbox import LocalSandbox
    from pinnace.session import SessionStore

    agent = PinnaceAgent(
        model=ChatAnthropic(model="claude-opus-4-6"),
        sandbox=LocalSandbox(str(tmp_path / "work"), unsafe_ok=True),
        session_store=SessionStore(tmp_path / "sessions"),
        log=lambda *a: None,
        prompt_caching=False,
    )
    tools = agent.bound.kwargs["tools"]
    assert all("cache_control" not in t for t in tools)


def test_tools_cache_breakpoint_noop_for_fake_model(tmp_path):
    """Non-Anthropic models: no breakpoint, no crash."""
    agent = _agent(tmp_path, [AIMessage(content="done")])
    assert agent._cache_system is False
    tools = getattr(getattr(agent, "bound", None), "kwargs", {}).get("tools", [])
    assert all("cache_control" not in t for t in tools)
