"""Public configuration and message contracts."""

from langchain_core.messages import BaseMessage

from pinnace import AgentConfig, Message, PinnaceAgent
from pinnace.config import DEFAULT_MODEL, DEFAULT_SYSTEM
from pinnace.sandbox import LocalSandbox
from pinnace.session import SessionStore


class FakeModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self


def test_resolve_model_precedence_and_defaults(monkeypatch):
    monkeypatch.delenv("PINNACE_MODEL", raising=False)
    assert AgentConfig.resolve().model == DEFAULT_MODEL

    monkeypatch.setenv("PINNACE_MODEL", "openai:test-model")
    assert AgentConfig.resolve().model == "openai:test-model"
    explicit = FakeModel()
    assert AgentConfig.resolve(model=explicit).model is explicit


def test_config_preserves_falsey_values_and_isolates_tools():
    first = AgentConfig.resolve(system_prompt="", max_turns=0)
    second = AgentConfig.resolve()
    assert first.system_prompt == ""
    assert first.max_turns == 0
    assert second.system_prompt == DEFAULT_SYSTEM
    assert first.tools is not second.tools


def test_from_config_forwards_values(tmp_path):
    model = FakeModel()
    sandbox = LocalSandbox(tmp_path / "work", unsafe_ok=True)
    store = SessionStore(tmp_path / "state")
    config = AgentConfig(
        model=model,
        sandbox=sandbox,
        system_prompt="custom",
        max_turns=7,
        compaction_tokens=1234,
        compaction_keep_last=3,
        session_store=store,
        session_id="run-1",
    )
    agent = PinnaceAgent.from_config(config)
    assert agent.model is model
    assert agent.sandbox is sandbox
    assert agent.system_prompt == "custom"
    assert agent.max_turns == 7
    assert agent.compaction_tokens == 1234
    assert agent.compaction_keep_last == 3
    assert agent.session_store is store
    assert agent.session_id == "run-1"
    sandbox.close()


def test_message_aliases_langchain_runtime_type():
    assert Message is BaseMessage
