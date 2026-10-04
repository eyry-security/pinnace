"""Exact inference usage metering tests — no provider calls required."""

import json
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage

from pinnace.agent import PinnaceAgent
from pinnace.config import AgentConfig
from pinnace.sandbox import LocalSandbox
from pinnace.session import SessionStore
from pinnace.usage import CostBasis, TokenUsage, UsageMeter, default_cost_basis


def _response(*, content="done", usage=True, unclassified=False):
    usage_metadata = None
    response_metadata = {"model_name": "claude-opus-4-6-20261001"}
    if usage:
        usage_metadata = {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "total_tokens": 1_100,
            "input_token_details": {
                "cache_read": 200,
                "cache_creation": 300,
            },
        }
        if not unclassified:
            response_metadata["usage"] = {
                "cache_creation_input_tokens": 300,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 100,
                    "ephemeral_1h_input_tokens": 200,
                },
            }
    return AIMessage(
        content=content,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )


def test_exact_usage_and_decimal_cost_include_cache_ttls():
    response = _response()
    usage = TokenUsage.from_response(response)
    basis = default_cost_basis("anthropic:claude-opus-4-6")

    assert usage.to_dict() == {
        "input_tokens": 1_000,
        "uncached_input_tokens": 500,
        "output_tokens": 100,
        "total_tokens": 1_100,
        "cache_read_tokens": 200,
        "cache_write_5m_tokens": 100,
        "cache_write_1h_tokens": 200,
        "cache_write_unclassified_tokens": 0,
    }
    assert usage.cost(basis) == Decimal("0.007725")


def test_meter_persists_complete_attribution_and_basis(tmp_path):
    path = tmp_path / "usage.jsonl"
    forwarded = []
    meter = UsageMeter(path, sink=forwarded.append)

    record = meter.record(
        _response(),
        model="claude-opus-4-6-20261001",
        model_ref="anthropic:claude-opus-4-6",
        call_kind="agent_turn",
        agent_id="reviewer-7",
        customer_id="team-3",
        session_id="scan-9",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == record == forwarded[0]
    assert persisted["agent_id"] == "reviewer-7"
    assert persisted["customer_id"] == "team-3"
    assert persisted["session_id"] == "scan-9"
    assert persisted["model"] == "claude-opus-4-6-20261001"
    assert persisted["cost"]["amount"] == "0.007725"
    assert persisted["cost"]["status"] == "priced"
    assert persisted["cost_basis"]["input"] == "5"
    assert persisted["cost_basis"]["cache_write_1h"] == "10"


def test_missing_or_ambiguous_usage_is_logged_but_never_guessed(tmp_path):
    path = tmp_path / "usage.jsonl"
    meter = UsageMeter(path)
    common = {
        "model": "claude-opus-4-6",
        "model_ref": "anthropic:claude-opus-4-6",
        "call_kind": "agent_turn",
        "agent_id": "agent",
        "customer_id": None,
        "session_id": None,
    }

    missing = meter.record(_response(usage=False), **common)
    ambiguous = meter.record(_response(unclassified=True), **common)

    assert missing["usage"]["input_tokens"] is None
    assert missing["cost"] == {
        "currency": "USD",
        "amount": None,
        "status": "usage_unavailable",
    }
    assert ambiguous["usage"]["cache_write_unclassified_tokens"] == 300
    assert ambiguous["cost"]["amount"] is None
    assert ambiguous["cost"]["status"] == "cache_write_ttl_unavailable"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_agent_meters_compaction_and_turn_with_per_agent_attribution(tmp_path):
    store = SessionStore(tmp_path / "state")
    store.save(
        "persistent-session",
        [
            HumanMessage(content="one"),
            AIMessage(content="two"),
            HumanMessage(content="three"),
            AIMessage(content="four"),
        ],
    )
    path = tmp_path / "meter" / "usage.jsonl"
    basis = CostBasis("1", "2", "0.1", "1.25", "2", "test", "2026-10-04")
    model = ScriptedModel([
        _response(content="compact summary"),
        _response(content="final answer"),
    ])
    agent = PinnaceAgent(
        model=model,
        sandbox=LocalSandbox(tmp_path / "work", unsafe_ok=True),
        session_store=store,
        session_id="persistent-session",
        compaction_tokens=0,
        compaction_keep_last=1,
        usage_meter=UsageMeter(path),
        cost_basis=basis,
        agent_id="named-agent",
        customer_id="customer-a",
    )

    result = agent.run("continue")

    assert result.final == "final answer"
    assert [record["call_kind"] for record in result.usage] == [
        "compaction",
        "agent_turn",
    ]
    assert all(record["agent_id"] == "named-agent" for record in result.usage)
    assert all(record["customer_id"] == "customer-a" for record in result.usage)
    assert all(record["session_id"] == "persistent-session" for record in result.usage)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert result.to_dict()["usage"] == result.usage


def test_agent_config_forwards_meter_and_attribution(tmp_path):
    meter = UsageMeter(tmp_path / "usage.jsonl")
    basis = CostBasis("1", "2", "0.1", "1.25", "2", "test", "2026-10-04")
    config = AgentConfig(
        model=ScriptedModel([_response()]),
        sandbox=LocalSandbox(tmp_path / "work-config", unsafe_ok=True),
        usage_meter=meter,
        cost_basis=basis,
        agent_id="configured-agent",
        customer_id="configured-customer",
    )

    agent = PinnaceAgent.from_config(config)

    assert agent.usage_meter is meter
    assert agent.cost_basis is basis
    assert agent.agent_id == "configured-agent"
    assert agent.customer_id == "configured-customer"


def test_cache_ttl_split_without_aggregate_is_not_double_billed():
    response = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 1_000,
            "output_tokens": 100,
            "total_tokens": 1_100,
            "input_token_details": {"cache_read": 200},
        },
        response_metadata={
            "usage": {
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 100,
                    "ephemeral_1h_input_tokens": 200,
                }
            }
        },
    )

    usage = TokenUsage.from_response(response)

    assert usage.uncached_input_tokens == 500
    assert usage.cache_write_unclassified_tokens == 0
    assert usage.cost(default_cost_basis("anthropic:claude-opus-4-6")) == Decimal(
        "0.007725"
    )
