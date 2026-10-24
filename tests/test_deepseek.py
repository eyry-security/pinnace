"""DeepSeek catalog and resolution tests — no provider calls or API keys."""

from decimal import Decimal

from langchain_core.messages import AIMessage

from pinnace.agent import PinnaceError, _make_model
from pinnace.usage import TokenUsage, UsageMeter, default_cost_basis


def _patched_init(monkeypatch):
    import langchain.chat_models

    seen = []

    def fake_init(name, model_provider=None, **kwargs):
        seen.append(
            {"name": name, "model_provider": model_provider, "kwargs": kwargs}
        )
        return "fake-model"

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init)
    return seen


def test_deepseek_models_resolve_through_openai_compatible_path(monkeypatch):
    seen = _patched_init(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key-for-test")

    assert _make_model("deepseek:deepseek-flash") == "fake-model"
    assert _make_model("deepseek:deepseek-v4-pro") == "fake-model"

    assert [call["name"] for call in seen] == ["deepseek-flash", "deepseek-v4-pro"]
    assert all(call["model_provider"] == "openai" for call in seen)
    assert all(
        call["kwargs"]["base_url"] == "https://api.deepseek.com/v1" for call in seen
    )
    assert all(
        call["kwargs"]["api_key"] == "dummy-key-for-test" for call in seen
    )


def test_deepseek_catalog_pricing_is_exact():
    flash = default_cost_basis("deepseek:deepseek-flash")
    pro = default_cost_basis("deepseek:deepseek-v4-pro")

    assert flash is not None and pro is not None
    assert flash.input_per_million == Decimal("0.15")
    assert flash.cache_read_per_million == Decimal("0.003")
    assert flash.output_per_million == Decimal("0.60")
    assert pro.input_per_million == Decimal("0.66")
    assert pro.output_per_million == Decimal("1.98")

    response = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1_000,
            "output_tokens": 100,
            "total_tokens": 1_100,
            "input_token_details": {"cache_read": 200},
        },
    )
    usage = TokenUsage.from_response(response)
    assert usage.cost(flash) == Decimal("0.0001806")


def test_unknown_models_stay_unpriced_and_never_borrow_other_tables(tmp_path):
    assert default_cost_basis("deepseek:deepseek-unknown") is None
    assert default_cost_basis("deepseek:claude-opus-4-6") is None
    assert default_cost_basis("openai:deepseek-flash") is None
    assert default_cost_basis("anthropic:claude-opus-4-6") is not None

    response = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
        },
    )
    record = UsageMeter(tmp_path / "usage.jsonl").record(
        response,
        model="deepseek-unknown",
        model_ref="deepseek:deepseek-unknown",
        call_kind="agent_turn",
        agent_id="agent",
        customer_id=None,
        session_id=None,
    )

    assert record["cost_basis"] is None
    assert record["cost"] == {
        "currency": "USD",
        "amount": None,
        "status": "cost_basis_unavailable",
    }


def test_deepseek_requires_its_own_key(monkeypatch):
    _patched_init(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    try:
        _make_model("deepseek:deepseek-flash")
    except PinnaceError as error:
        assert "DEEPSEEK_API_KEY" in str(error)
    else:
        raise AssertionError("deepseek model resolved without its key")
