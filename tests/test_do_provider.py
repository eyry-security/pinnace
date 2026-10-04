"""DigitalOcean inference provider (do:) — no API keys needed for these tests."""

import os

import pytest

from pinnace.agent import PinnaceError, _make_model


def test_do_provider_needs_key(monkeypatch):
    monkeypatch.delenv("DO_INFERENCE_KEY", raising=False)
    with pytest.raises(PinnaceError, match="DO_INFERENCE_KEY"):
        _make_model("do:anthropic-claude-opus-4.6")


def test_do_provider_builds_openai_compatible_model(monkeypatch):
    monkeypatch.setenv("DO_INFERENCE_KEY", "test-key")
    model = _make_model("do:anthropic-claude-opus-4.6")
    assert model.model_name == "anthropic-claude-opus-4.6"
    assert model.openai_api_base == "https://inference.do-ai.run/v1"


def test_do_provider_bad_ref():
    with pytest.raises(PinnaceError, match="bad model ref"):
        _make_model("do:")
