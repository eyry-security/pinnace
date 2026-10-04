"""Compaction is pure logic — no model calls except the summarizer, which we fake."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pinnace.compaction import compact_messages, estimate_tokens, needs_compaction


class FakeSummarizer:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="SUMMARY: did stuff")


def test_estimate_tokens_scales_with_content():
    short = [HumanMessage(content="hi")]
    long = [HumanMessage(content="x" * 4000)]
    assert estimate_tokens(long) > estimate_tokens(short)
    assert estimate_tokens(short) == len("hi") // 4


def test_estimate_counts_tool_calls():
    m = [AIMessage(content="", tool_calls=[
        {"name": "shell", "args": {"command": "x" * 400}, "id": "1", "type": "tool_call"}
    ])]
    assert estimate_tokens(m) >= 100


def test_needs_compaction_tripwire():
    msgs = [HumanMessage(content="x" * 4000)]  # ~1000 tokens
    assert needs_compaction(msgs, 500)
    assert not needs_compaction(msgs, 5000)


def test_compact_keeps_system_and_tail():
    model = FakeSummarizer()
    msgs = [SystemMessage(content="ORIGINAL SYSTEM")] + [
        HumanMessage(content=f"old message {i} " + "y" * 500) for i in range(10)
    ] + [HumanMessage(content="recent")]
    out = compact_messages(model, msgs, keep_last=2)
    assert model.calls == 1
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "ORIGINAL SYSTEM"
    assert "SUMMARY" in out[1].content
    assert out[-1].content == "recent"
    assert out[-2].content.startswith("old message 9")


def test_compact_noop_when_short():
    model = FakeSummarizer()
    msgs = [HumanMessage(content="hi")]
    assert compact_messages(model, msgs) is msgs
    assert model.calls == 0


from langchain_core.messages import HumanMessage, SystemMessage

from pinnace.compaction_controls import (
    CompactionConfig,
    compact_with_config,
)


class _EchoModel:
    def invoke(self, messages):
        m = HumanMessage(content="summary of stuff")
        return m


def test_no_compaction_below_tripwire():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    out, report = compact_with_config(_EchoModel(), msgs)
    assert out == msgs
    assert report is None


def test_compaction_report():
    cfg = CompactionConfig(max_tokens=10, keep_last=1)
    msgs = [SystemMessage(content="sys")] + [HumanMessage(content="x" * 100) for _ in range(5)]
    out, report = compact_with_config(_EchoModel(), msgs, cfg)
    assert report is not None
    assert report.messages_summarized == 4
    assert report.tokens_before > report.tokens_after
    assert "compacted 4 messages" in report.one_liner()
    # head kept, summary inserted, tail kept
    assert isinstance(out[0], SystemMessage) and out[0].content == "sys"
    assert "compacted" in out[1].content
    assert len(out) == 3  # head + summary + 1 tail


def test_custom_summarize_prompt():
    seen = {}

    class SpyModel:
        def invoke(self, messages):
            seen["prompt"] = messages[0].content
            return HumanMessage(content="s")

    cfg = CompactionConfig(max_tokens=1, keep_last=0, summarize_prompt="CUSTOM")
    compact_with_config(SpyModel(), [HumanMessage(content="y" * 100)], cfg)
    assert seen["prompt"] == "CUSTOM"


def test_config_defaults_match_legacy():
    cfg = CompactionConfig()
    assert cfg.max_tokens == 100_000
    assert cfg.keep_last == 8


def test_force_compaction_bypasses_tripwire_and_observes_response():
    model = _EchoModel()
    cfg = CompactionConfig(max_tokens=1_000_000, keep_last=1)
    msgs = [HumanMessage(content="old"), HumanMessage(content="recent")]
    observed = []

    unchanged, report = compact_with_config(model, msgs, cfg)
    assert unchanged is msgs
    assert report is None

    out, report = compact_with_config(
        model,
        msgs,
        cfg,
        force=True,
        on_response=observed.append,
    )
    assert report is not None
    assert report.messages_summarized == 1
    assert len(observed) == 1
    assert out[-1].content == "recent"
    assert report.to_dict()["summary_chars"] == len("summary of stuff")


def test_compaction_config_rejects_negative_limits():
    import pytest

    with pytest.raises(ValueError, match="max_tokens"):
        CompactionConfig(max_tokens=-1)
    with pytest.raises(ValueError, match="keep_last"):
        CompactionConfig(keep_last=-1)
