"""CLI construction and sandbox lifecycle checks."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pinnace import AgentResult
from pinnace import cli
from pinnace.agent import PinnaceError
from pinnace.config import DEFAULT_SYSTEM
from pinnace.session import SessionStore


class FakeSandbox:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeAgent:
    def run(self, prompt):
        return AgentResult(prompt, None, 1, 0, None)


def _args(*extra):
    return cli.build_parser().parse_args([
        "run", "--prompt", "hello", "--quiet", *extra,
    ])


def test_cli_uses_default_and_explicit_empty_system(monkeypatch, capsys):
    captured = []
    sandboxes = []

    def make_sandbox(args):
        sandbox = FakeSandbox()
        sandboxes.append(sandbox)
        return sandbox

    def from_config(config):
        captured.append(config)
        return FakeAgent()

    monkeypatch.setattr(cli, "_make_sandbox", make_sandbox)
    monkeypatch.setattr(cli.PinnaceAgent, "from_config", from_config)

    assert cli.cmd_run(_args()) == 0
    assert cli.cmd_run(_args("--system", "")) == 0
    assert captured[0].system_prompt == DEFAULT_SYSTEM
    assert captured[1].system_prompt == ""
    assert all(sandbox.closed for sandbox in sandboxes)
    capsys.readouterr()


def test_cli_closes_sandbox_when_agent_construction_fails(monkeypatch, capsys):
    sandbox = FakeSandbox()
    monkeypatch.setattr(cli, "_make_sandbox", lambda args: sandbox)

    def fail(config):
        raise PinnaceError("bad model")

    monkeypatch.setattr(cli.PinnaceAgent, "from_config", fail)
    assert cli.cmd_run(_args()) == 1
    assert sandbox.closed
    assert "bad model" in capsys.readouterr().err


class FakeCompactionModel:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("summarizer unavailable")
        return AIMessage(
            content="dense session summary",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        )


def _compact_args(tmp_path, *extra):
    return cli.build_parser().parse_args([
        "compact",
        "demo",
        "--root", str(tmp_path),
        "--model", "test:summary",
        *extra,
    ])


def test_cli_compacts_saved_session_and_meters_usage(monkeypatch, tmp_path, capsys):
    store = SessionStore(tmp_path)
    store.save("demo", [
        SystemMessage(content="system"),
        HumanMessage(content="old one"),
        HumanMessage(content="old two"),
        HumanMessage(content="recent"),
    ])
    model = FakeCompactionModel()
    usage_log = tmp_path / "usage.jsonl"
    monkeypatch.setattr(cli, "_make_model", lambda model_ref: model)

    args = _compact_args(
        tmp_path,
        "--keep-last", "1",
        "--usage-log", str(usage_log),
        "--json",
    )
    assert cli.cmd_compact(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "demo"
    assert payload["compacted"] is True
    assert payload["messages_summarized"] == 2
    assert payload["message_count_after"] == 3
    assert payload["usage"][0]["call_kind"] == "compaction"

    saved = store.load("demo")
    assert saved is not None
    assert [message.content for message in saved] == [
        "system",
        "Conversation so far (compacted — details summarized, tail kept verbatim):\n"
        "dense session summary",
        "recent",
    ]
    usage = json.loads(usage_log.read_text())
    assert usage["session_id"] == "demo"
    assert usage["agent_id"] == "demo"
    assert usage["usage"]["total_tokens"] == 15


def test_cli_compaction_noop_does_not_build_model(monkeypatch, tmp_path, capsys):
    store = SessionStore(tmp_path)
    original = [SystemMessage(content="system"), HumanMessage(content="recent")]
    store.save("demo", original)

    def unexpected_model(_model_ref):
        raise AssertionError("model should not be built for a no-op")

    monkeypatch.setattr(cli, "_make_model", unexpected_model)
    assert cli.cmd_compact(_compact_args(tmp_path, "--json")) == 0
    assert json.loads(capsys.readouterr().out)["compacted"] is False
    saved = store.load("demo")
    assert saved is not None
    assert [message.content for message in saved] == ["system", "recent"]


def test_cli_compaction_failure_preserves_session(monkeypatch, tmp_path, capsys):
    store = SessionStore(tmp_path)
    original = [HumanMessage(content=f"message {i}") for i in range(3)]
    store.save("demo", original)
    monkeypatch.setattr(
        cli,
        "_make_model",
        lambda model_ref: FakeCompactionModel(fail=True),
    )

    assert cli.cmd_compact(_compact_args(tmp_path, "--keep-last", "1")) == 1
    assert "summarizer unavailable" in capsys.readouterr().err
    saved = store.load("demo")
    assert saved is not None
    assert [message.content for message in saved] == [message.content for message in original]


def test_cli_compaction_reports_missing_session(tmp_path, capsys):
    assert cli.cmd_compact(_compact_args(tmp_path)) == 1
    assert "not found" in capsys.readouterr().err
