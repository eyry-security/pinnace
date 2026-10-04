"""CLI construction and sandbox lifecycle checks."""

from pinnace import AgentResult
from pinnace import cli
from pinnace.agent import PinnaceError
from pinnace.config import DEFAULT_SYSTEM


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
