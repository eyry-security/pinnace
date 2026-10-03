"""The agent loop, driven by a scripted fake model — no API keys."""

from langchain_core.messages import AIMessage

from pinnace.agent import PinnaceAgent
from pinnace.sandbox import LocalSandbox
from pinnace.session import SessionStore


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
