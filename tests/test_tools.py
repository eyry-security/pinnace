"""python_scratchpad tests: a mock sandbox, no execution of real code."""

import pytest

from pinnace.sandbox import ExecResult
from pinnace.tools import builtin_tools


class MockSandbox:
    """Records write_file/exec calls; returns a configurable ExecResult."""

    def __init__(self, result=None, write_error=None):
        self.result = result or ExecResult(stdout="", stderr="", exit_code=0)
        self.write_error = write_error
        self.written = {}
        self.last_cmd = None

    def exec(self, cmd, timeout=120.0):
        self.last_cmd = cmd
        return self.result

    def read_file(self, path):
        return self.written[path]

    def write_file(self, path, content):
        if self.write_error is not None:
            raise self.write_error
        self.written[path] = content

    def close(self):
        pass


def _tool(sb):
    tools = {t.name: t for t in builtin_tools(sb)}
    assert "python_scratchpad" in tools
    return tools["python_scratchpad"]


def test_writes_code_to_scratchpad_and_runs_it():
    sb = MockSandbox(ExecResult(stdout="hello\n", stderr="", exit_code=0))
    out = _tool(sb).invoke({"code": "print('hello')"})
    assert sb.written["_scratchpad.py"] == "print('hello')"
    assert sb.last_cmd == "python3 _scratchpad.py"
    assert out == "$ python3 _scratchpad.py\nhello\n\n[exit 0]"


def test_formats_stderr_and_nonzero_exit():
    sb = MockSandbox(ExecResult(stdout="partial", stderr="boom", exit_code=2))
    out = _tool(sb).invoke({"code": "raise SystemExit(2)"})
    assert out.startswith("$ python3 _scratchpad.py\npartial")
    assert "[stderr]\nboom" in out
    assert "[exit 2]" in out


def test_truncated_flag_shown():
    sb = MockSandbox(ExecResult(stdout="x", stderr="", exit_code=0, truncated=True))
    out = _tool(sb).invoke({"code": "pass"})
    assert out.endswith("[exit 0] (truncated)")


def test_empty_output_still_formatted():
    sb = MockSandbox(ExecResult(stdout="", stderr="", exit_code=0))
    out = _tool(sb).invoke({"code": "x = 1"})
    assert out == "$ python3 _scratchpad.py\n\n[exit 0]"


def test_write_failure_surfaced_as_error():
    sb = MockSandbox(write_error=OSError("disk full"))
    out = _tool(sb).invoke({"code": "x = 1"})
    assert out.startswith("error:")
    assert "disk full" in out
    assert sb.last_cmd is None  # never got to run


def test_docstring_guides_model_use():
    doc = _tool(MockSandbox()).description
    assert "stdlib" in doc.lower()
