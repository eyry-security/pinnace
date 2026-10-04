"""Security policy tests for host-side built-in tools."""

import socket

import pytest

from pinnace import tools as tools_module
from pinnace.sandbox import LocalSandbox
from pinnace.tools import _PinnedHTTPConnection, builtin_tools


def _fetch_tool(tmp_path):
    sandbox = LocalSandbox(str(tmp_path), unsafe_ok=True)
    return next(tool for tool in builtin_tools(sandbox) if tool.name == "fetch_url")


def _address(ip, port=443):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
    return (family, socket.SOCK_STREAM, 6, "", sockaddr)


class FakeResponse:
    def __init__(self, status=200, body=b"public body", location=None, reason="OK"):
        self.status = status
        self.body = body
        self.location = location
        self.reason = reason

    def getheader(self, name):
        return self.location if name.lower() == "location" else None

    def read(self, limit):
        return self.body[:limit]


class FakeConnection:
    responses = []
    calls = []

    def __init__(self, host, port, addresses, timeout):
        self.host = host
        self.port = port
        self.addresses = addresses
        self.timeout = timeout
        type(self).calls.append(self)

    def request(self, method, path, headers):
        self.request_args = (method, path, headers)

    def getresponse(self):
        return type(self).responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_connection():
    FakeConnection.responses = []
    FakeConnection.calls = []


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://service.localhost/admin",
        "http://user@example.test/",
        "http://[::1]/",
    ],
)
def test_fetch_url_rejects_unsafe_urls(monkeypatch, tmp_path, url):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [_address("::1")],
    )

    result = _fetch_tool(tmp_path).invoke({"url": url})

    assert result.startswith("error:")


def test_fetch_url_rejects_private_dns_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [_address("10.0.0.8")],
    )

    result = _fetch_tool(tmp_path).invoke({"url": "https://example.test/"})

    assert "private destinations" in result


def test_fetch_url_rejects_mixed_public_and_private_dns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            _address("93.184.216.34"),
            _address("127.0.0.1"),
        ],
    )

    result = _fetch_tool(tmp_path).invoke({"url": "https://example.test/"})

    assert "private destinations" in result


def test_fetch_url_allows_public_destination(monkeypatch, tmp_path):
    resolved = [_address("93.184.216.34")]
    resolve_calls = []

    def resolve(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return resolved

    FakeConnection.responses = [FakeResponse()]
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(tools_module, "_PinnedHTTPSConnection", FakeConnection)

    result = _fetch_tool(tmp_path).invoke({"url": "https://example.test/page?q=1"})

    assert result == "public body"
    assert len(resolve_calls) == 1
    connection = FakeConnection.calls[0]
    assert connection.addresses is resolved
    assert connection.request_args == (
        "GET",
        "/page?q=1",
        {"Host": "example.test", "User-Agent": "pinnace/0.1"},
    )
    assert connection.closed


def test_fetch_url_revalidates_redirect_destination(monkeypatch, tmp_path):
    FakeConnection.responses = [
        FakeResponse(status=302, location="http://127.0.0.1/private")
    ]

    def resolve(host, *args, **kwargs):
        ip = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
        return [_address(ip)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(tools_module, "_PinnedHTTPSConnection", FakeConnection)

    result = _fetch_tool(tmp_path).invoke({"url": "https://example.test/"})

    assert "private destinations" in result
    assert len(FakeConnection.calls) == 1
    assert FakeConnection.calls[0].closed


def test_pinned_connection_does_not_repeat_dns_resolution(monkeypatch):
    connected = []

    class FakeSocket:
        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, sockaddr):
            connected.append(sockaddr)

        def close(self):
            raise AssertionError("successful socket should remain open")

    def no_dns(*args, **kwargs):
        raise AssertionError("pinned connection must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    public_address = _address("93.184.216.34", 80)

    connection = _PinnedHTTPConnection(
        "rebind.example",
        80,
        [public_address],
        timeout=30,
    )
    connection.connect()

    assert connected == [("93.184.216.34", 80)]
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
