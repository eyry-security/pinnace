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
