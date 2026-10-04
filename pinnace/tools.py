"""Built-in tools. Each one executes against a Sandbox; the agent never
touches the host directly (unless you hand it a LocalSandbox, which is on you).
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import urllib.parse

from langchain_core.tools import BaseTool, StructuredTool

from .sandbox import Sandbox
from .tool_policy import ToolPolicyError, check_shell, check_write

_FETCH_MAX_CHARS = 100_000
_FETCH_MAX_REDIRECTS = 10


def _resolve_public_url(url: str) -> tuple[urllib.parse.SplitResult, list[tuple]]:
    """Parse a URL and resolve its hostname once to validated public addresses."""
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"invalid URL: {e}") from e
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only http(s) URLs with a hostname are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("local and private destinations are not allowed")

    try:
        addresses = socket.getaddrinfo(
            host,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as e:
        raise ValueError(f"could not resolve hostname: {e}") from e
    if not addresses:
        raise ValueError("hostname did not resolve")

    for address in addresses:
        raw_ip = address[4][0].split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as e:
            raise ValueError("hostname resolved to an invalid address") from e
        if not ip.is_global:
            raise ValueError("local and private destinations are not allowed")
    return parsed, addresses


def _connect_validated(addresses: list[tuple], timeout: float) -> socket.socket:
    """Connect only to an already-resolved address, with no second DNS lookup."""
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            last_error = e
            sock.close()
    raise OSError("could not connect to destination") from last_error


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection pinned to addresses validated by _resolve_public_url."""

    def __init__(self, host: str, port: int, addresses: list[tuple], timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._addresses = addresses

    def connect(self) -> None:
        self.sock = _connect_validated(self._addresses, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Pinned HTTPS connection that still authenticates the original hostname."""

    def __init__(self, host: str, port: int, addresses: list[tuple], timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._addresses = addresses

    def connect(self) -> None:
        sock = _connect_validated(self._addresses, self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


def _fetch_public_url(url: str) -> str:
    """Fetch a URL through pinned public addresses, revalidating each redirect."""
    current_url = url
    for redirect_count in range(_FETCH_MAX_REDIRECTS + 1):
        parsed, addresses = _resolve_public_url(current_url)
        host = parsed.hostname
        assert host is not None  # guaranteed by _resolve_public_url
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection_type = (
            _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        )
        connection = connection_type(host, port, addresses, timeout=30)
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            connection.request(
                "GET",
                path,
                headers={"Host": parsed.netloc, "User-Agent": "pinnace/0.1"},
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise OSError("redirect response missing Location header")
                if redirect_count == _FETCH_MAX_REDIRECTS:
                    raise OSError("too many redirects")
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            if response.status >= 400:
                raise OSError(f"HTTP {response.status} {response.reason}")
            body = response.read(_FETCH_MAX_CHARS + 1).decode("utf-8", "replace")
        finally:
            connection.close()
        if len(body) > _FETCH_MAX_CHARS:
            body = body[:_FETCH_MAX_CHARS] + "\n…[truncated]"
        return body
    raise OSError("too many redirects")


def _format_exec(cmd: str, r) -> str:
    out = f"$ {cmd}\n{r.stdout}"
    if r.stderr.strip():
        out += f"\n[stderr]\n{r.stderr}"
    out += f"\n[exit {r.exit_code}]"
    if r.truncated:
        out += " (truncated)"
    return out


def builtin_tools(sandbox: Sandbox) -> list[BaseTool]:
    """The default toolset: shell, read/write files in the sandbox, a Python
    scratchpad for ad-hoc analysis, fetch a URL, and finish (end the run with
    a structured result)."""

    def shell(command: str) -> str:
        """Run a shell command inside the sandbox and return its output.

        Prefer this over guessing: explore the environment, run tools, check
        results. Long-running commands should background themselves."""
        try:
            check_shell(command)
        except ToolPolicyError as e:
            return f"error: {e}"
        return _format_exec(command, sandbox.exec(command))

    def read_file(path: str) -> str:
        """Read a file from the sandbox workdir. Path is relative, e.g. 'out.txt'."""
        try:
            return sandbox.read_file(path)
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"

    def write_file(path: str, content: str) -> str:
        """Write a file into the sandbox workdir. Creates parent dirs. Path is relative."""
        try:
            check_write(path, content)
        except ToolPolicyError as e:
            return f"error: {e}"
        try:
            sandbox.write_file(path, content)
            return f"wrote {len(content)} chars to {path}"
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"

    def fetch_url(url: str) -> str:
        """GET a public URL and return the body as text (truncated at 100k chars).

        Runs from the host, not the sandbox. Local/private destinations and
        redirects are rejected."""
        try:
            return _fetch_public_url(url)
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"

    def python_scratchpad(code: str) -> str:
        """Run Python code inside the sandbox and return its output.

        Use for custom analysis, data munging, or quick calculations during a
        run — parsing tool output, scoring candidates, transforming data —
        instead of wrestling with bash one-liners. The code is written to
        _scratchpad.py in the sandbox workdir and executed with python3.
        Stdlib only: no pip installs, no network guarantees."""
        try:
            sandbox.write_file("_scratchpad.py", code)
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"
        return _format_exec("python3 _scratchpad.py", sandbox.exec("python3 _scratchpad.py"))

    def finish(result: str) -> str:
        """End the run with a structured result. Pass a JSON object as a string,
        e.g. '{"verdict": "vulnerable", "notes": "..."}'. Call this when the job
        is done instead of just stopping."""
        return f"__PINNACE_FINISH__{result}"

    return [
        StructuredTool.from_function(shell, name="shell",
            description="Run a shell command inside the sandbox. Returns stdout, stderr, exit code."),
        StructuredTool.from_function(read_file, name="read_file",
            description="Read a file from the sandbox workdir (relative path)."),
        StructuredTool.from_function(write_file, name="write_file",
            description="Write a file into the sandbox workdir (relative path). Creates parent dirs."),
        StructuredTool.from_function(python_scratchpad, name="python_scratchpad",
            description="Run Python code (stdlib only) in the sandbox. Returns stdout, stderr, exit code."),
        StructuredTool.from_function(fetch_url, name="fetch_url",
            description="GET a public URL and return the body as text. Plain GET, no JS."),
        StructuredTool.from_function(finish, name="finish",
            description="End the run with a structured result: a JSON object as a string."),
    ]


FINISH_PREFIX = "__PINNACE_FINISH__"


def parse_finish(tool_output: str):
    """Pull the structured payload out of a finish() call.

    Returns None when the output isn't a finish marker. Otherwise returns the
    parsed JSON payload, or {"result": raw_string} when it isn't valid JSON —
    so callers always get a dict back (or None)."""
    if not tool_output.startswith(FINISH_PREFIX):
        return None
    raw = tool_output[len(FINISH_PREFIX):]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"result": raw}
