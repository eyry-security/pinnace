"""Built-in tools. Each one executes against a Sandbox; the agent never
touches the host directly (unless you hand it a LocalSandbox, which is on you).
"""

from __future__ import annotations

import json
import urllib.request

from langchain_core.tools import BaseTool, StructuredTool

from .sandbox import Sandbox

_FETCH_MAX_CHARS = 100_000


def _format_exec(cmd: str, r) -> str:
    out = f"$ {cmd}\n{r.stdout}"
    if r.stderr.strip():
        out += f"\n[stderr]\n{r.stderr}"
    out += f"\n[exit {r.exit_code}]"
    if r.truncated:
        out += " (truncated)"
    return out


def builtin_tools(sandbox: Sandbox) -> list[BaseTool]:
    """The default toolset: shell, read/write files in the sandbox, fetch a URL,
    and finish (end the run with a structured result)."""

    def shell(command: str) -> str:
        """Run a shell command inside the sandbox and return its output.

        Prefer this over guessing: explore the environment, run tools, check
        results. Long-running commands should background themselves."""
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
            sandbox.write_file(path, content)
            return f"wrote {len(content)} chars to {path}"
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"

    def fetch_url(url: str) -> str:
        """GET a URL and return the body as text (truncated at 100k chars).

        Runs from the host, not the sandbox — it's a plain GET, no JS."""
        if not url.startswith(("http://", "https://")):
            return "error: only http(s) URLs"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pinnace/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read(_FETCH_MAX_CHARS + 1).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001 - surface it to the model
            return f"error: {e}"
        if len(body) > _FETCH_MAX_CHARS:
            body = body[:_FETCH_MAX_CHARS] + "\n…[truncated]"
        return body

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
        StructuredTool.from_function(fetch_url, name="fetch_url",
            description="GET a URL and return the body as text. Plain GET, no JS."),
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
