"""Tool registry and base protocol.

A tool is anything the agent can invoke between turns. Each tool declares its
name, a description (shown to the LLM), a JSON-schema for its parameters, and
an ``execute`` method that does the work and returns a string result.

Built-in tools:
  * ``shell`` — run a command inside the Docker sandbox
  * ``read_file`` — read a file from the sandbox filesystem
  * ``write_file`` — write a file into the sandbox filesystem

Callers (Aplomado, Quarterdeck) register domain-specific tools on top.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    """What a tool invocation returns to the agent loop."""
    output: str
    ok: bool = True

    def to_message(self) -> str:
        if self.ok:
            return self.output
        return f"[error] {self.output}"


class Tool(ABC):
    """Base class for a tool the agent can invoke."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema ``properties`` for the tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def openai_schema(self) -> dict:
        """Return the tool in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class ToolRegistry:
    """Name-indexed collection of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# --------------------------------------------------------------------------- #
# Built-in tools (sandbox-backed; sandbox is injected at agent start)
# --------------------------------------------------------------------------- #

class ShellTool(Tool):
    """Run a shell command inside the Docker sandbox."""

    name = "shell"
    description = (
        "Execute a shell command inside a sandboxed Docker container. "
        "Returns stdout and stderr. Use for running scripts, installing "
        "packages, or any CLI operation."
    )
    parameters = {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
    }

    def __init__(self, sandbox: Any = None) -> None:
        self._sandbox = sandbox

    def bind(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def execute(self, **kwargs) -> ToolResult:
        if self._sandbox is None:
            return ToolResult("sandbox not initialised", ok=False)
        command = kwargs.get("command", "")
        if not command:
            return ToolResult("no command given", ok=False)
        return self._sandbox.exec(command)


class ReadFileTool(Tool):
    """Read a file from the sandbox filesystem."""

    name = "read_file"
    description = "Read the contents of a file inside the sandbox container."
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path to the file inside the container.",
        },
    }

    def __init__(self, sandbox: Any = None) -> None:
        self._sandbox = sandbox

    def bind(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def execute(self, **kwargs) -> ToolResult:
        if self._sandbox is None:
            return ToolResult("sandbox not initialised", ok=False)
        path = kwargs.get("path", "")
        if not path:
            return ToolResult("no path given", ok=False)
        return self._sandbox.exec(f"cat {path!r}")


class WriteFileTool(Tool):
    """Write a file into the sandbox filesystem."""

    name = "write_file"
    description = "Write content to a file inside the sandbox container. Creates parent directories."
    parameters = {
        "path": {
            "type": "string",
            "description": "Absolute path to the file inside the container.",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file.",
        },
    }

    def __init__(self, sandbox: Any = None) -> None:
        self._sandbox = sandbox

    def bind(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def execute(self, **kwargs) -> ToolResult:
        if self._sandbox is None:
            return ToolResult("sandbox not initialised", ok=False)
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not path:
            return ToolResult("no path given", ok=False)
        # Use heredoc to avoid quoting issues with the content.
        escaped = content.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`")
        cmd = f"mkdir -p $(dirname {path!r}) && cat > {path!r} << 'PINNACE_EOF'\n{escaped}\nPINNACE_EOF"
        return self._sandbox.exec(cmd)


def default_tools() -> list[Tool]:
    """The built-in tool set (sandbox binding happens later)."""
    return [ShellTool(), ReadFileTool(), WriteFileTool()]
