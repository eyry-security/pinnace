"""pinnace: general multi-turn agent runtime with compaction, tools, and a Docker sandbox."""

from .agent import AgentResult, PinnaceAgent, PinnaceError
from .config import AgentConfig
from .context import Message
from .sandbox import DockerSandbox, ExecResult, LocalSandbox, Sandbox, SandboxError
from .session import SessionStore
from .tools import builtin_tools

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AgentConfig",
    "AgentResult",
    "DockerSandbox",
    "ExecResult",
    "LocalSandbox",
    "Message",
    "PinnaceAgent",
    "PinnaceError",
    "Sandbox",
    "SandboxError",
    "SessionStore",
    "builtin_tools",
]
