"""Policy enforcement for agent tools.

Defense-in-depth on top of sandboxing: the Docker sandbox is the real
isolation boundary, but these checks catch accidents (and malice) earlier,
with clear error messages the model can learn from. Matters most for
LocalSandbox, which runs on the host.
"""

from __future__ import annotations

import re

# Max bytes per write_file call. Prevents disk-fill accidents.
MAX_WRITE_BYTES = 1_000_000

# Shell patterns that are never allowed, even inside the sandbox.
# Kept tight to avoid false positives on legitimate recon commands.
# Each is a regex searched against the full command string.
BLOCKED_SHELL_PATTERNS: list[tuple[str, str]] = [
    # Fork bomb
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?", "fork bomb"),
    # rm -r[f] aimed at /, /*, or ~ — the catastrophic cases.
    # (rm -rf ./build is fine and not matched.)
    (
        r"\brm\s+[^\n;|&]*-[a-zA-Z]*r[a-zA-Z]*[^\n;|&]*"
        r"(\s/(\s|$)|/\*(\s|$)|\s~(\s|/|$))",
        "recursive delete of system path",
    ),
    # Filesystem / device destruction
    (r"\b(mkfs|mkswap)\b", "filesystem formatting"),
    (r"\bdd\b[^\n;|&]*\bof=/dev/", "raw device write via dd"),
    (r">\s*/dev/(sd|hd|vd|nvme)", "redirect to block device"),
    # Host power control (matters for LocalSandbox)
    (r"\b(shutdown|reboot|halt|poweroff)\b", "host power control"),
]


class ToolPolicyError(ValueError):
    """A tool call was rejected by policy."""


def check_shell(command: str) -> None:
    """Raise ToolPolicyError if the command matches a blocked pattern."""
    for pattern, label in BLOCKED_SHELL_PATTERNS:
        if re.search(pattern, command):
            raise ToolPolicyError(f"blocked by policy ({label})")


def check_write(path: str, content: str) -> None:
    """Raise ToolPolicyError if the write violates policy."""
    size = len(content.encode("utf-8", "replace"))
    if size > MAX_WRITE_BYTES:
        raise ToolPolicyError(
            f"write too large ({size} bytes > {MAX_WRITE_BYTES} byte limit)"
        )
