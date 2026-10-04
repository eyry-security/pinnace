
"""Tests for tool policy enforcement."""

import pytest

from pinnace.tool_policy import (
    MAX_WRITE_BYTES,
    ToolPolicyError,
    check_shell,
    check_write,
)


def test_shell_allows_normal_commands():
    for cmd in [
        "echo hello",
        "curl -s http://example.com/ | head -20",
        "rm -rf ./build",
        "rm -rf /tmp/work",
        "ls -la /etc/hosts",
        "python3 -c 'print(1)'",
        "nmap -sV 127.0.0.1",
    ]:
        check_shell(cmd)  # must not raise


def test_shell_blocks_fork_bomb():
    with pytest.raises(ToolPolicyError, match="fork bomb"):
        check_shell(":(){ :|:& };:")


def test_shell_blocks_rm_rf_root():
    for cmd in ["rm -rf /", "rm -rf /*", "rm -fr ~", "sudo rm -r /"]:
        with pytest.raises(ToolPolicyError, match="recursive delete"):
            check_shell(cmd)


def test_shell_blocks_device_destruction():
    with pytest.raises(ToolPolicyError):
        check_shell("mkfs.ext4 /dev/sda1")
    with pytest.raises(ToolPolicyError):
        check_shell("dd if=/dev/zero of=/dev/sda bs=1M")
    with pytest.raises(ToolPolicyError):
        check_shell("echo hi > /dev/sda")


def test_shell_blocks_power_control():
    with pytest.raises(ToolPolicyError):
        check_shell("sudo reboot")


def test_write_allows_normal():
    check_write("out.txt", "hello")


def test_write_blocks_oversize():
    with pytest.raises(ToolPolicyError, match="too large"):
        check_write("big.bin", "x" * (MAX_WRITE_BYTES + 1))


def test_write_boundary():
    check_write("ok.bin", "x" * MAX_WRITE_BYTES)  # exactly at limit: fine


def test_policy_errors_surface_as_tool_errors(tmp_path):
    """Policy rejections come back as error strings, not exceptions."""
    from pinnace.sandbox import LocalSandbox
    from pinnace.tools import builtin_tools

    sb = LocalSandbox(str(tmp_path), unsafe_ok=True)
    tools = {t.name: t for t in builtin_tools(sb)}
    out = tools["shell"].invoke({"command": "rm -rf /"})
    assert out.startswith("error: blocked by policy")
    out = tools["write_file"].invoke(
        {"path": "big.txt", "content": "x" * (MAX_WRITE_BYTES + 1)}
    )
    assert out.startswith("error: write too large")
