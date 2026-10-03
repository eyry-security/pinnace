"""Sandbox tests run against LocalSandbox only — no Docker needed."""

import pytest

from pinnace.sandbox import LocalSandbox, SandboxError


@pytest.fixture()
def sb(tmp_path):
    return LocalSandbox(str(tmp_path), unsafe_ok=True)


def test_requires_unsafe_ok(tmp_path):
    with pytest.raises(SandboxError):
        LocalSandbox(str(tmp_path))


def test_exec_echo(sb):
    r = sb.exec("echo hello")
    assert r.exit_code == 0
    assert r.stdout.strip() == "hello"


def test_exec_failure(sb):
    r = sb.exec("exit 3")
    assert r.exit_code == 3


def test_exec_timeout(sb):
    r = sb.exec("sleep 5", timeout=0.5)
    assert r.exit_code == 124


def test_write_and_read(sb):
    sb.write_file("a/b.txt", "contents here")
    assert sb.read_file("a/b.txt") == "contents here"


def test_read_missing(sb):
    with pytest.raises(SandboxError):
        sb.read_file("nope.txt")


def test_path_traversal_rejected(sb):
    with pytest.raises(SandboxError):
        sb.write_file("../../evil.txt", "x")
    with pytest.raises(SandboxError):
        sb.read_file("/etc/passwd")


def test_output_truncated(sb):
    r = sb.exec("python3 -c \"print('x' * 300000)\"")
    assert r.truncated
    assert len(r.stdout) < 300000
