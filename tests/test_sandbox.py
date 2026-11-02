"""Sandbox tests run against LocalSandbox only — no Docker needed."""

import pytest

from pinnace.sandbox import LocalSandbox, SandboxError


@pytest.fixture()
def sb(tmp_path):
    return LocalSandbox(str(tmp_path), unsafe_ok=True)


def test_requires_unsafe_ok(tmp_path):
    with pytest.raises(SandboxError):
        LocalSandbox(str(tmp_path))


def test_workdir_defaults_to_tmpdir():
    sb = LocalSandbox(unsafe_ok=True)
    import os

    assert os.path.isdir(sb.workdir)
    assert "pinnace-" in sb.workdir


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


def _docker_sb(out):
    """DockerSandbox with a fake container returning a canned demux pair."""
    from pinnace.sandbox import DockerSandbox

    class FakeContainer:
        def exec_run(self, *a, **k):
            return 0, out

    sb = DockerSandbox.__new__(DockerSandbox)
    sb._container = FakeContainer()
    return sb


def test_docker_exec_none_streams():
    """Regression: demux=True returns None for empty streams (or no output).

    Previously crashed with 'NoneType' object has no attribute 'decode',
    breaking every shell call in the Docker sandbox.
    """
    for out in [(b"hello\n", None), (None, b"oops\n"), (None, None), None]:
        r = _docker_sb(out).exec("echo hello")
        assert r.exit_code == 0


def test_docker_exec_none_stream_values():
    r = _docker_sb((b"hello\n", None)).exec("echo hello")
    assert r.stdout.strip() == "hello"
    assert r.stderr == ""
    r = _docker_sb((None, b"oops\n")).exec("echo hello")
    assert r.stdout == ""
    assert r.stderr.strip() == "oops"


def test_docker_sandbox_sets_memory_and_cpu_caps(monkeypatch):
    import sys
    import types

    from pinnace.sandbox import DockerSandbox

    calls = []

    class Containers:
        def run(self, **kwargs):
            calls.append(kwargs)
            return object()

    class Client:
        containers = Containers()

        def ping(self):
            return True

    monkeypatch.setitem(
        sys.modules, "docker", types.SimpleNamespace(from_env=lambda: Client()))
    monkeypatch.delenv("PINNACE_SANDBOX_MEMORY", raising=False)
    monkeypatch.delenv("PINNACE_SANDBOX_CPUS", raising=False)

    DockerSandbox(image="test-image")

    assert calls
    assert calls[0]["mem_limit"] == "1g"
    assert calls[0]["nano_cpus"] == 1_000_000_000


def test_docker_sandbox_caps_are_env_configurable(monkeypatch):
    import sys
    import types

    from pinnace.sandbox import DockerSandbox

    calls = []

    class Containers:
        def run(self, **kwargs):
            calls.append(kwargs)
            return object()

    class Client:
        containers = Containers()

        def ping(self):
            return True

    monkeypatch.setitem(
        sys.modules, "docker", types.SimpleNamespace(from_env=lambda: Client()))
    monkeypatch.setenv("PINNACE_SANDBOX_MEMORY", "512m")
    monkeypatch.setenv("PINNACE_SANDBOX_CPUS", "0.5")

    DockerSandbox(image="test-image")

    assert calls[0]["mem_limit"] == "512m"
    assert calls[0]["nano_cpus"] == 500_000_000
