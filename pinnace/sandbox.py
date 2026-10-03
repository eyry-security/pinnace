"""Execution sandboxes. Agent tools run inside one of these.

DockerSandbox is the real thing: each instance is a throwaway container, files
live in /work, and you can cut network egress entirely. LocalSandbox runs on
your own machine — fine for dev and tests, never for untrusted model output.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Cap tool output so one chatty command can't eat the context window.
MAX_OUTPUT_CHARS = 200_000


class SandboxError(RuntimeError):
    """Raised when a sandbox can't do what was asked."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n…[truncated at {MAX_OUTPUT_CHARS} chars]", True
    return text, False


class Sandbox(ABC):
    """Where agent tools execute."""

    @abstractmethod
    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        """Run a shell command, return the result."""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a file (path relative to the workdir)."""

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write a file (path relative to the workdir)."""

    @abstractmethod
    def close(self) -> None:
        """Tear the sandbox down."""


def _safe_rel(path: str) -> str:
    """Keep file access inside the workdir. Rejects absolute paths and '..'."""
    rel = os.path.normpath(path).lstrip("/")
    if rel.startswith("..") or os.path.isabs(path):
        raise SandboxError(f"path escapes workdir: {path!r}")
    return rel


class DockerSandbox(Sandbox):
    """One throwaway container per instance. Files live in /work.

    Pass network="none" to cut container egress entirely. The image needs a
    shell and python; python:3.12-slim is a fine default.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        workdir: str = "/work",
        network: str | None = None,
        mem_limit: str = "1g",
    ) -> None:
        try:
            import docker
        except ImportError as e:
            raise SandboxError("the 'docker' package isn't installed (pip install docker)") from e
        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as e:
            raise SandboxError(f"docker daemon not reachable: {e}") from e
        self.workdir = workdir
        run_kwargs: dict = {
            "image": image,
            "command": "sleep infinity",
            "detach": True,
            "working_dir": workdir,
            "mem_limit": mem_limit,
            "stdin_open": True,
        }
        if network is not None:
            run_kwargs["network_mode"] = network
        try:
            self._container = self._client.containers.run(**run_kwargs)
        except Exception as e:
            raise SandboxError(f"couldn't start {image}: {e}") from e

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        # No timeout knob on exec_run; enforce it with the timeout(1) wrapper.
        wrapped = f"timeout {int(timeout)}s bash -c {shlex.quote(cmd)}"
        try:
            code, out = self._container.exec_run(wrapped, demux=True)
        except Exception as e:
            raise SandboxError(f"exec failed: {e}") from e
        stdout_b, stderr_b = out or (b"", b"")
        stdout, t1 = _truncate(stdout_b.decode("utf-8", "replace"))
        stderr, t2 = _truncate(stderr_b.decode("utf-8", "replace"))
        return ExecResult(stdout, stderr, code, truncated=t1 or t2)

    def _tar_for(self, rel: str, content: bytes) -> io.BytesIO:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=rel)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)
        return buf

    def write_file(self, path: str, content: str) -> None:
        rel = _safe_rel(path)
        # put_archive needs the parent dir; create it first.
        parent = os.path.dirname(f"{self.workdir}/{rel}") or self.workdir
        self.exec(f"mkdir -p {shlex.quote(parent)}")
        ok = self._container.put_archive(self.workdir, self._tar_for(rel, content.encode()))
        if not ok:
            raise SandboxError(f"put_archive failed for {path!r}")

    def read_file(self, path: str) -> str:
        rel = _safe_rel(path)
        try:
            chunks, _stat = self._container.get_archive(f"{self.workdir}/{rel}")
        except Exception as e:
            raise SandboxError(f"can't read {path!r}: {e}") from e
        buf = io.BytesIO(b"".join(chunks))
        with tarfile.open(fileobj=buf) as tar:
            member = tar.next()
            if member is None:
                raise SandboxError(f"empty archive for {path!r}")
            f = tar.extractfile(member)
            if f is None:
                raise SandboxError(f"can't extract {path!r}")
            return f.read().decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self._container.remove(force=True)
        except Exception:
            pass


class LocalSandbox(Sandbox):
    """Runs commands on YOUR machine. Dev and tests only.

    Never point this at untrusted model output. You must pass unsafe_ok=True
    to prove you read this docstring.
    """

    def __init__(self, workdir: str, unsafe_ok: bool = False) -> None:
        if not unsafe_ok:
            raise SandboxError(
                "LocalSandbox runs model-generated commands on your machine. "
                "Pass unsafe_ok=True only for dev/tests."
            )
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        try:
            p = subprocess.run(
                cmd, shell=True, cwd=self.workdir,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            out, t1 = _truncate((e.stdout or "") if isinstance(e.stdout, str) else "")
            err, t2 = _truncate((e.stderr or "") if isinstance(e.stderr, str) else "")
            return ExecResult(out, err, 124, truncated=True or t1 or t2)
        stdout, t1 = _truncate(p.stdout)
        stderr, t2 = _truncate(p.stderr)
        return ExecResult(stdout, stderr, p.returncode, truncated=t1 or t2)

    def read_file(self, path: str) -> str:
        rel = _safe_rel(path)
        full = os.path.join(self.workdir, rel)
        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as e:
            raise SandboxError(f"can't read {path!r}: {e}") from e

    def write_file(self, path: str, content: str) -> None:
        rel = _safe_rel(path)
        full = os.path.join(self.workdir, rel)
        os.makedirs(os.path.dirname(full) or self.workdir, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def close(self) -> None:
        pass
