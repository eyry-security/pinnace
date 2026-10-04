"""Docker sandbox for tool execution.

Every command an agent's tools run goes through here. We spin up a single
container per agent session, run commands inside it via ``exec_run``, and
tear it down when the session ends. The container is:

  * ephemeral (removed on close)
  * resource-capped (memory limit, optional CPU shares)
  * network-isolated by default (``network_mode="none"``)
  * time-bounded (per-command timeout)

This keeps the host safe from whatever an LLM-driven agent decides to run.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from .tools import ToolResult

log = logging.getLogger(__name__)


class Sandbox:
    """A Docker container scoped to one agent session."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 120,
        mem_limit: str = "512m",
        network: bool = False,
        workdir: str = "/work",
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.network = network
        self.workdir = workdir

        self._client: Any = None
        self._container: Any = None

    @property
    def running(self) -> bool:
        return self._container is not None

    def start(self) -> None:
        """Create and start the container. Requires the ``docker`` package and
        a running Docker daemon."""
        import docker  # late import so the module loads without docker installed

        self._client = docker.from_env()
        network_mode = "bridge" if self.network else "none"
        self._container = self._client.containers.run(
            self.image,
            command="sleep infinity",
            detach=True,
            remove=True,
            working_dir=self.workdir,
            mem_limit=self.mem_limit,
            network_mode=network_mode,
            stdin_open=True,
        )
        log.info("sandbox started: container=%s image=%s", self._container.short_id, self.image)

    def exec(self, command: str) -> ToolResult:
        """Run a shell command inside the container, return stdout/stderr."""
        if self._container is None:
            return ToolResult("sandbox not started", ok=False)

        try:
            exit_code, output = self._container.exec_run(
                ["bash", "-c", command],
                workdir=self.workdir,
                demux=False,
                timeout=self.timeout,
            )
        except Exception as exc:
            return ToolResult(f"exec failed: {exc}", ok=False)

        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output or "")
        # Cap output to avoid blowing up the context window.
        max_chars = 50_000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"

        return ToolResult(output=text, ok=(exit_code == 0))

    def upload(self, local_path: str, container_path: str) -> None:
        """Copy a file from the host into the container."""
        import tarfile, io, os

        if self._container is None:
            raise RuntimeError("sandbox not started")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(local_path, arcname=os.path.basename(container_path))
        buf.seek(0)
        dest_dir = os.path.dirname(container_path) or "/"
        self._container.put_archive(dest_dir, buf)

    def close(self) -> None:
        """Stop and remove the container."""
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
            except Exception:
                try:
                    self._container.kill()
                except Exception:
                    pass
            self._container = None
            log.info("sandbox stopped")
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
