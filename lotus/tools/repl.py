"""A sandboxed Python REPL tool for agentic operators.

The REPL is exposed to the agent as a single tool: it receives Python ``code`` and
returns captured stdout/stderr. Execution runs behind a :class:`Sandbox` interface so
the backend is swappable and mockable in tests:

- ``LocalSandbox``  — runs code in a restricted subprocess + temp dir. Zero infra, so
  it works everywhere (demos, CI). Weaker isolation — for trusted use.
- ``DockerSandbox`` — runs code in an ephemeral container. Stronger isolation, closer
  to Devin's verify sandbox. Requires Docker.

Design decision (see notes/notes.md): Docker is the intended production default; the
local backend exists so the minimal implementation is runnable/testable without Docker.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .base import Tool


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int

    def as_text(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr.rstrip()}")
        if self.exit_code != 0 and not self.stderr:
            parts.append(f"[exit code {self.exit_code}]")
        return "\n".join(parts) if parts else "(no output)"


class Sandbox(Protocol):
    """A backend that can execute a snippet of Python and return its output."""

    def run_code(self, code: str, files: dict[str, str] | None = None) -> ExecResult: ...


class LocalSandbox:
    """Run code in a subprocess inside a fresh temp dir. Works without Docker."""

    def __init__(self, timeout: int = 30, python: str | None = None):
        self.timeout = timeout
        self.python = python or sys.executable

    def run_code(self, code: str, files: dict[str, str] | None = None) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="lotus_repl_") as workdir:
            wd = Path(workdir)
            for rel, content in (files or {}).items():
                p = wd / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            script = wd / "_cell.py"
            script.write_text(code)
            try:
                proc = subprocess.run(
                    [self.python, str(script)],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return ExecResult(stdout="", stderr=f"Execution timed out after {self.timeout}s", exit_code=124)
            return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


class DockerSandbox:
    """Run code in an ephemeral Docker container (stronger isolation).

    Kept intentionally simple for the first cut: one ``docker run`` per execution,
    network disabled by default. Requires the ``docker`` CLI on PATH.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        network: bool = False,
        packages: list[str] | None = None,
    ):
        self.image = image
        self.timeout = timeout
        self.network = network
        self.packages = packages or []

    def run_code(self, code: str, files: dict[str, str] | None = None) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="lotus_repl_docker_") as workdir:
            wd = Path(workdir)
            for rel, content in (files or {}).items():
                p = wd / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            (wd / "_cell.py").write_text(code)
            pip = f"pip install -q {' '.join(self.packages)} && " if self.packages else ""
            cmd = [
                "docker", "run", "--rm",
                "--network", "bridge" if self.network else "none",
                "-v", f"{workdir}:/work", "-w", "/work",
                self.image, "sh", "-c", f"{pip}python _cell.py",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 60)
            except subprocess.TimeoutExpired:
                return ExecResult(stdout="", stderr=f"Execution timed out after {self.timeout}s", exit_code=124)
            except FileNotFoundError:
                return ExecResult(stdout="", stderr="Docker not available on PATH.", exit_code=127)
            return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


class _REPLArgs(BaseModel):
    code: str = Field(..., description="Python code to execute. Use print() to return results.")


class PythonREPLTool(Tool):
    """A Python REPL the agent can call to compute, parse files, etc."""

    name = "python_repl"
    description = (
        "Execute Python code in a sandbox and return its stdout/stderr. "
        "Use print() to emit results you want back. State does not persist between calls."
    )
    args_schema = _REPLArgs

    def __init__(self, sandbox: Sandbox | None = None, **local_kwargs):
        # Default to the local sandbox so it runs without Docker; pass
        # sandbox=DockerSandbox(...) for container isolation.
        self.sandbox: Sandbox = sandbox or LocalSandbox(**local_kwargs)

    def run(self, code: str) -> str:  # type: ignore[override]
        return self.sandbox.run_code(code).as_text()
