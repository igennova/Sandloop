"""
Phase 1 — naive Docker sandbox runner (no agent, no hardening yet).

Dependency: pip install docker
Requires: Docker daemon running locally.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import DockerException


SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "python": {
        "image": "python:3.12-slim",
        "filename": "script.py",
        "command": ["python", "/code/script.py"],
    },
}


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float


def run_code(code: str, language: str) -> ExecutionResult:
    """
    Run `code` inside a fresh Docker container and return captured output.

    Spins up a container, executes the code once, then kills and removes it.
    """
    language_key = language.lower().strip()
    if language_key not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language {language!r}. Supported: {supported}")

    spec = SUPPORTED_LANGUAGES[language_key]
    container = None
    workdir: Path | None = None
    started_at = time.monotonic()

    try:
        client = docker.from_env()

        workdir = Path(tempfile.mkdtemp(prefix="sandloop-"))
        script_path = workdir / spec["filename"]
        script_path.write_text(code, encoding="utf-8")

        container = client.containers.run(
            image=spec["image"],
            command=spec["command"],
            volumes={str(workdir): {"bind": "/code", "mode": "ro"}},
            detach=True,
        )

        wait_result = container.wait()
        exit_code = int(wait_result.get("StatusCode", 1))

        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started_at,
        )
    except DockerException as exc:
        raise RuntimeError(f"Docker execution failed: {exc}") from exc
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except DockerException:
                pass

        if workdir is not None:
            for path in workdir.iterdir():
                path.unlink(missing_ok=True)
            workdir.rmdir()


if __name__ == "__main__":
    result = run_code('print("hello from sandbox")\n', language="python")
    print("stdout:", result.stdout.strip())
    print("stderr:", result.stderr.strip() or "(empty)")
    print("exit_code:", result.exit_code)
    print("duration_seconds:", round(result.duration_seconds, 3))
