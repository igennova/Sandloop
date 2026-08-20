"""
Phase 1 — Docker sandbox runner.

Executes a snippet of code inside a throwaway Docker container and returns its
stdout, stderr and exit code. There is no agent loop yet (Phase 3) and no
isolation hardening yet (Phase 2); this layer only has to be correct, bounded,
and leak nothing.

    pip install -r requirements.txt

Requires a running Docker daemon.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker
import requests.exceptions
from docker.errors import DockerException, ImageNotFound, NotFound

DEFAULT_TIMEOUT_SECONDS = 30.0

# Guard against a script that prints forever. The container is capped by
# cgroups in Phase 2; this cap protects *this* process from decoding a
# multi-gigabyte log into memory.
MAX_OUTPUT_BYTES = 64 * 1024
TRUNCATION_NOTICE = "\n... [output truncated by sandbox]"


@dataclass(frozen=True)
class LanguageSpec:
    image: str
    filename: str
    command: list[str]


SUPPORTED_LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        image="python:3.12-slim",
        filename="script.py",
        command=["python", "/code/script.py"],
    ),
    "node": LanguageSpec(
        image="node:20-slim",
        filename="script.js",
        command=["node", "/code/script.js"],
    ),
}


@dataclass(frozen=True)
class SandboxLimits:
    """
    Container-level isolation applied to every execution.

    These are enforced by the kernel through Docker. The wall-clock timeout is
    deliberately not here: it is enforced by this process rather than by the
    container, so it stays a `run_code()` argument.
    """

    network: bool = False
    memory: str = "256m"
    cpus: float = 0.5

    def to_run_kwargs(self) -> dict[str, object]:
        return {
            "network_mode": "bridge" if self.network else "none",
            "mem_limit": self.memory,
            # Matching swap to the memory ceiling stops the container from
            # buying itself extra room in swap once RAM is exhausted.
            "memswap_limit": self.memory,
            "nano_cpus": int(self.cpus * 1_000_000_000),
        }


DEFAULT_LIMITS = SandboxLimits()


class SandboxError(RuntimeError):
    """The sandbox itself failed. Code that merely exits non-zero is not an error."""


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _resolve_language(language: str) -> LanguageSpec:
    key = language.lower().strip()
    if key not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language {language!r}. Supported: {supported}")
    return SUPPORTED_LANGUAGES[key]


def _get_client() -> docker.DockerClient:
    try:
        return docker.from_env()
    except DockerException as exc:
        raise SandboxError(
            "Could not reach the Docker daemon — is Docker running? "
            f"(underlying error: {exc})"
        ) from exc


def _ensure_image(client: docker.DockerClient, image: str) -> None:
    """Pull `image` if it is not already local, so the first run is not a silent stall."""
    try:
        client.images.get(image)
        return
    except ImageNotFound:
        pass
    except DockerException as exc:
        raise SandboxError(f"Could not inspect local images: {exc}") from exc

    try:
        client.images.pull(image)
    except DockerException as exc:
        raise SandboxError(f"Could not pull image {image!r}: {exc}") from exc


def _decode(raw: bytes) -> str:
    if len(raw) > MAX_OUTPUT_BYTES:
        return raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace") + TRUNCATION_NOTICE
    return raw.decode("utf-8", errors="replace")


def _kill_quietly(container) -> None:
    try:
        container.kill()
    except (DockerException, NotFound):
        pass


def _cleanup(container, workdir: Path | None) -> None:
    """Best-effort teardown. Must never raise, or it will mask the real failure."""
    if container is not None:
        try:
            container.remove(force=True)
        except (DockerException, NotFound):
            pass
    if workdir is not None:
        shutil.rmtree(workdir, ignore_errors=True)


def run_code(
    code: str,
    language: str = "python",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    limits: SandboxLimits = DEFAULT_LIMITS,
) -> ExecutionResult:
    """
    Run `code` in a fresh container and return what it printed.

    The container is always removed and the staging directory always deleted,
    including on timeout or Docker failure. A non-zero `exit_code` is a normal
    result; only sandbox-level failures raise (`SandboxError`).
    """
    spec = _resolve_language(language)
    client = _get_client()
    _ensure_image(client, spec.image)

    container = None
    workdir: Path | None = None
    started_at = time.monotonic()
    timed_out = False

    try:
        workdir = Path(tempfile.mkdtemp(prefix="sandloop-"))
        (workdir / spec.filename).write_text(code, encoding="utf-8")

        container = client.containers.run(
            image=spec.image,
            command=spec.command,
            volumes={str(workdir): {"bind": "/code", "mode": "ro"}},
            working_dir="/code",
            detach=True,
            **limits.to_run_kwargs(),
        )

        try:
            wait_result = container.wait(timeout=timeout_seconds)
            exit_code = int(wait_result.get("StatusCode", 1))
        except requests.exceptions.RequestException:
            # docker-py surfaces a read timeout as a requests error; the
            # container is still running, so stop it before reading logs.
            timed_out = True
            exit_code = -1
            _kill_quietly(container)

        return ExecutionResult(
            stdout=_decode(container.logs(stdout=True, stderr=False)),
            stderr=_decode(container.logs(stdout=False, stderr=True)),
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started_at,
            timed_out=timed_out,
        )
    except DockerException as exc:
        raise SandboxError(f"Docker execution failed: {exc}") from exc
    finally:
        _cleanup(container, workdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a code snippet inside a throwaway Docker container."
    )
    parser.add_argument("file", nargs="?", help="file to run; reads stdin when omitted")
    parser.add_argument(
        "-l", "--language", default="python", choices=sorted(SUPPORTED_LANGUAGES)
    )
    parser.add_argument(
        "-t", "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"wall-clock seconds before the container is killed (default {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--allow-network", action="store_true",
        help="give the container egress (off by default)",
    )
    args = parser.parse_args(argv)

    code = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()

    try:
        result = run_code(
            code,
            language=args.language,
            timeout_seconds=args.timeout,
            limits=SandboxLimits(network=args.allow_network),
        )
    except (SandboxError, ValueError) as exc:
        print(f"sandbox error: {exc}", file=sys.stderr)
        return 2

    sys.stdout.write(result.stdout)
    sys.stdout.flush()
    sys.stderr.write(result.stderr)
    print(
        f"[exit={result.exit_code} timed_out={result.timed_out} "
        f"duration={result.duration_seconds:.2f}s]",
        file=sys.stderr,
    )
    return 1 if result.exit_code < 0 else result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
