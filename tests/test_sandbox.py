"""
Phase 1 tests. These are integration tests: they need a running Docker daemon
and the python:3.12-slim image (pulled automatically on first run).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sandbox import (
    MAX_OUTPUT_BYTES,
    SUPPORTED_LANGUAGES,
    SandboxLimits,
    TRUNCATION_NOTICE,
    ExecutionResult,
    _resolve_language,
    run_code,
)

try:
    import docker

    _client = docker.from_env()
    _client.ping()
    DOCKER_UP = True
except Exception:  # noqa: BLE001 - any failure means "no daemon to test against"
    DOCKER_UP = False

needs_docker = pytest.mark.skipif(not DOCKER_UP, reason="Docker daemon not available")


def _sandbox_tempdirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("sandloop-*"))


def _running_container_ids() -> set[str]:
    return {c.id for c in _client.containers.list(all=True)}


# --- language resolution (no Docker needed) ---------------------------------


def test_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported language"):
        run_code("print(1)", language="brainfuck")


def test_language_lookup_is_case_and_space_insensitive():
    assert _resolve_language("  PYTHON  ") is SUPPORTED_LANGUAGES["python"]


# --- happy path -------------------------------------------------------------


@needs_docker
def test_captures_stdout_and_zero_exit():
    result = run_code('print("hello from sandbox")')
    assert isinstance(result, ExecutionResult)
    assert result.stdout.strip() == "hello from sandbox"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.ok is True
    assert result.duration_seconds > 0


@needs_docker
def test_captures_stderr_separately():
    result = run_code('import sys; sys.stderr.write("boom\\n")')
    assert result.stdout == ""
    assert result.stderr.strip() == "boom"
    assert result.exit_code == 0


@needs_docker
def test_streams_do_not_bleed_into_each_other():
    result = run_code(
        'import sys\n'
        'sys.stdout.write("OUT\\n")\n'
        'sys.stderr.write("ERR\\n")\n'
    )
    assert "OUT" in result.stdout and "ERR" not in result.stdout
    assert "ERR" in result.stderr and "OUT" not in result.stderr


# --- failure is a result, not an exception ----------------------------------


@needs_docker
def test_nonzero_exit_is_reported_not_raised():
    result = run_code("raise SystemExit(3)")
    assert result.exit_code == 3
    assert result.ok is False


@needs_docker
def test_syntax_error_surfaces_traceback_on_stderr():
    result = run_code("def broken(:\n")
    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr


# --- bounds -----------------------------------------------------------------


@needs_docker
def test_large_output_is_truncated():
    result = run_code(f'print("x" * {MAX_OUTPUT_BYTES * 4})')
    assert result.stdout.endswith(TRUNCATION_NOTICE)
    assert len(result.stdout.encode()) <= MAX_OUTPUT_BYTES + len(TRUNCATION_NOTICE.encode())


@needs_docker
def test_infinite_loop_times_out_and_is_killed():
    result = run_code("while True:\n    pass\n", timeout_seconds=3)
    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.ok is False
    assert result.duration_seconds >= 3


# --- teardown guarantees ----------------------------------------------------


@needs_docker
def test_container_is_removed_after_success():
    before = _running_container_ids()
    run_code('print("ok")')
    assert _running_container_ids() == before


@needs_docker
def test_container_is_removed_after_timeout():
    before = _running_container_ids()
    run_code("while True:\n    pass\n", timeout_seconds=3)
    assert _running_container_ids() == before


@needs_docker
def test_staging_directory_is_deleted():
    before = _sandbox_tempdirs()
    run_code('print("ok")')
    assert _sandbox_tempdirs() == before


@needs_docker
def test_staging_directory_is_deleted_after_failure():
    before = _sandbox_tempdirs()
    run_code("raise SystemExit(1)")
    assert _sandbox_tempdirs() == before


# --- isolation smoke check (real hardening lands in Phase 2) ----------------


@needs_docker
def test_code_cannot_write_to_its_own_mount():
    result = run_code(
        'try:\n'
        '    open("/code/evil.txt", "w").write("x")\n'
        '    print("WROTE")\n'
        'except OSError as e:\n'
        '    print("BLOCKED", type(e).__name__)\n'
    )
    assert "BLOCKED" in result.stdout
    assert "WROTE" not in result.stdout


# --- Phase 2: network isolation ---------------------------------------------

NET_PROBE = """
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=4)
    print("REACHABLE")
except OSError as exc:
    print("BLOCKED", type(exc).__name__)
"""

DNS_PROBE = """
import socket
try:
    print("RESOLVED", socket.gethostbyname("example.com"))
except OSError as exc:
    print("BLOCKED", type(exc).__name__)
"""


@needs_docker
def test_network_is_disabled_by_default():
    result = run_code(NET_PROBE, timeout_seconds=20)
    assert "BLOCKED" in result.stdout
    assert "REACHABLE" not in result.stdout


@needs_docker
def test_dns_resolution_is_disabled_by_default():
    result = run_code(DNS_PROBE, timeout_seconds=20)
    assert "BLOCKED" in result.stdout


@needs_docker
def test_network_can_be_opted_into_explicitly():
    result = run_code(NET_PROBE, timeout_seconds=20, limits=SandboxLimits(network=True))
    assert "REACHABLE" in result.stdout
