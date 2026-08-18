# sandloop

A sandboxed code-execution agent, built from scratch to learn agent architecture
and container isolation. An LLM writes code; the code runs inside a throwaway
Docker container instead of on the host.

This is a hand-built miniature of what E2B, Modal Sandboxes and Code Interpreter
run in production.

## Architecture

![Architecture diagram](docs/architecture.png)

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Docker runner — execute code, capture output, guarantee teardown | **done** |
| 2 | Isolation hardening — no network, cgroup limits, read-only rootfs, non-root | next |
| 3 | Agent loop — LLM calls the sandbox as a tool | |
| 4 | File I/O — a persistent per-session workspace | |
| 5 | Concurrency and a reaper for stale containers | |
| 6 | Adversarial testing — fork bombs, memory hogs, egress attempts | |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run a file, or pipe from stdin:

```bash
python sandbox.py examples/hello.py
echo 'print(6 * 7)' | python sandbox.py
python sandbox.py -l node -t 10 script.js
```

From Python:

```python
from sandbox import run_code

result = run_code('print("hi")')
print(result.stdout, result.exit_code, result.ok)
```

## The contract

`run_code(code, language="python", timeout_seconds=30) -> ExecutionResult`

| Field | Meaning |
|-------|---------|
| `stdout` / `stderr` | captured separately, never interleaved |
| `exit_code` | the process exit code; `-1` when killed by timeout |
| `duration_seconds` | wall clock, measured around the whole run |
| `timed_out` | `True` if the container was killed rather than exiting |
| `ok` | `exit_code == 0 and not timed_out` |

Two rules the rest of the system depends on:

**Code failing is a result, not an exception.** A syntax error, a non-zero exit
or a timeout all come back as a populated `ExecutionResult`. Only the sandbox
itself failing — no Docker daemon, unpullable image — raises `SandboxError`.
Phase 3's agent loop needs to feed failures back to the model, so they must be
data rather than control flow.

**Teardown always happens.** The container is force-removed and the staging
directory deleted in a `finally`, including on timeout and on Docker errors.
Cleanup is best-effort and never raises, so it cannot mask the original failure.
Two tests assert no container and no temp directory survives a run.

## What Phase 1 does *not* do

The container is not hardened yet. Code running in it currently has network
access, no memory or CPU ceiling, no PID limit, a writable root filesystem, and
runs as root. The only things standing between a script and the host are
Docker's default namespacing and the read-only `/code` mount.

Two bounds do exist, because without them the runner cannot be tested safely:
a wall-clock timeout that kills the container, and a 64 KB cap on captured
output so a runaway `print` loop cannot exhaust host memory.

Closing the rest is Phase 2: `--network none`, `--memory`, `--cpus`,
`--pids-limit`, `--read-only` with a tmpfs `/tmp`, and a non-root user.

## Tests

```bash
pytest tests/ -v
```

14 integration tests covering output capture, stream separation, exit codes,
truncation, timeout kill, and teardown guarantees. They need a running Docker
daemon and skip cleanly without one.
