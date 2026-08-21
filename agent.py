"""
Phase 3 — the agent loop.

Deliberately provider-agnostic: the loop talks to the `LLMClient` protocol
below, so neither it nor the sandbox imports a vendor SDK. Concrete adapters
(Anthropic, OpenAI, a local model) implement one method and plug in.

The sandbox is exposed to the model as a single tool, `execute_code`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from sandbox import (
    DEFAULT_TIMEOUT_SECONDS,
    SUPPORTED_LANGUAGES,
    TRUNCATION_NOTICE,
    DEFAULT_LIMITS,
    ExecutionResult,
    SandboxError,
    run_code,
)

# Built from the sandbox's own registry, so adding a language to
# SUPPORTED_LANGUAGES offers it to the model automatically instead of leaving
# the tool schema silently stale.
EXECUTE_CODE_TOOL: dict[str, Any] = {
    "name": "execute_code",
    "description": (
        "Run a snippet of code in an isolated sandbox and get back its stdout, "
        "stderr and exit code. The sandbox has no network access, 256 MB of "
        "memory, a read-only filesystem except for /tmp, and is destroyed after "
        "every call — nothing persists between calls, so each snippet must be "
        "self-contained. Use this to compute, test and verify rather than "
        "reasoning about results in your head."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The complete program to run. Must print its result.",
            },
            "language": {
                "type": "string",
                "enum": sorted(SUPPORTED_LANGUAGES),
                "description": "Language to run the snippet as.",
            },
        },
        "required": ["code", "language"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What we hand back for a single ToolCall."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    """
    One model turn. `tool_calls` holds more than one entry when the model asks
    for several tools at once; every one of them must be answered in a single
    following ToolResultTurn.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class ToolResultTurn:
    results: tuple[ToolResult, ...]


Turn = UserTurn | AssistantTurn | ToolResultTurn


class LLMClient(Protocol):
    """
    The whole provider surface. An adapter converts `transcript` into whatever
    its API expects, makes one call, and converts the reply back.
    """

    def next_turn(self, transcript: Sequence[Turn]) -> AssistantTurn: ...


DEFAULT_MAX_ITERATIONS = 10

# Signature of the thing that actually runs code. Injectable so the loop can be
# tested without Docker, and so Phase 4 can swap in a session-scoped runner.
Executor = Callable[..., ExecutionResult]


@dataclass(frozen=True)
class AgentRun:
    transcript: tuple[Turn, ...]
    final_text: str
    iterations: int
    hit_iteration_cap: bool


# Deaths by signal arrive with an empty stderr — the kernel kills the process,
# so there is no traceback to read. Without a note the model sees a bare exit
# code and retries the identical snippet.
SIGNAL_NOTES: dict[int, str] = {
    137: (
        f"killed by the sandbox (SIGKILL): it exceeded the "
        f"{DEFAULT_LIMITS.memory} memory limit. Process less data at a time, or "
        f"stream instead of building the whole result in memory."
    ),
    139: "crashed with a segmentation fault.",
    143: "terminated by the sandbox (SIGTERM).",
}


def _format_result(
    result: ExecutionResult, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> str:
    """
    Render an ExecutionResult as something the model can act on.

    Anything the model could not infer from stdout alone is stated outright:
    why it was killed, that output was cut, or that it printed nothing.
    """
    lines: list[str] = []

    if result.timed_out:
        lines.append(
            f"TIMED OUT: still running after {timeout_seconds:g}s and killed. "
            f"The work is too slow or the code does not terminate."
        )
    elif result.exit_code in SIGNAL_NOTES:
        lines.append(f"exit_code: {result.exit_code} — {SIGNAL_NOTES[result.exit_code]}")
    else:
        lines.append(f"exit_code: {result.exit_code}")

    if result.stdout:
        lines.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        lines.append(f"stderr:\n{result.stderr.rstrip()}")

    if TRUNCATION_NOTICE in result.stdout or TRUNCATION_NOTICE in result.stderr:
        lines.append(
            "NOTE: output was truncated. Print less, or summarise inside the snippet."
        )

    # Only on a clean run. After a kill the absence of output is explained by
    # the kill, and saying "you printed nothing" would just be misleading.
    if result.ok and not result.stdout and not result.stderr:
        lines.append(
            "NOTE: the snippet ran but printed nothing. Results are only visible "
            "if the code prints them."
        )

    return "\n".join(lines)


def _dispatch(
    call: ToolCall,
    executor: Executor,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    """
    Run one tool call. Never raises: a malformed call or a broken sandbox comes
    back as an error result so the model can read it and adjust, which is the
    whole point of the loop.
    """
    if call.name != EXECUTE_CODE_TOOL["name"]:
        return ToolResult(
            call.id,
            f"No tool named {call.name!r}. Available: {EXECUTE_CODE_TOOL['name']}.",
            is_error=True,
        )

    code = call.arguments.get("code")
    language = call.arguments.get("language", "python")
    if not isinstance(code, str) or not code.strip():
        return ToolResult(call.id, "The 'code' argument is required.", is_error=True)

    try:
        result = executor(code, language=language, timeout_seconds=timeout_seconds)
    except ValueError as exc:  # unsupported language
        return ToolResult(call.id, str(exc), is_error=True)
    except SandboxError as exc:  # the sandbox itself is broken
        return ToolResult(call.id, f"Sandbox unavailable: {exc}", is_error=True)

    return ToolResult(
        call.id, _format_result(result, timeout_seconds), is_error=not result.ok
    )


def run_agent(
    task: str,
    client: LLMClient,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    executor: Executor = run_code,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> AgentRun:
    """
    Drive the model until it answers without asking for a tool.

    Stops after `max_iterations` model turns regardless, so a model that keeps
    calling the sandbox forever costs a bounded amount rather than running
    until someone notices.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    transcript: list[Turn] = [UserTurn(task)]
    turn = AssistantTurn()

    for iteration in range(1, max_iterations + 1):
        turn = client.next_turn(transcript)
        transcript.append(turn)

        if not turn.wants_tools:
            return AgentRun(tuple(transcript), turn.text, iteration, False)

        # Every call in the turn is answered, together, in one result turn.
        results = tuple(
            _dispatch(call, executor, timeout_seconds) for call in turn.tool_calls
        )
        transcript.append(ToolResultTurn(results))

    return AgentRun(tuple(transcript), turn.text, max_iterations, True)
