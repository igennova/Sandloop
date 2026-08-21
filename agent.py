"""
Phase 3 — the agent loop.

Deliberately provider-agnostic: the loop talks to the `LLMClient` protocol
below, so neither it nor the sandbox imports a vendor SDK. Concrete adapters
(Anthropic, OpenAI, a local model) implement one method and plug in.

The sandbox is exposed to the model as a single tool, `execute_code`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from sandbox import SUPPORTED_LANGUAGES

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
