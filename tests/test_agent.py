"""Phase 3 tests. Provider-agnostic: no network and no API key required."""

from __future__ import annotations

from agent import (
    EXECUTE_CODE_TOOL,
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultTurn,
    UserTurn,
)
from sandbox import SUPPORTED_LANGUAGES


def test_tool_schema_shape():
    assert EXECUTE_CODE_TOOL["name"] == "execute_code"
    schema = EXECUTE_CODE_TOOL["input_schema"]
    assert schema["required"] == ["code", "language"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"code", "language"}


def test_tool_languages_track_the_sandbox_registry():
    # The point of deriving the enum: a new language cannot be offered by the
    # sandbox while the model is still told it does not exist.
    assert EXECUTE_CODE_TOOL["input_schema"]["properties"]["language"]["enum"] == sorted(
        SUPPORTED_LANGUAGES
    )


def test_tool_description_states_the_sandbox_constraints():
    # The model cannot discover these by trying; they have to be in the prompt.
    description = EXECUTE_CODE_TOOL["description"].lower()
    for constraint in ("no network", "read-only", "nothing persists"):
        assert constraint in description


def test_assistant_turn_defaults_to_no_tools():
    assert AssistantTurn(text="done").wants_tools is False


def test_assistant_turn_reports_pending_tools():
    turn = AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "1"}),))
    assert turn.wants_tools is True


def test_turns_are_hashable_value_objects():
    # Frozen dataclasses: a transcript can be compared and reasoned about in
    # tests without identity games.
    assert UserTurn("hi") == UserTurn("hi")
    assert ToolResultTurn((ToolResult("t1", "ok"),)) == ToolResultTurn(
        (ToolResult("t1", "ok"),)
    )
