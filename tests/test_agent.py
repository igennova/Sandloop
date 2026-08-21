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


# --- the loop ---------------------------------------------------------------

import pytest

from agent import AgentRun, DEFAULT_MAX_ITERATIONS, run_agent
from sandbox import ExecutionResult, SandboxError


class ScriptedClient:
    """An LLMClient that replays a fixed list of turns and records what it saw."""

    def __init__(self, *turns: AssistantTurn):
        self._turns = list(turns)
        self.transcripts: list[tuple] = []

    def next_turn(self, transcript):
        self.transcripts.append(tuple(transcript))
        return self._turns.pop(0) if self._turns else AssistantTurn(text="(exhausted)")


class LoopingClient:
    """Never stops asking for tools — stands in for a confused model."""

    def __init__(self):
        self.calls = 0

    def next_turn(self, transcript):
        self.calls += 1
        return AssistantTurn(
            tool_calls=(ToolCall(f"t{self.calls}", "execute_code", {"code": "pass", "language": "python"}),)
        )


def fake_executor(code, language="python", **kwargs):
    return ExecutionResult(stdout=f"ran {code!r}\n", stderr="", exit_code=0, duration_seconds=0.01)


def test_returns_immediately_when_no_tool_is_requested():
    run = run_agent("hi", ScriptedClient(AssistantTurn(text="hello")), executor=fake_executor)
    assert isinstance(run, AgentRun)
    assert run.final_text == "hello"
    assert run.iterations == 1
    assert run.hit_iteration_cap is False


def test_executes_a_tool_then_finishes():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "print(1)", "language": "python"}),)),
        AssistantTurn(text="the answer is 1"),
    )
    run = run_agent("compute", client, executor=fake_executor)
    assert run.final_text == "the answer is 1"
    assert run.iterations == 2

    kinds = [type(t).__name__ for t in run.transcript]
    assert kinds == ["UserTurn", "AssistantTurn", "ToolResultTurn", "AssistantTurn"]


def test_tool_output_is_fed_back_to_the_model():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "print(1)", "language": "python"}),)),
        AssistantTurn(text="done"),
    )
    run_agent("go", client, executor=fake_executor)
    # The transcript handed to the model on its second turn must contain the result.
    second_call = client.transcripts[1]
    result_turn = second_call[-1]
    assert isinstance(result_turn, ToolResultTurn)
    assert "ran 'print(1)'" in result_turn.results[0].content


def test_iteration_cap_stops_a_runaway_model():
    client = LoopingClient()
    run = run_agent("loop forever", client, max_iterations=4, executor=fake_executor)
    assert run.hit_iteration_cap is True
    assert run.iterations == 4
    assert client.calls == 4


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        run_agent("x", ScriptedClient(), max_iterations=0, executor=fake_executor)


def test_parallel_tool_calls_are_all_answered_in_one_turn():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(
            ToolCall("a", "execute_code", {"code": "print(1)", "language": "python"}),
            ToolCall("b", "execute_code", {"code": "print(2)", "language": "python"}),
        )),
        AssistantTurn(text="both done"),
    )
    run = run_agent("two things", client, executor=fake_executor)
    result_turn = run.transcript[2]
    assert isinstance(result_turn, ToolResultTurn)
    assert [r.call_id for r in result_turn.results] == ["a", "b"]


# --- dispatch never raises --------------------------------------------------


def test_unknown_tool_becomes_an_error_result():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "rm_rf", {}),)),
        AssistantTurn(text="oh"),
    )
    run = run_agent("x", client, executor=fake_executor)
    result = run.transcript[2].results[0]
    assert result.is_error is True
    assert "rm_rf" in result.content


def test_missing_code_argument_becomes_an_error_result():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"language": "python"}),)),
        AssistantTurn(text="oh"),
    )
    run = run_agent("x", client, executor=fake_executor)
    assert run.transcript[2].results[0].is_error is True


def test_unsupported_language_becomes_an_error_result():
    def picky(code, language="python", **kwargs):
        raise ValueError(f"Unsupported language {language!r}")

    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "x", "language": "cobol"}),)),
        AssistantTurn(text="oh"),
    )
    run = run_agent("x", client, executor=picky)
    result = run.transcript[2].results[0]
    assert result.is_error is True
    assert "cobol" in result.content


def test_broken_sandbox_is_reported_not_raised():
    def broken(code, language="python", **kwargs):
        raise SandboxError("daemon down")

    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "x", "language": "python"}),)),
        AssistantTurn(text="oh"),
    )
    run = run_agent("x", client, executor=broken)
    result = run.transcript[2].results[0]
    assert result.is_error is True
    assert "daemon down" in result.content


def test_failing_code_is_flagged_but_does_not_stop_the_loop():
    def failing(code, language="python", **kwargs):
        return ExecutionResult(stdout="", stderr="Traceback...", exit_code=1, duration_seconds=0.01)

    client = ScriptedClient(
        AssistantTurn(tool_calls=(ToolCall("t1", "execute_code", {"code": "boom", "language": "python"}),)),
        AssistantTurn(text="I will fix it"),
    )
    run = run_agent("x", client, executor=failing)
    assert run.transcript[2].results[0].is_error is True
    assert run.final_text == "I will fix it"
