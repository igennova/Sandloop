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


# --- results the model can act on -------------------------------------------

from agent import _format_result
from sandbox import TRUNCATION_NOTICE


def _result(**kwargs):
    base = dict(stdout="", stderr="", exit_code=0, duration_seconds=0.1)
    return ExecutionResult(**{**base, **kwargs})


def test_oom_kill_explains_itself():
    # exit 137 with empty stderr is the case the model cannot diagnose alone.
    text = _format_result(_result(exit_code=137))
    assert "137" in text
    assert "memory" in text.lower()
    assert "256m" in text.lower()


def test_timeout_states_the_limit_it_hit():
    text = _format_result(_result(timed_out=True, exit_code=-1), timeout_seconds=5)
    assert "TIMED OUT" in text
    assert "5s" in text


def test_segfault_is_named():
    assert "segmentation fault" in _format_result(_result(exit_code=139))


def test_silent_success_tells_the_model_to_print():
    text = _format_result(_result(exit_code=0))
    assert "printed nothing" in text


def test_a_killed_run_is_not_also_scolded_for_printing_nothing():
    # The kill already explains the silence; both notes at once is noise.
    assert "printed nothing" not in _format_result(_result(exit_code=137))
    assert "printed nothing" not in _format_result(_result(timed_out=True, exit_code=-1))


def test_truncated_output_is_flagged():
    text = _format_result(_result(stdout="x" * 100 + TRUNCATION_NOTICE))
    assert "truncated" in text


def test_ordinary_output_is_not_editorialised():
    text = _format_result(_result(stdout="42\n"))
    assert "42" in text
    assert "NOTE:" not in text


def test_stderr_is_labelled_separately():
    text = _format_result(_result(stdout="out", stderr="err", exit_code=1))
    assert "stdout:\nout" in text
    assert "stderr:\nerr" in text


# --- end to end through the real sandbox ------------------------------------

try:
    import docker as _docker

    _docker.from_env().ping()
    _DOCKER_UP = True
except Exception:  # noqa: BLE001
    _DOCKER_UP = False

real_docker = pytest.mark.skipif(not _DOCKER_UP, reason="Docker daemon not available")


@real_docker
def test_loop_runs_real_code_and_returns_real_output():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(
            ToolCall("t1", "execute_code", {"code": "print(6 * 7)", "language": "python"}),
        )),
        AssistantTurn(text="42"),
    )
    run = run_agent("what is 6*7", client)  # real run_code, no fake
    assert "42" in run.transcript[2].results[0].content
    assert run.final_text == "42"


@real_docker
def test_real_oom_reaches_the_model_with_an_explanation():
    client = ScriptedClient(
        AssistantTurn(tool_calls=(
            ToolCall("t1", "execute_code", {
                "code": 'b = bytearray(512 * 1024 * 1024)\nprint("done")',
                "language": "python",
            }),
        )),
        AssistantTurn(text="too big"),
    )
    run = run_agent("allocate a lot", client, timeout_seconds=60)
    result = run.transcript[2].results[0]
    assert result.is_error is True
    assert "memory" in result.content.lower()
