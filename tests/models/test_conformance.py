"""Verify the reusable backend conformance runner and every standard contract.

The suite exercises factory lifecycles, request and stream invariants,
multi-request scenarios, normalized provider failures, cancellation identity,
report aggregation, and public-boundary validation.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
from typing import cast

import pytest

from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.caching import CacheMode, PromptStability
from purrcept_core.models.conformance import (
    BackendConformanceCase,
    BackendConformanceError,
    BackendConformanceReport,
    BackendConformanceResult,
    BackendConformanceScenario,
    BackendConformanceStep,
    BackendScenarioFactory,
    run_backend_conformance,
)
from purrcept_core.models.content import TextBlock, ToolCallBlock, ToolResultBlock
from purrcept_core.models.continuation import ModelContinuation
from purrcept_core.models.errors import ModelError, ModelUnavailableError
from purrcept_core.models.messages import Message, MessageRole
from purrcept_core.models.model import Model
from purrcept_core.models.reminders import ReminderPlacement, ReminderScope
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse
from purrcept_core.models.settings import ToolChoice
from purrcept_core.models.streaming import (
    ModelStreamCompleted,
    ModelStreamStarted,
    TextDelta,
)


class ScenarioBackend:
    def __init__(
        self,
        scenario: BackendConformanceScenario,
        *,
        stream_mode: str = "complete",
        error_mode: str = "expected",
        response_override: ModelResponse | None = None,
        follow_up_response_override: ModelResponse | None = None,
    ) -> None:
        self.scenario = scenario
        self.stream_mode = stream_mode
        self.error_mode = error_mode
        self.response_override = response_override
        self.follow_up_response_override = follow_up_response_override
        self.calls = 0

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            expected_request = self.scenario.request
        else:
            expected_request = self.scenario.follow_up_steps[self.calls - 2].request
        assert request is expected_request
        assert model == self.scenario.model_name
        assert emit is not None

        expected_error = self.scenario.expected_error
        if expected_error is not None:
            if self.error_mode == "swallow":
                return ModelResponse(Message.assistant("swallowed"))
            if self.error_mode == "replace_model_error":
                raise ModelUnavailableError("replacement")
            if self.error_mode == "replace_cancellation":
                raise asyncio.CancelledError("replacement")
            if self.error_mode == "generic":
                raise LookupError("unexpected")
            raise expected_error

        if self.calls == 1:
            response = self.response_override or self.scenario.expected_response
        else:
            response = (
                self.follow_up_response_override
                or self.scenario.follow_up_steps[self.calls - 2].expected_response
            )
        assert response is not None
        if self.scenario.streaming:
            if self.stream_mode in {
                "boundaries_only",
                "complete",
                "started_only",
                "mismatch",
            }:
                emit(
                    ModelStreamStarted(
                        model=model,
                        response_id=response.response_id,
                    )
                )
            if self.stream_mode == "complete":
                emit(TextDelta(response.text))
                emit(ModelStreamCompleted(response))
            elif self.stream_mode == "completion_only":
                emit(ModelStreamCompleted(response))
            elif self.stream_mode == "boundaries_only":
                emit(ModelStreamCompleted(response))
            elif self.stream_mode == "mismatch":
                expected = self.scenario.expected_response
                assert expected is not None
                emit(ModelStreamCompleted(expected))
        return response


def _model_for(
    scenario: BackendConformanceScenario,
    *,
    stream_mode: str = "complete",
    error_mode: str = "expected",
    response_override: ModelResponse | None = None,
    follow_up_response_override: ModelResponse | None = None,
    model_name: str | None = None,
) -> Model:
    """Bind one configurable fake transport to the scenario's expected model name."""

    return Model(
        ScenarioBackend(
            scenario,
            stream_mode=stream_mode,
            error_mode=error_mode,
            response_override=response_override,
            follow_up_response_override=follow_up_response_override,
        ),
        scenario.model_name if model_name is None else model_name,
    )


def _conforming_factory(scenario: BackendConformanceScenario) -> Model:
    return _model_for(scenario)


def _simple_scenario(
    *,
    response: ModelResponse | None = None,
    error: BaseException | None = None,
    streaming: bool = False,
) -> BackendConformanceScenario:
    """Build a minimal scenario for focused runner failure tests."""

    return BackendConformanceScenario(
        BackendConformanceCase.BASIC_RESPONSE,
        "fixture-model",
        ModelRequest((Message.user("hello"),)),
        expected_response=response,
        expected_error=error,
        streaming=streaming,
    )


async def test_standard_conformance_run_exercises_all_contracts() -> None:
    seen: list[BackendConformanceScenario] = []

    def factory(scenario: BackendConformanceScenario) -> Model:
        seen.append(scenario)
        if scenario.case is BackendConformanceCase.REQUEST_SEMANTICS:
            request = scenario.request
            assert len(request.messages) == 3
            assert request.messages[0].metadata["source"] == "suite"
            tool_call = request.messages[1].content[0]
            assert isinstance(tool_call, ToolCallBlock)
            assert tool_call.id == "conformance-tool-call"
            assert tool_call.name == "lookup"
            assert tool_call.arguments == {
                "query": "purrcept",
                "options": {"exact": True, "limit": 1},
            }
            tool_result = request.messages[2].content[0]
            assert isinstance(tool_result, ToolResultBlock)
            assert tool_result.tool_call_id == tool_call.id
            assert tool_result.content == (TextBlock("fixture lookup result"),)
            assert tool_result.is_error is False
            assert request.messages[2].name == "lookup"
            assert request.messages[2].metadata["source"] == "fixture"
            assert request.instructions[0].key == "contract"
            assert request.instructions[0].stability is PromptStability.GROWING
            assert tuple(reminder.key for reminder in request.reminders) == (
                "concise",
                "format",
                "fallback",
            )
            assert tuple(reminder.scope for reminder in request.reminders) == (
                ReminderScope.TURN,
                ReminderScope.TURN,
                ReminderScope.NEXT_REQUEST,
            )
            assert tuple(reminder.placement for reminder in request.reminders) == (
                ReminderPlacement.TAIL,
                ReminderPlacement.INSTRUCTIONS,
                ReminderPlacement.AUTO,
            )
            assert tuple(reminder.priority for reminder in request.reminders) == (7, 7, -2)
            assert request.tools[0].name == "lookup"
            assert request.tools[0].parameters["additionalProperties"] is False
            assert request.settings.temperature == 0.25
            assert request.settings.max_output_tokens == 64
            assert request.settings.stop_sequences == ("<END>",)
            assert request.settings.tool_choice is ToolChoice.REQUIRED
            assert request.settings.parallel_tool_calls is True
            assert request.cache.mode is CacheMode.EXPLICIT
            assert request.cache.key == "conformance-cache"
            assert request.cache.strict is True
            assert request.continuation == ModelContinuation(
                "conformance-provider",
                {"cursor": "request-cursor"},
            )
            assert request.metadata["trace"] == "conformance"
            assert request.provider_options["fixture"]["enabled"] is True
        return _model_for(scenario)

    report = await run_backend_conformance(factory)

    assert report.passed
    assert report.failures == ()
    assert [result.case for result in report.results] == list(BackendConformanceCase)
    assert all(result.passed for result in report.results)
    assert [scenario.case for scenario in seen] == list(BackendConformanceCase)
    assert not hasattr(report, "__dict__")
    report.raise_for_failures()
    with pytest.raises(FrozenInstanceError):
        report.results = ()  # type: ignore[misc]


async def test_async_factory_and_non_streaming_profile_are_supported() -> None:
    async def factory(scenario: BackendConformanceScenario) -> Model:
        await asyncio.sleep(0)
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert report.passed
    assert [result.case for result in report.results] == [
        BackendConformanceCase.BASIC_RESPONSE,
        BackendConformanceCase.REQUEST_SEMANTICS,
        BackendConformanceCase.REMINDER_REMOVAL,
        BackendConformanceCase.REMINDER_REPLACEMENT,
        BackendConformanceCase.MODEL_ERROR,
        BackendConformanceCase.CANCELLATION,
    ]


@pytest.mark.parametrize(
    ("factory", "model_name", "include_streaming", "match"),
    [
        (
            cast(BackendScenarioFactory, object()),
            "model",
            True,
            "factory must be callable",
        ),
        (
            _conforming_factory,
            cast(str, object()),
            True,
            "model_name must be a string",
        ),
        (
            _conforming_factory,
            "",
            True,
            "model_name must not be empty",
        ),
        (
            _conforming_factory,
            "model",
            cast(bool, 1),
            "include_streaming must be a bool",
        ),
    ],
)
async def test_runner_validates_its_public_boundary(
    factory: BackendScenarioFactory,
    model_name: str,
    include_streaming: bool,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        await run_backend_conformance(
            factory,
            model_name=model_name,
            include_streaming=include_streaming,
        )


def test_scenario_is_immutable_and_validates_its_public_boundary() -> None:
    response = ModelResponse(Message.assistant("ok"))
    scenario = _simple_scenario(response=response)

    assert scenario.expected_response is response
    assert not hasattr(scenario, "__dict__")
    scenario_fields = {item.name: item for item in fields(BackendConformanceScenario)}
    assert scenario_fields["expected_response"].kw_only
    assert scenario_fields["expected_error"].kw_only
    assert scenario_fields["streaming"].kw_only
    assert scenario_fields["follow_up_steps"].kw_only
    with pytest.raises(FrozenInstanceError):
        scenario.streaming = True  # type: ignore[misc]
    with pytest.raises(TypeError, match="positional"):
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            "model",
            ModelRequest((Message.user("hi"),)),
            response,  # type: ignore[misc]
        )

    with pytest.raises(TypeError, match="case must be"):
        BackendConformanceScenario(
            cast(BackendConformanceCase, object()),
            "model",
            ModelRequest((Message.user("hi"),)),
            expected_response=response,
        )
    with pytest.raises(TypeError, match="request must be"):
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            "model",
            cast(ModelRequest, object()),
            expected_response=response,
        )
    with pytest.raises(TypeError, match="expected_response must be"):
        _simple_scenario(response=cast(ModelResponse, object()))
    with pytest.raises(TypeError, match="expected_error must be"):
        _simple_scenario(error=cast(BaseException, LookupError("bad")))
    with pytest.raises(ValueError, match="exactly one"):
        _simple_scenario()
    with pytest.raises(ValueError, match="exactly one"):
        _simple_scenario(response=response, error=ModelError("bad"))
    with pytest.raises(TypeError, match="streaming must be a bool"):
        _simple_scenario(response=response, streaming=cast(bool, 1))
    with pytest.raises(ValueError, match="streaming scenarios"):
        _simple_scenario(error=ModelError("bad"), streaming=True)


def test_follow_up_steps_are_immutable_and_require_successful_non_streaming_scenarios() -> None:
    response = ModelResponse(Message.assistant("ok"))
    step = BackendConformanceStep(
        ModelRequest((Message.user("follow up"),)),
        response,
    )

    assert step.expected_response is response
    assert not hasattr(step, "__dict__")
    with pytest.raises(FrozenInstanceError):
        step.request = ModelRequest((Message.user("changed"),))  # type: ignore[misc]
    with pytest.raises(TypeError, match="request must be"):
        BackendConformanceStep(cast(ModelRequest, object()), response)
    with pytest.raises(TypeError, match="expected_response must be"):
        BackendConformanceStep(
            ModelRequest((Message.user("x"),)),
            cast(ModelResponse, object()),
        )
    with pytest.raises(TypeError, match="follow_up_steps must be an iterable"):
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            "model",
            ModelRequest((Message.user("x"),)),
            expected_response=response,
            follow_up_steps=cast(tuple[BackendConformanceStep, ...], object()),
        )
    with pytest.raises(TypeError, match="only BackendConformanceStep"):
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            "model",
            ModelRequest((Message.user("x"),)),
            expected_response=response,
            follow_up_steps=cast(tuple[BackendConformanceStep, ...], (object(),)),
        )
    with pytest.raises(ValueError, match="require a non-streaming"):
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            "model",
            ModelRequest((Message.user("x"),)),
            expected_error=ModelError("failed"),
            follow_up_steps=(step,),
        )
    with pytest.raises(ValueError, match="require a non-streaming"):
        BackendConformanceScenario(
            BackendConformanceCase.STREAMING,
            "model",
            ModelRequest((Message.user("x"),)),
            expected_response=response,
            streaming=True,
            follow_up_steps=(step,),
        )


def test_result_and_report_validate_and_normalize_their_public_boundaries() -> None:
    result = BackendConformanceResult(BackendConformanceCase.BASIC_RESPONSE)
    failed = BackendConformanceResult(
        BackendConformanceCase.MODEL_ERROR,
        error=ValueError("failed"),
    )
    report = BackendConformanceReport(cast(tuple[BackendConformanceResult, ...], [result, failed]))

    assert result.passed
    assert not failed.passed
    assert report.results == (result, failed)
    assert isinstance(report.results, tuple)
    with pytest.raises(TypeError, match="case must be"):
        BackendConformanceResult(cast(BackendConformanceCase, object()))
    with pytest.raises(TypeError, match="error must be"):
        BackendConformanceResult(
            BackendConformanceCase.CANCELLATION,
            error=cast(Exception, asyncio.CancelledError()),
        )
    with pytest.raises(TypeError, match="results must be an iterable"):
        BackendConformanceReport(cast(tuple[BackendConformanceResult, ...], object()))
    with pytest.raises(TypeError, match="only BackendConformanceResult"):
        BackendConformanceReport(cast(tuple[BackendConformanceResult, ...], (object(),)))


async def test_invalid_factory_result_is_reported_and_can_be_raised() -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        del scenario
        return cast(Model, object())

    report = await run_backend_conformance(factory, include_streaming=False)

    assert not report.passed
    assert len(report.failures) == 6
    assert all(isinstance(result.error, TypeError) for result in report.failures)
    with pytest.raises(BackendConformanceError) as caught:
        report.raise_for_failures()
    assert caught.value.report is report
    assert "6 model backend conformance scenario(s) failed" in str(caught.value)
    assert "basic_response: TypeError" in str(caught.value)


async def test_follow_up_response_mismatch_is_reported() -> None:
    wrong = ModelResponse(Message.assistant("wrong follow up"))

    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.REMINDER_REMOVAL:
            return _model_for(
                scenario,
                follow_up_response_override=wrong,
            )
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert len(report.failures) == 1
    assert report.failures[0].case is BackendConformanceCase.REMINDER_REMOVAL
    assert isinstance(report.failures[0].error, AssertionError)
    assert "wrong follow up" in str(report.failures[0].error)


async def test_wrong_model_name_is_a_reported_contract_failure() -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        return _model_for(scenario, model_name="wrong-model")

    report = await run_backend_conformance(factory, include_streaming=False)

    assert not report.passed
    assert all(
        isinstance(result.error, AssertionError)
        and "factory returned model name" in str(result.error)
        for result in report.failures
    )


async def test_response_mismatch_is_reported() -> None:
    replacement = ModelResponse(Message.assistant("different"))

    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.BASIC_RESPONSE:
            return _model_for(scenario, response_override=replacement)
        return _model_for(scenario)

    report = await run_backend_conformance(factory)

    assert len(report.failures) == 1
    assert report.failures[0].case is BackendConformanceCase.BASIC_RESPONSE
    assert "backend returned" in str(report.failures[0].error)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "non-negative int"),
        ("1", "non-negative int"),
        (-1, "non-negative int"),
    ],
)
async def test_invalid_usage_is_reported(value: object, match: str) -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.REQUEST_SEMANTICS:
            response = scenario.expected_response
            assert response is not None
            usage = response.usage
            assert usage is not None
            object.__setattr__(usage, "input_tokens", value)
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert len(report.failures) == 1
    assert match in str(report.failures[0].error)


async def test_non_assistant_response_is_reported_even_if_backend_bypasses_constructor() -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.BASIC_RESPONSE:
            response = scenario.expected_response
            assert response is not None
            object.__setattr__(response.message, "role", MessageRole.USER)
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert len(report.failures) == 1
    assert "role must be assistant" in str(report.failures[0].error)


@pytest.mark.parametrize(
    ("stream_mode", "match"),
    [
        ("none", "ModelStreamStarted first"),
        ("completion_only", "ModelStreamStarted first"),
        ("started_only", "ModelStreamCompleted last"),
        ("boundaries_only", "at least one TextDelta"),
    ],
)
async def test_streaming_profile_requires_explicit_boundaries(
    stream_mode: str,
    match: str,
) -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.STREAMING:
            return _model_for(scenario, stream_mode=stream_mode)
        return _model_for(scenario)

    report = await run_backend_conformance(factory)

    assert len(report.failures) == 1
    assert report.failures[0].case is BackendConformanceCase.STREAMING
    assert match in str(report.failures[0].error)


async def test_generate_detects_mismatched_stream_completion() -> None:
    replacement = ModelResponse(Message.assistant("different"))

    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.STREAMING:
            return _model_for(
                scenario,
                stream_mode="mismatch",
                response_override=replacement,
            )
        return _model_for(scenario)

    report = await run_backend_conformance(factory)

    assert len(report.failures) == 1
    assert isinstance(report.failures[0].error, TypeError)
    assert "must equal the ModelResponse returned" in str(report.failures[0].error)


@pytest.mark.parametrize(
    ("case", "error_mode"),
    [
        (BackendConformanceCase.MODEL_ERROR, "swallow"),
        (BackendConformanceCase.CANCELLATION, "swallow"),
        (BackendConformanceCase.MODEL_ERROR, "replace_model_error"),
        (BackendConformanceCase.MODEL_ERROR, "generic"),
    ],
)
async def test_expected_failures_must_propagate_unchanged(
    case: BackendConformanceCase,
    error_mode: str,
) -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is case:
            return _model_for(scenario, error_mode=error_mode)
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert len(report.failures) == 1
    assert report.failures[0].case is case
    assert isinstance(report.failures[0].error, AssertionError)


async def test_unexpected_cancellation_is_never_converted_to_a_report_failure() -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.CANCELLATION:
            return _model_for(scenario, error_mode="replace_cancellation")
        return _model_for(scenario)

    with pytest.raises(asyncio.CancelledError, match="replacement"):
        await run_backend_conformance(factory, include_streaming=False)


async def test_unexpected_cancellation_from_a_success_case_propagates() -> None:
    class CancellingBackend:
        async def generate(
            self,
            request: ModelRequest,
            *,
            model: str,
            emit: ModelEventSink | None = None,
        ) -> ModelResponse:
            del request, model, emit
            raise asyncio.CancelledError("external")

    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.BASIC_RESPONSE:
            return Model(CancellingBackend(), scenario.model_name)
        return _model_for(scenario)

    with pytest.raises(asyncio.CancelledError, match="external"):
        await run_backend_conformance(factory, include_streaming=False)


async def test_unexpected_regular_exception_is_recorded() -> None:
    class FailingBackend:
        async def generate(
            self,
            request: ModelRequest,
            *,
            model: str,
            emit: ModelEventSink | None = None,
        ) -> ModelResponse:
            del request, model, emit
            raise RuntimeError("provider bug")

    def factory(scenario: BackendConformanceScenario) -> Model:
        if scenario.case is BackendConformanceCase.BASIC_RESPONSE:
            return Model(FailingBackend(), scenario.model_name)
        return _model_for(scenario)

    report = await run_backend_conformance(factory, include_streaming=False)

    assert len(report.failures) == 1
    assert isinstance(report.failures[0].error, RuntimeError)
    assert str(report.failures[0].error) == "provider bug"
