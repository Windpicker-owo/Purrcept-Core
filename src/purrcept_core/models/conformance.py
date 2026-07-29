"""Reusable conformance checks for third-party model backend packages.

The runner owns provider-neutral requests and assertions.  A provider package
supplies a scenario factory that configures its backend with a fake transport
for each scenario and returns a :class:`~purrcept_core.models.model.Model`.
This keeps conformance tests deterministic while still exercising the
provider's request translation, streaming conversion, and error propagation.

Each standard scenario receives a fresh model so provider tests control state
isolation; explicitly declared follow-up steps reuse that scenario's model to
exercise continuation-like adapter state. Contract violations become report
entries, while unexpected cancellation remains a control-flow outcome and
propagates to the caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from inspect import isawaitable
from typing import TypeAlias, cast

from ..context import ExecutionContext
from ..executor import InlineExecutor
from ._utils import require_non_empty_string
from .caching import CacheMode, PromptCachePolicy, PromptStability
from .content import TextBlock, ToolCallBlock
from .continuation import ModelContinuation
from .errors import ModelError, ModelUnavailableError
from .instructions import SystemInstruction
from .messages import Message, MessageRole
from .model import Model
from .reminders import ReminderPlacement, ReminderScope, SystemReminder
from .requests import ModelRequest, ToolSpec
from .responses import FinishReason, ModelResponse, TokenUsage
from .settings import ModelSettings, ToolChoice
from .streaming import ModelStreamCompleted, ModelStreamStarted, TextDelta


class BackendConformanceCase(StrEnum):
    """A standard behavior exercised by the backend conformance runner."""

    BASIC_RESPONSE = "basic_response"
    REQUEST_SEMANTICS = "request_semantics"
    REMINDER_REMOVAL = "reminder_removal"
    REMINDER_REPLACEMENT = "reminder_replacement"
    STREAMING = "streaming"
    MODEL_ERROR = "model_error"
    CANCELLATION = "cancellation"


@dataclass(frozen=True, slots=True)
class BackendConformanceStep:
    """One follow-up request executed against the same backend instance."""

    request: ModelRequest
    expected_response: ModelResponse

    def __post_init__(self) -> None:
        _validate_request(self.request)
        if not isinstance(cast(object, self.expected_response), ModelResponse):
            raise TypeError("expected_response must be a ModelResponse.")


@dataclass(frozen=True, slots=True)
class BackendConformanceScenario:
    """One deterministic contract a provider's fake transport must implement.

    The factory should arrange for ``expected_response`` or ``expected_error``
    to be produced unchanged.  When ``streaming`` is true, the backend must
    emit a ``ModelStreamStarted`` event first and a matching
    ``ModelStreamCompleted`` event last.
    """

    case: BackendConformanceCase
    model_name: str
    request: ModelRequest
    expected_response: ModelResponse | None = field(default=None, kw_only=True)
    expected_error: BaseException | None = field(
        default=None,
        repr=False,
        kw_only=True,
    )
    streaming: bool = field(default=False, kw_only=True)
    follow_up_steps: tuple[BackendConformanceStep, ...] = field(
        default=(),
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_case(self.case)
        require_non_empty_string(self.model_name, field_name="model_name")
        _validate_request(self.request)
        _validate_expected_response(self.expected_response)
        _validate_expected_error(self.expected_error)
        if (self.expected_response is None) == (self.expected_error is None):
            raise ValueError(
                "exactly one of expected_response and expected_error must be provided."
            )
        _validate_streaming(self.streaming)
        steps = _normalize_steps(self.follow_up_steps)
        if self.streaming and self.expected_response is None:
            raise ValueError("streaming scenarios must provide expected_response.")
        if steps and (self.expected_response is None or self.streaming):
            raise ValueError("follow-up steps require a non-streaming expected_response scenario.")
        object.__setattr__(self, "follow_up_steps", steps)


BackendScenarioFactory: TypeAlias = Callable[
    [BackendConformanceScenario],
    Model | Awaitable[Model],
]
"""Create a scenario-specific model backed by deterministic fake transport.

Provider test suites implement this callable and pass it to
``run_backend_conformance``. It is called once per scenario and may construct
the model synchronously or asynchronously. The returned model name must match
the scenario, and the configured backend must produce the supplied expected
objects unchanged.
"""


@dataclass(frozen=True, slots=True)
class BackendConformanceResult:
    """The outcome of one conformance scenario."""

    case: BackendConformanceCase
    error: Exception | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        _validate_case(self.case)
        _validate_result_error(self.error)

    @property
    def passed(self) -> bool:
        """Whether the scenario completed without a contract violation."""

        return self.error is None


@dataclass(frozen=True, slots=True)
class BackendConformanceReport:
    """All outcomes from one backend conformance run."""

    results: tuple[BackendConformanceResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", _normalize_results(self.results))

    @property
    def failures(self) -> tuple[BackendConformanceResult, ...]:
        """Return only failed scenario outcomes."""

        return tuple(result for result in self.results if not result.passed)

    @property
    def passed(self) -> bool:
        """Whether every selected scenario passed."""

        return not self.failures

    def raise_for_failures(self) -> None:
        """Raise one readable assertion error when any scenario failed."""

        if not self.passed:
            raise BackendConformanceError(self)


class BackendConformanceError(AssertionError):
    """One or more model backend conformance scenarios failed."""

    def __init__(self, report: BackendConformanceReport) -> None:
        self.report = report
        failures = report.failures
        details = "; ".join(
            f"{result.case.value}: {type(result.error).__name__}: {result.error}"
            for result in failures
        )
        super().__init__(f"{len(failures)} model backend conformance scenario(s) failed: {details}")


async def run_backend_conformance(
    factory: BackendScenarioFactory,
    *,
    model_name: str = "purrcept-conformance-model",
    include_streaming: bool = True,
) -> BackendConformanceReport:
    """Run deterministic standard scenarios against models built by ``factory``.

    ``factory`` may be synchronous or asynchronous.  It receives the exact
    request, expected outcome, requested model name, and streaming intent for
    each scenario.  Unexpected task cancellation is deliberately propagated
    rather than converted into a report failure.

    Set ``include_streaming=False`` only for a backend that intentionally does
    not implement the optional stream-event capability.
    """

    if not callable(factory):
        raise TypeError("factory must be callable.")
    require_non_empty_string(model_name, field_name="model_name")
    _validate_include_streaming(include_streaming)

    scenarios = _standard_scenarios(model_name)
    if not include_streaming:
        scenarios = tuple(
            scenario
            for scenario in scenarios
            if scenario.case is not BackendConformanceCase.STREAMING
        )

    # Every scenario is isolated by a factory call. Ordinary contract failures
    # are accumulated so one adapter run reports its complete compatibility
    # surface; CancelledError remains outside Exception and aborts immediately.
    results: list[BackendConformanceResult] = []
    for index, scenario in enumerate(scenarios):
        try:
            await _run_scenario(factory, scenario, index=index)
        except Exception as error:
            results.append(BackendConformanceResult(scenario.case, error=error))
        else:
            results.append(BackendConformanceResult(scenario.case))
    return BackendConformanceReport(tuple(results))


async def _run_scenario(
    factory: BackendScenarioFactory,
    scenario: BackendConformanceScenario,
    *,
    index: int,
) -> None:
    """Exercise one scenario and raise on its first contract violation."""

    model = await _build_model(factory, scenario)
    if model.name != scenario.model_name:
        raise AssertionError(
            f"factory returned model name {model.name!r}; expected {scenario.model_name!r}."
        )

    # Streaming progress is captured through the same Generate effect path used
    # by applications, so this validates both the backend callback contract and
    # core stream forwarding.
    progress: list[object] = []
    context = ExecutionContext(
        "model-backend-conformance",
        scenario.case.value,
        index,
        object(),
        _progress_callback=progress.append,
    )
    expected_error = scenario.expected_error
    if expected_error is not None:
        await _assert_expected_error(model, scenario.request, context, expected_error)
        return

    response = await InlineExecutor().execute(model.generate(scenario.request), context)
    _assert_response_invariants(response)
    if response != scenario.expected_response:
        raise AssertionError(
            f"backend returned {response!r}; expected {scenario.expected_response!r}."
        )
    if scenario.streaming:
        _assert_complete_stream(progress)
    # Follow-up steps intentionally reuse this model instance. They cover
    # adapter state that must survive multiple calls without leaking between
    # otherwise independent scenarios.
    for step_index, step in enumerate(scenario.follow_up_steps, start=1):
        follow_up_context = ExecutionContext(
            "model-backend-conformance",
            f"{scenario.case.value}-{step_index}",
            index,
            object(),
        )
        follow_up_response = await InlineExecutor().execute(
            model.generate(step.request),
            follow_up_context,
        )
        _assert_response_invariants(follow_up_response)
        if follow_up_response != step.expected_response:
            raise AssertionError(
                f"backend returned {follow_up_response!r}; expected {step.expected_response!r}."
            )


async def _build_model(
    factory: BackendScenarioFactory,
    scenario: BackendConformanceScenario,
) -> Model:
    """Resolve a synchronous or asynchronous factory and validate its result."""

    model_or_awaitable = factory(scenario)
    if isawaitable(model_or_awaitable):
        model_or_awaitable = await model_or_awaitable
    return _require_model(model_or_awaitable)


async def _assert_expected_error(
    model: Model,
    request: ModelRequest,
    context: ExecutionContext[object],
    expected: BaseException,
) -> None:
    """Require the exact supplied error object to propagate unchanged.

    Identity, rather than only type or equality, detects provider adapters that
    incorrectly wrap normalized failures or cancellation.
    """

    try:
        await InlineExecutor().execute(model.generate(request), context)
    except asyncio.CancelledError as error:
        if error is expected:
            return
        raise
    except Exception as error:
        if error is expected:
            return
        raise AssertionError(
            f"backend raised {error!r}; expected the supplied {expected!r} unchanged."
        ) from error
    raise AssertionError(f"backend did not raise the supplied {expected!r}.")


def _assert_response_invariants(response: ModelResponse) -> None:
    """Check response properties that every provider adapter must preserve."""

    if response.message.role is not MessageRole.ASSISTANT:
        raise AssertionError("backend response message role must be assistant.")
    usage = response.usage
    if usage is None:
        return
    counts = (
        ("input_tokens", usage.input_tokens),
        ("output_tokens", usage.output_tokens),
        ("cached_input_tokens", usage.cached_input_tokens),
        ("cache_write_input_tokens", usage.cache_write_input_tokens),
        ("reasoning_tokens", usage.reasoning_tokens),
    )
    for name, value in counts:
        _assert_token_count(name, value)


def _assert_complete_stream(progress: list[object]) -> None:
    """Require explicit stream boundaries and at least one text delta."""

    if not progress or not isinstance(progress[0], ModelStreamStarted):
        raise AssertionError("streaming backend must emit ModelStreamStarted first.")
    if not isinstance(progress[-1], ModelStreamCompleted):
        raise AssertionError("streaming backend must emit ModelStreamCompleted last.")
    if not any(isinstance(event, TextDelta) for event in progress[1:-1]):
        raise AssertionError("streaming backend must emit at least one TextDelta.")


def _validate_case(value: object) -> None:
    if not isinstance(value, BackendConformanceCase):
        raise TypeError("case must be a BackendConformanceCase.")


def _validate_request(value: object) -> None:
    if not isinstance(value, ModelRequest):
        raise TypeError("request must be a ModelRequest.")


def _validate_expected_response(value: object) -> None:
    if value is not None and not isinstance(value, ModelResponse):
        raise TypeError("expected_response must be a ModelResponse or None.")


def _validate_expected_error(value: object) -> None:
    if value is not None and not isinstance(
        value,
        (ModelError, asyncio.CancelledError),
    ):
        raise TypeError("expected_error must be a ModelError, asyncio.CancelledError, or None.")


def _validate_streaming(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("streaming must be a bool.")


def _validate_result_error(value: object) -> None:
    if value is not None and not isinstance(value, Exception):
        raise TypeError("error must be an Exception or None.")


def _normalize_results(value: object) -> tuple[BackendConformanceResult, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("results must be an iterable.")
    results = tuple(cast(Iterable[object], value))
    if not all(isinstance(result, BackendConformanceResult) for result in results):
        raise TypeError("results must contain only BackendConformanceResult instances.")
    return cast(tuple[BackendConformanceResult, ...], results)


def _normalize_steps(value: object) -> tuple[BackendConformanceStep, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("follow_up_steps must be an iterable.")
    steps = tuple(cast(Iterable[object], value))
    if not all(isinstance(step, BackendConformanceStep) for step in steps):
        raise TypeError("follow_up_steps must contain only BackendConformanceStep instances.")
    return cast(tuple[BackendConformanceStep, ...], steps)


def _validate_include_streaming(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("include_streaming must be a bool.")


def _require_model(value: object) -> Model:
    if not isinstance(value, Model):
        raise TypeError("factory must return a Model or an awaitable yielding a Model.")
    return value


def _assert_token_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssertionError(f"backend response {name} must be a non-negative int.")


def _standard_scenarios(model_name: str) -> tuple[BackendConformanceScenario, ...]:
    """Build fresh canonical requests and expected outcomes for one run.

    Values are created per invocation so provider fake transports may retain or
    compare them without sharing mutable test state across conformance runs.
    """

    basic_request = ModelRequest((Message.user("Reply with pong."),))
    basic_response = ModelResponse(
        Message.assistant("pong"),
        finish_reason=FinishReason.STOP,
        model=model_name,
        response_id="conformance-basic",
    )

    continuation = ModelContinuation(
        "conformance-provider",
        {"cursor": "request-cursor"},
    )
    request_semantics = ModelRequest(
        (
            Message.user(
                "Use the lookup tool if needed.",
                metadata={"source": "suite"},
            ),
            Message(
                MessageRole.ASSISTANT,
                (
                    ToolCallBlock(
                        "conformance-tool-call",
                        "lookup",
                        arguments={
                            "query": "purrcept",
                            "options": {"exact": True, "limit": 1},
                        },
                    ),
                ),
            ),
            Message.tool(
                "conformance-tool-call",
                "fixture lookup result",
                name="lookup",
                metadata={"source": "fixture"},
            ),
        ),
        instructions=(
            SystemInstruction.from_text(
                "Follow the conformance contract.",
                key="contract",
                stability=PromptStability.GROWING,
            ),
        ),
        reminders=(
            SystemReminder(
                (TextBlock("Return concise output."),),
                key="concise",
                scope=ReminderScope.TURN,
                placement=ReminderPlacement.TAIL,
                priority=7,
            ),
            SystemReminder(
                (TextBlock("Use the requested response format."),),
                key="format",
                scope=ReminderScope.TURN,
                placement=ReminderPlacement.INSTRUCTIONS,
                priority=7,
            ),
            SystemReminder(
                (TextBlock("Preserve the adapter fallback semantics."),),
                key="fallback",
                scope=ReminderScope.NEXT_REQUEST,
                placement=ReminderPlacement.AUTO,
                priority=-2,
            ),
        ),
        tools=(
            ToolSpec(
                "lookup",
                description="Look up one value.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ),
        settings=ModelSettings(
            temperature=0.25,
            max_output_tokens=64,
            stop_sequences=("<END>",),
            tool_choice=ToolChoice.REQUIRED,
            parallel_tool_calls=True,
        ),
        cache=PromptCachePolicy(
            mode=CacheMode.EXPLICIT,
            key="conformance-cache",
            ttl=timedelta(minutes=5),
            strict=True,
        ),
        continuation=continuation,
        metadata={"trace": "conformance"},
        provider_options={"fixture": {"enabled": True}},
    )
    semantics_response = ModelResponse(
        Message.assistant("request semantics accepted"),
        usage=TokenUsage(
            11,
            4,
            cached_input_tokens=3,
            cache_write_input_tokens=2,
            reasoning_tokens=1,
        ),
        finish_reason=FinishReason.STOP,
        model=model_name,
        response_id="conformance-semantics",
        continuation=ModelContinuation(
            "conformance-provider",
            {"cursor": "response-cursor"},
        ),
        provider_metadata={"region": "fixture"},
    )

    streaming_request = ModelRequest((Message.user("Stream one response."),))
    streaming_response = ModelResponse(
        Message.assistant("streamed response"),
        usage=TokenUsage(4, 2),
        finish_reason=FinishReason.STOP,
        model=model_name,
        response_id="conformance-stream",
    )

    error_request = ModelRequest((Message.user("Raise the supplied model error."),))
    expected_model_error = ModelUnavailableError(
        "Injected conformance failure.",
        provider="conformance-provider",
        model=model_name,
    )
    cancellation_request = ModelRequest((Message.user("Propagate cancellation."),))
    expected_cancellation = asyncio.CancelledError("Injected conformance cancellation.")

    removal_first_response = ModelResponse(
        Message.assistant("temporary reminder accepted"),
        model=model_name,
        response_id="conformance-reminder-removal-1",
        continuation=ModelContinuation(
            "conformance-provider",
            {"cursor": "reminder-removal"},
        ),
    )
    removal_second_response = ModelResponse(
        Message.assistant("temporary reminder removed"),
        model=model_name,
        response_id="conformance-reminder-removal-2",
    )
    removal_first_request = ModelRequest(
        (Message.user("Apply the temporary control."),),
        reminders=(
            SystemReminder(
                (TextBlock("temporary-control-old"),),
                key="temporary-control",
                scope=ReminderScope.NEXT_REQUEST,
                placement=ReminderPlacement.TAIL,
            ),
        ),
    )
    removal_second_request = ModelRequest(
        (Message.user("The temporary control is now absent."),),
        continuation=removal_first_response.continuation,
    )

    replacement_first_response = ModelResponse(
        Message.assistant("old control accepted"),
        model=model_name,
        response_id="conformance-reminder-replacement-1",
        continuation=ModelContinuation(
            "conformance-provider",
            {"cursor": "reminder-replacement"},
        ),
    )
    replacement_second_response = ModelResponse(
        Message.assistant("new control accepted"),
        model=model_name,
        response_id="conformance-reminder-replacement-2",
    )
    replacement_first_request = ModelRequest(
        (Message.user("Use the old control."),),
        reminders=(
            SystemReminder(
                (TextBlock("persistent-control-old"),),
                key="persistent-control",
                scope=ReminderScope.CONVERSATION,
                placement=ReminderPlacement.INSTRUCTIONS,
                priority=10,
            ),
        ),
    )
    replacement_second_request = ModelRequest(
        (Message.user("Replace the control."),),
        reminders=(
            SystemReminder(
                (TextBlock("persistent-control-new"),),
                key="persistent-control",
                scope=ReminderScope.CONVERSATION,
                placement=ReminderPlacement.TAIL,
                priority=10,
            ),
        ),
        continuation=replacement_first_response.continuation,
    )

    return (
        BackendConformanceScenario(
            BackendConformanceCase.BASIC_RESPONSE,
            model_name,
            basic_request,
            expected_response=basic_response,
        ),
        BackendConformanceScenario(
            BackendConformanceCase.REQUEST_SEMANTICS,
            model_name,
            request_semantics,
            expected_response=semantics_response,
        ),
        BackendConformanceScenario(
            BackendConformanceCase.REMINDER_REMOVAL,
            model_name,
            removal_first_request,
            expected_response=removal_first_response,
            follow_up_steps=(
                BackendConformanceStep(
                    removal_second_request,
                    removal_second_response,
                ),
            ),
        ),
        BackendConformanceScenario(
            BackendConformanceCase.REMINDER_REPLACEMENT,
            model_name,
            replacement_first_request,
            expected_response=replacement_first_response,
            follow_up_steps=(
                BackendConformanceStep(
                    replacement_second_request,
                    replacement_second_response,
                ),
            ),
        ),
        BackendConformanceScenario(
            BackendConformanceCase.STREAMING,
            model_name,
            streaming_request,
            expected_response=streaming_response,
            streaming=True,
        ),
        BackendConformanceScenario(
            BackendConformanceCase.MODEL_ERROR,
            model_name,
            error_request,
            expected_error=expected_model_error,
        ),
        BackendConformanceScenario(
            BackendConformanceCase.CANCELLATION,
            model_name,
            cancellation_request,
            expected_error=expected_cancellation,
        ),
    )


__all__ = [
    "BackendConformanceCase",
    "BackendConformanceError",
    "BackendConformanceReport",
    "BackendConformanceResult",
    "BackendConformanceScenario",
    "BackendConformanceStep",
    "BackendScenarioFactory",
    "run_backend_conformance",
]
