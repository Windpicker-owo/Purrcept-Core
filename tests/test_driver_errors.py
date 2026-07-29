"""Verify driver failure routing, cleanup, and primary-error precedence.

The scenarios distinguish effect failures from event infrastructure failures,
exercise custom run contract violations, and assert that close or observer
errors cannot replace cancellation or the original execution outcome.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.driver import AgentDriver
from purrcept_core.effect import Effect
from purrcept_core.errors import EventDispatchError
from purrcept_core.events import (
    EffectFailed,
    EffectProgress,
    EffectStarted,
    EffectSucceeded,
    RunEvent,
    RunFailed,
    RunStarted,
    RunSucceeded,
)
from purrcept_core.flow import AgentFlow
from purrcept_core.middleware import MiddlewareExecutor, RetryMiddleware
from purrcept_core.testing import RecordingEventSink, ScriptedExecutor


@dataclass(frozen=True, slots=True)
class Work(Effect[int]):
    name: str


@pytest.mark.asyncio
async def test_effect_error_can_be_caught_by_flow() -> None:
    failure = ValueError("primary failed")

    def flow() -> AgentFlow[int]:
        try:
            yield Work("primary")
        except ValueError:
            return (yield Work("fallback"))
        raise AssertionError("unreachable")

    events = RecordingEventSink()
    executor = ScriptedExecutor(
        [
            (Work("primary"), failure),
            (Work("fallback"), 7),
        ]
    )

    result = await AgentDriver(executor, event_sink=events).run(flow(), host=None)

    assert result == 7
    assert events.of_type(EffectFailed)[0].error is failure
    assert not events.of_type(RunFailed)
    assert len(events.of_type(RunSucceeded)) == 1
    executor.assert_finished()


@pytest.mark.asyncio
async def test_unhandled_effect_error_fails_and_closes_run() -> None:
    failure = LookupError("missing")
    finalized = False

    def flow() -> AgentFlow[None]:
        nonlocal finalized
        try:
            yield Work("primary")
        finally:
            finalized = True

    events = RecordingEventSink()
    with pytest.raises(LookupError) as caught:
        await AgentDriver(
            ScriptedExecutor([(Work("primary"), failure)]),
            event_sink=events,
        ).run(flow(), host=None)

    assert caught.value is failure
    assert finalized
    assert [type(event) for event in events] == [
        RunStarted,
        EffectStarted,
        EffectFailed,
        RunFailed,
    ]
    assert events.of_type(RunFailed)[0].error is failure


@pytest.mark.asyncio
async def test_flow_failure_before_first_yield_emits_run_failed() -> None:
    failure = RuntimeError("flow failed")

    def flow() -> AgentFlow[None]:
        raise failure
        yield Work("unreachable")

    events = RecordingEventSink()
    with pytest.raises(RuntimeError) as caught:
        await AgentDriver(ScriptedExecutor([]), event_sink=events).run(flow(), host=None)

    assert caught.value is failure
    assert [type(event) for event in events] == [RunStarted, RunFailed]


@pytest.mark.asyncio
async def test_flow_failure_after_result_is_distinct_from_effect_failure() -> None:
    failure = RuntimeError("post-result flow failure")

    def flow() -> AgentFlow[None]:
        yield Work("step")
        raise failure

    events = RecordingEventSink()
    with pytest.raises(RuntimeError) as caught:
        await AgentDriver(
            ScriptedExecutor([(Work("step"), 1)]),
            event_sink=events,
        ).run(flow(), host=None)

    assert caught.value is failure
    assert len(events.of_type(EffectSucceeded)) == 1
    assert not events.of_type(EffectFailed)
    assert events.of_type(RunFailed)[0].error is failure


@pytest.mark.asyncio
async def test_progress_sink_failure_is_not_thrown_into_flow() -> None:
    class ObserverFailure(RuntimeError):
        pass

    observer_failure = ObserverFailure("observer unavailable")
    recorded: list[RunEvent] = []
    caught_by_flow = False
    finalized = False
    attempts = 0

    class FailingSink:
        def emit(self, event: RunEvent) -> None:
            recorded.append(event)
            if isinstance(event, EffectProgress):
                raise observer_failure

    class ProgressExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> int:
            nonlocal attempts
            attempts += 1
            context.progress("halfway")
            return 1

    def flow() -> AgentFlow[int]:
        nonlocal caught_by_flow, finalized
        try:
            try:
                return (yield Work("step"))
            except Exception:
                caught_by_flow = True
                return -1
        finally:
            finalized = True

    executor = MiddlewareExecutor(
        ProgressExecutor(),
        [RetryMiddleware(max_attempts=3, predicate=lambda _error: True)],
    )
    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(executor, event_sink=FailingSink()).run(flow(), host=None)

    assert isinstance(caught.value.event, EffectProgress)
    assert caught.value.__cause__ is observer_failure
    assert not caught_by_flow
    assert finalized
    assert attempts == 1
    assert not any(isinstance(event, EffectFailed) for event in recorded)
    assert not any(isinstance(event, RunFailed) for event in recorded)


@pytest.mark.asyncio
async def test_progress_sink_failure_cannot_be_suppressed_by_executor() -> None:
    observer_failure = RuntimeError("observer unavailable")
    recorded: list[RunEvent] = []
    finalized = False

    class FailingSink:
        def emit(self, event: RunEvent) -> None:
            recorded.append(event)
            if isinstance(event, EffectProgress):
                raise observer_failure

    class SuppressingExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> int:
            first_error: EventDispatchError | None = None
            try:
                context.progress("halfway")
            except EventDispatchError as error:
                first_error = error
            with pytest.raises(EventDispatchError) as repeated:
                context.progress("again")
            assert repeated.value is first_error
            return 1

    def flow() -> AgentFlow[int]:
        nonlocal finalized
        try:
            return (yield Work("step"))
        finally:
            finalized = True

    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(SuppressingExecutor(), event_sink=FailingSink()).run(
            flow(),
            host=None,
        )

    assert caught.value.__cause__ is observer_failure
    assert finalized
    assert len([event for event in recorded if isinstance(event, EffectProgress)]) == 1
    assert not any(isinstance(event, EffectSucceeded) for event in recorded)
    assert not any(isinstance(event, EffectFailed) for event in recorded)


@pytest.mark.asyncio
async def test_latched_progress_failure_wins_over_a_later_executor_error() -> None:
    observer_failure = RuntimeError("observer unavailable")
    executor_failure = ValueError("executor transformed the failure")

    class FailingSink:
        def emit(self, event: RunEvent) -> None:
            if isinstance(event, EffectProgress):
                raise observer_failure

    class TransformingExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> int:
            try:
                context.progress("halfway")
            except EventDispatchError:
                raise executor_failure from None
            return 1

    def flow() -> AgentFlow[int]:
        return (yield Work("step"))

    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(TransformingExecutor(), event_sink=FailingSink()).run(
            flow(),
            host=None,
        )

    assert caught.value.__cause__ is observer_failure
    assert caught.value.__context__ is executor_failure
    assert any("executor transformed" in note for note in caught.value.__notes__)


@pytest.mark.asyncio
async def test_non_exception_base_exception_is_only_cleaned_up() -> None:
    class StopRun(BaseException):
        pass

    stop = StopRun()
    caught_by_flow: BaseException | None = None
    finalized = False

    class StoppingExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> Any:
            raise stop

    def flow() -> AgentFlow[None]:
        nonlocal caught_by_flow, finalized
        try:
            try:
                yield Work("step")
            except BaseException as error:
                caught_by_flow = error
        finally:
            finalized = True

    events = RecordingEventSink()
    with pytest.raises(StopRun) as caught:
        await AgentDriver(StoppingExecutor(), event_sink=events).run(flow(), host=None)

    assert caught.value is stop
    # Closing a suspended generator injects GeneratorExit. The executor's
    # StopRun instance itself was never thrown into flow code.
    assert isinstance(caught_by_flow, GeneratorExit)
    assert finalized
    assert [type(event) for event in events] == [RunStarted, EffectStarted]


@pytest.mark.asyncio
async def test_invalid_custom_run_state_fails_and_closes_run() -> None:
    class InvalidStateRun:
        closed = False

        def start(self) -> object:
            return object()

        def send(self, value: object) -> object:
            return value

        def throw(self, error: BaseException) -> object:
            return error

        def close(self) -> None:
            self.closed = True

    run = InvalidStateRun()
    events = RecordingEventSink()

    with pytest.raises(TypeError, match="unsupported state"):
        await AgentDriver(ScriptedExecutor([]), event_sink=events).run(run, host=None)

    assert run.closed
    assert isinstance(events.events[-1], RunFailed)


@pytest.mark.asyncio
async def test_close_failure_is_not_allowed_to_replace_primary_error() -> None:
    primary = RuntimeError("start failed")
    cleanup = KeyboardInterrupt("close failed")

    class FailingRun:
        def start(self) -> object:
            raise primary

        def send(self, value: object) -> object:
            return value

        def throw(self, error: BaseException) -> object:
            return error

        def close(self) -> None:
            raise cleanup

    with pytest.raises(RuntimeError) as caught:
        await AgentDriver(ScriptedExecutor([])).run(FailingRun(), host=None)

    assert caught.value is primary
    assert any("close() also raised" in note for note in primary.__notes__)


@pytest.mark.asyncio
async def test_existing_event_dispatch_error_is_propagated_unchanged() -> None:
    sentinel_event = RunStarted(
        run_id="sentinel",
        occurred_at=datetime.now(UTC),
    )
    dispatch_error = EventDispatchError(sentinel_event)

    class ClosingRun:
        closed = False

        def start(self) -> object:
            raise AssertionError("event dispatch should fail first")

        def send(self, value: object) -> object:
            return value

        def throw(self, error: BaseException) -> object:
            return error

        def close(self) -> None:
            self.closed = True

    class ExistingErrorSink:
        def emit(self, event: RunEvent) -> None:
            raise dispatch_error

    run = ClosingRun()
    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(ScriptedExecutor([]), event_sink=ExistingErrorSink()).run(
            run,
            host=None,
        )

    assert caught.value is dispatch_error
    assert run.closed


@pytest.mark.asyncio
async def test_preflight_failure_closes_custom_run() -> None:
    failure = RuntimeError("id factory failed")

    class ClosingRun:
        closed = False

        def start(self) -> object:
            raise AssertionError("run must not start")

        def send(self, value: object) -> object:
            return value

        def throw(self, error: BaseException) -> object:
            return error

        def close(self) -> None:
            self.closed = True

    def fail_id() -> str:
        raise failure

    run = ClosingRun()
    with pytest.raises(RuntimeError) as caught:
        await AgentDriver(ScriptedExecutor([]), id_factory=fail_id).run(run, host=None)

    assert caught.value is failure
    assert run.closed


@pytest.mark.asyncio
async def test_async_event_sink_is_rejected_as_infrastructure_failure() -> None:
    class AsyncSink:
        async def emit(self, event: RunEvent) -> None:
            return None

    def flow() -> AgentFlow[str]:
        if False:
            yield Work("never")
        return "done"

    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(
            ScriptedExecutor([]),
            event_sink=AsyncSink(),  # type: ignore[arg-type]
        ).run(flow(), host=None)

    assert isinstance(caught.value.__cause__, TypeError)
    assert "synchronous" in str(caught.value.__cause__)


@pytest.mark.asyncio
async def test_custom_awaitable_event_sink_result_is_rejected() -> None:
    class CustomAwaitable:
        def __await__(self) -> Any:
            if False:
                yield None
            return None

    class AwaitableSink:
        def emit(self, event: RunEvent) -> Any:
            return CustomAwaitable()

    def flow() -> AgentFlow[str]:
        if False:
            yield Work("never")
        return "done"

    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(
            ScriptedExecutor([]),
            event_sink=AwaitableSink(),
        ).run(flow(), host=None)

    assert isinstance(caught.value.__cause__, TypeError)
    assert "synchronous" in str(caught.value.__cause__)


@pytest.mark.asyncio
async def test_non_none_event_sink_result_is_rejected() -> None:
    class ReturningSink:
        def emit(self, event: RunEvent) -> Any:
            return 1

    def flow() -> AgentFlow[str]:
        if False:
            yield Work("never")
        return "done"

    with pytest.raises(EventDispatchError) as caught:
        await AgentDriver(
            ScriptedExecutor([]),
            event_sink=ReturningSink(),
        ).run(flow(), host=None)

    assert isinstance(caught.value.__cause__, TypeError)
    assert "got int" in str(caught.value.__cause__)
