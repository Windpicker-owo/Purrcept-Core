"""Verify successful driver lifecycles, event order, iteration, and nested flows."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.driver import AgentDriver
from purrcept_core.effect import Effect
from purrcept_core.errors import InvalidRunStateError
from purrcept_core.events import (
    EffectProgress,
    EffectStarted,
    EffectSucceeded,
    RunStarted,
    RunSucceeded,
)
from purrcept_core.flow import AgentFlow
from purrcept_core.run import Returned, RunState, Yielded
from purrcept_core.testing import RecordingEventSink


@dataclass(frozen=True, slots=True)
class Number(Effect[int]):
    value: int


class NumberExecutor:
    def __init__(self) -> None:
        self.contexts: list[ExecutionContext[Any]] = []

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        assert isinstance(effect, Number)
        self.contexts.append(context)
        context.progress({"value": effect.value})
        return effect.value


def _clock() -> Iterator[datetime]:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    index = 0
    while True:
        yield origin + timedelta(seconds=index)
        index += 1


@pytest.mark.asyncio
async def test_driver_runs_multiple_effects_with_deterministic_events() -> None:
    def flow() -> AgentFlow[int]:
        first = yield Number(2)
        second = yield Number(first + 3)
        return second * 2

    executor = NumberExecutor()
    events = RecordingEventSink()
    ids = iter(("run-id", "effect-1", "effect-2"))
    clock_values = _clock()
    monotonic_values = iter((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    metadata = {"trace": "yes"}
    driver = AgentDriver(
        executor,
        event_sink=events,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        id_factory=lambda: next(ids),
    )

    result = await driver.run(flow(), host={"runtime": 1}, metadata=metadata)

    assert result == 10
    assert [type(event) for event in events] == [
        RunStarted,
        EffectStarted,
        EffectProgress,
        EffectSucceeded,
        EffectStarted,
        EffectProgress,
        EffectSucceeded,
        RunSucceeded,
    ]
    assert [event.effect_id for event in events.of_type(EffectStarted)] == [
        "effect-1",
        "effect-2",
    ]
    assert [event.step_index for event in events.of_type(EffectStarted)] == [0, 1]
    assert [event.elapsed_seconds for event in events.of_type(EffectSucceeded)] == [1.0, 1.0]
    assert events.of_type(RunSucceeded)[0].elapsed_seconds == 5.0
    assert all(event.run_id == "run-id" for event in events)
    assert executor.contexts[0].metadata == metadata
    metadata["trace"] = "changed"
    assert executor.contexts[0].metadata == {"trace": "yes"}


@pytest.mark.asyncio
async def test_explicit_run_id_does_not_consume_an_id() -> None:
    def flow() -> AgentFlow[int]:
        return (yield Number(4))

    ids = iter(("effect-id",))
    events = RecordingEventSink()
    result = await AgentDriver(
        NumberExecutor(),
        event_sink=events,
        id_factory=lambda: next(ids),
    ).run(flow(), host=None, run_id="provided")

    assert result == 4
    assert events.of_type(EffectStarted)[0].effect_id == "effect-id"
    assert all(event.run_id == "provided" for event in events)


@pytest.mark.asyncio
async def test_ten_thousand_effects_use_iterative_driver_loop() -> None:
    class Noop(Effect[None]):
        __slots__ = ()

    class NoopExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> None:
            assert isinstance(effect, Noop)
            assert context.step_index >= 0

    def flow() -> AgentFlow[int]:
        count = 0
        for _ in range(10_000):
            yield Noop()
            count += 1
        return count

    assert await AgentDriver(NoopExecutor()).run(flow(), host=None) == 10_000


@pytest.mark.asyncio
async def test_default_clock_and_id_factory_create_traceable_events() -> None:
    def flow() -> AgentFlow[int]:
        return (yield Number(4))

    executor = NumberExecutor()
    events = RecordingEventSink()
    driver = AgentDriver(executor, event_sink=events)

    result = await driver.run(flow(), host=None)

    started = events.of_type(RunStarted)[0]
    effect_started = events.of_type(EffectStarted)[0]
    assert result == 4
    assert driver.executor is executor
    assert driver.event_sink is events
    assert started.occurred_at.tzinfo is UTC
    assert len(started.run_id) == 32
    assert len(effect_started.effect_id) == 32
    assert effect_started.effect_id != started.run_id


@pytest.mark.asyncio
async def test_retained_context_cannot_emit_progress_after_effect_finishes() -> None:
    def flow() -> AgentFlow[int]:
        return (yield Number(4))

    executor = NumberExecutor()
    events = RecordingEventSink()

    assert await AgentDriver(executor, event_sink=events).run(flow(), host=None) == 4
    event_count = len(events)

    with pytest.raises(InvalidRunStateError, match="finished"):
        executor.contexts[0].progress("late")

    assert len(events) == event_count


@pytest.mark.asyncio
async def test_driver_accepts_a_non_generator_agent_run() -> None:
    class ExplicitRun:
        started = False
        closed = False

        def start(self) -> RunState[int]:
            self.started = True
            return Yielded(Number(3))

        def send(self, value: object) -> RunState[int]:
            assert value == 3
            return Returned(6)

        def throw(self, error: BaseException) -> RunState[int]:
            raise error

        def close(self) -> None:
            self.closed = True

    run = ExplicitRun()

    result = await AgentDriver(NumberExecutor()).run(run, host=None)

    assert result == 6
    assert run.started
    assert not run.closed


@pytest.mark.asyncio
async def test_nested_flow_uses_one_run_id_and_continuous_step_indexes() -> None:
    def child(value: int) -> AgentFlow[int]:
        return (yield Number(value))

    def parent() -> AgentFlow[int]:
        first = yield from child(2)
        return (yield from child(first + 1))

    events = RecordingEventSink()

    result = await AgentDriver(NumberExecutor(), event_sink=events).run(
        parent(),
        host=None,
        run_id="nested-run",
    )

    effect_events = events.of_type(EffectStarted)
    assert result == 3
    assert [event.run_id for event in effect_events] == ["nested-run", "nested-run"]
    assert [event.step_index for event in effect_events] == [0, 1]
