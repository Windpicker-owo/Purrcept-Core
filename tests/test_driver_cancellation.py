"""Verify driver cancellation closes runs and remains primary over observer failure."""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.driver import AgentDriver
from purrcept_core.effect import Effect
from purrcept_core.errors import EventDispatchError
from purrcept_core.events import (
    EffectFailed,
    RunCancelled,
    RunEvent,
    RunFailed,
)
from purrcept_core.flow import AgentFlow
from purrcept_core.testing import RecordingEventSink


@dataclass(frozen=True, slots=True)
class Wait(Effect[None]):
    label: str


@pytest.mark.asyncio
async def test_cancellation_closes_flow_emits_event_and_propagates() -> None:
    entered = asyncio.Event()
    finalized = False
    closed_before_event = False
    recording = RecordingEventSink()

    class WaitingExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> None:
            entered.set()
            await asyncio.Future()

    class CheckingSink:
        def emit(self, event: RunEvent) -> None:
            nonlocal closed_before_event
            recording.emit(event)
            if isinstance(event, RunCancelled):
                closed_before_event = finalized

    def flow() -> AgentFlow[None]:
        nonlocal finalized
        try:
            yield Wait("forever")
        finally:
            finalized = True

    task = asyncio.create_task(
        AgentDriver(WaitingExecutor(), event_sink=CheckingSink()).run(flow(), host=None)
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finalized
    assert closed_before_event
    assert len(recording.of_type(RunCancelled)) == 1
    assert not recording.of_type(EffectFailed)
    assert not recording.of_type(RunFailed)


@pytest.mark.asyncio
async def test_cancel_remains_primary_when_cancel_event_dispatch_fails() -> None:
    entered = asyncio.Event()
    finalized = False
    observer_failure = RuntimeError("observer failed")

    class WaitingExecutor:
        async def execute(
            self,
            effect: Effect[Any],
            context: ExecutionContext[None],
        ) -> None:
            entered.set()
            await asyncio.Future()

    class FailingSink:
        def emit(self, event: RunEvent) -> None:
            if isinstance(event, RunCancelled):
                raise observer_failure

    def flow() -> AgentFlow[None]:
        nonlocal finalized
        try:
            yield Wait("forever")
        finally:
            finalized = True

    task = asyncio.create_task(
        AgentDriver(WaitingExecutor(), event_sink=FailingSink()).run(flow(), host=None)
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert task.cancelled()
    assert finalized
    assert isinstance(caught.value.__cause__, EventDispatchError)
    assert caught.value.__cause__.__cause__ is observer_failure
    assert any("RunCancelled" in note for note in caught.value.__notes__)
