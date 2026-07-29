"""Verify deterministic scripted executor and recording event-sink test doubles."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.effect import Effect
from purrcept_core.events import RunStarted, RunSucceeded
from purrcept_core.testing import RecordingEventSink, ScriptedExecutor


@dataclass(frozen=True, slots=True)
class Query(Effect[str]):
    text: str


def _context() -> ExecutionContext[None]:
    return ExecutionContext(
        run_id="run",
        effect_id="effect",
        step_index=0,
        host=None,
    )


@pytest.mark.asyncio
async def test_scripted_executor_returns_and_raises_scripted_outcomes() -> None:
    failure = ValueError("no result")
    executor = ScriptedExecutor(
        [
            (Query("first"), "answer"),
            (Query("second"), failure),
        ]
    )

    assert await executor.execute(Query("first"), _context()) == "answer"
    with pytest.raises(ValueError) as caught:
        await executor.execute(Query("second"), _context())

    assert caught.value is failure
    executor.assert_finished()
    assert executor.effects == (Query("first"), Query("second"))
    assert executor.remaining == 0


@pytest.mark.asyncio
async def test_scripted_executor_describes_mismatch_and_exhaustion() -> None:
    executor = ScriptedExecutor([(Query("expected"), "answer")])

    with pytest.raises(AssertionError, match="script index 0") as mismatch:
        await executor.execute(Query("actual"), _context())
    assert "expected" in str(mismatch.value)
    assert "actual" in str(mismatch.value)

    assert await executor.execute(Query("expected"), _context()) == "answer"
    with pytest.raises(AssertionError, match="already exhausted"):
        await executor.execute(Query("extra"), _context())


def test_scripted_executor_assert_finished_reports_next_effect() -> None:
    executor = ScriptedExecutor([(Query("pending"), "answer")])

    with pytest.raises(AssertionError, match="pending"):
        executor.assert_finished()


def test_recording_event_sink_preserves_order_filters_and_clears() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    started = RunStarted("run", at)
    succeeded = RunSucceeded("run", at, result=3, elapsed_seconds=0.5)
    sink = RecordingEventSink([started])

    sink.emit(succeeded)

    assert list(sink) == [started, succeeded]
    assert len(sink) == 2
    assert sink.of_type(RunSucceeded) == [succeeded]
    sink.clear()
    assert list(sink) == []
