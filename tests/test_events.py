"""Verify immutable lifecycle events and synchronous sink composition contracts."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from purrcept_core.events import (
    CallbackEventSink,
    CompositeEventSink,
    NullEventSink,
    RunStarted,
    SafeEventSink,
)


def test_run_started_copies_metadata_and_is_frozen() -> None:
    source = {"request": "abc"}
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC), source)

    source["request"] = "changed"

    assert event.metadata == {"request": "abc"}
    with pytest.raises(TypeError):
        event.metadata["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        event.run_id = "other"  # type: ignore[misc]


def test_null_callback_and_composite_sinks_dispatch_in_order() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    received: list[tuple[str, object]] = []
    first = CallbackEventSink(lambda item: received.append(("first", item)))
    second = CallbackEventSink(lambda item: received.append(("second", item)))

    NullEventSink().emit(event)
    CompositeEventSink(first, second).emit(event)

    assert received == [("first", event), ("second", event)]


def test_composite_accepts_an_iterable_of_sinks() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    received: list[object] = []

    CompositeEventSink(
        [
            CallbackEventSink(received.append),
            CallbackEventSink(received.append),
        ]
    ).emit(event)

    assert received == [event, event]


def test_safe_sink_reports_and_isolates_ordinary_errors() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    failure = RuntimeError("observer failed")
    reported: list[Exception] = []

    def fail(_event: object) -> None:
        raise failure

    SafeEventSink(CallbackEventSink(fail), reported.append).emit(event)

    assert reported == [failure]


def test_safe_sink_does_not_swallow_base_exceptions() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))

    def stop(_event: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SafeEventSink(CallbackEventSink(stop)).emit(event)


def test_safe_sink_can_silently_isolate_errors_when_requested() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))

    def fail(_event: object) -> None:
        raise RuntimeError("observer failed")

    SafeEventSink(CallbackEventSink(fail)).emit(event)


def test_composite_rejects_ambiguous_iterable_and_positional_sinks() -> None:
    sink = NullEventSink()

    with pytest.raises(TypeError, match="additional sinks"):
        CompositeEventSink([sink], sink)


def test_callback_sink_rejects_an_async_callback() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))

    async def callback(_event: object) -> None:
        return None

    sink = CallbackEventSink(callback)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be synchronous"):
        sink.emit(event)


def test_safe_sink_rejects_an_async_error_callback() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))

    def fail(_event: object) -> None:
        raise RuntimeError("observer failed")

    async def report(_error: Exception) -> None:
        return None

    sink = SafeEventSink(
        CallbackEventSink(fail),
        report,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="must be synchronous"):
        sink.emit(event)


def test_callback_sink_rejects_a_non_none_result() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    sink = CallbackEventSink(lambda _event: 1)  # type: ignore[arg-type, return-value]

    with pytest.raises(TypeError, match=r"must return None.*int"):
        sink.emit(event)


def test_callback_sink_rejects_a_non_coroutine_awaitable() -> None:
    class CustomAwaitable:
        def __await__(self) -> object:
            if False:
                yield None
            return None

    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    sink = CallbackEventSink(
        lambda _event: CustomAwaitable(),  # type: ignore[arg-type, return-value]
    )

    with pytest.raises(TypeError, match="must be synchronous"):
        sink.emit(event)


def test_safe_sink_passes_through_a_successful_event() -> None:
    event = RunStarted("run-1", datetime(2026, 1, 1, tzinfo=UTC))
    received: list[object] = []

    SafeEventSink(CallbackEventSink(received.append)).emit(event)

    assert received == [event]
