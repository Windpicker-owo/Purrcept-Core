"""Define immutable run lifecycle events and their synchronous delivery boundary.

The driver creates the event values in this module and calls an
:class:`EventSink` inline at each lifecycle transition. This module owns local
fan-out and failure-isolation adapters; it intentionally does not queue,
persist, or asynchronously transport events. Runtimes that need those services
must enqueue from a sink and perform the slow work elsewhere.

Event order is the driver's execution order. Sink callbacks must therefore be
synchronous and return ``None`` so delivery cannot outlive or reorder the
transition being observed.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from inspect import isawaitable
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .effect import Effect


def _empty_metadata() -> Mapping[str, Any]:
    return {}


def _require_synchronous_callback(result: object, callback_name: str) -> None:
    """Reject callback results that would weaken inline ordering guarantees."""

    if result is None:
        return
    if isawaitable(result):
        if isinstance(result, Coroutine):
            result.close()
        raise TypeError(f"{callback_name} must be synchronous and return None.")
    raise TypeError(f"{callback_name} must return None, got {type(result).__name__}.")


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Base class for events emitted while driving one agent run."""

    run_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class RunStarted(RunEvent):
    """Emitted before an agent run is first advanced."""

    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EffectEvent(RunEvent):
    """Base class for events associated with one yielded effect."""

    effect_id: str
    step_index: int
    effect: Effect[Any]


@dataclass(frozen=True, slots=True)
class EffectStarted(EffectEvent):
    """Emitted immediately before an effect is passed to the executor."""


@dataclass(frozen=True, slots=True)
class EffectProgress(EffectEvent):
    """An incremental progress update reported by the current effect."""

    payload: object


@dataclass(frozen=True, slots=True)
class EffectSucceeded(EffectEvent):
    """Emitted after an executor returns a result."""

    result: object
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EffectFailed(EffectEvent):
    """Emitted after an executor raises an ordinary exception."""

    error: Exception
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RunSucceeded(RunEvent):
    """Emitted when the run returns its final value."""

    result: object
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RunFailed(RunEvent):
    """Emitted when an ordinary exception escapes the run."""

    error: Exception
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RunCancelled(RunEvent):
    """Emitted when the task driving the run is cancelled."""

    elapsed_seconds: float


@runtime_checkable
class EventSink(Protocol):
    """Contract implemented by synchronous lifecycle-event recipients.

    Applications or runtimes create sinks and pass them to
    :class:`~purrcept_core.driver.AgentDriver`. The driver calls :meth:`emit`
    inline and in lifecycle order, potentially from concurrent run tasks when a
    driver is shared. Implementations own any mutable-state synchronization.

    ``emit`` must return ``None`` and must not return an awaitable. Exceptions
    propagate to the driver, which translates ordinary failures to
    :class:`~purrcept_core.errors.EventDispatchError`.
    """

    def emit(self, event: RunEvent) -> None:
        """Receive one event.

        Implementations should return quickly. A runtime that needs network or
        persistent delivery should enqueue the event here and perform that work
        outside the driver.
        """

        ...


class NullEventSink:
    """Stateless built-in sink that discards every event."""

    __slots__ = ()

    def emit(self, event: RunEvent) -> None:
        """Accept and discard one event."""

        del event


class CallbackEventSink:
    """Adapt one synchronous callback to the :class:`EventSink` contract."""

    __slots__ = ("_callback",)

    def __init__(self, callback: Callable[[RunEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: RunEvent) -> None:
        """Call the configured callback and enforce the sink return contract."""

        result: object = self._callback(event)
        _require_synchronous_callback(result, "Event callback")


class CompositeEventSink:
    """Deliver each event to an immutable, ordered sequence of sinks.

    Dispatch stops at the first failing sink. The constructor snapshots an
    iterable so later caller mutation cannot change delivery order.
    """

    __slots__ = ("_sinks",)

    def __init__(
        self,
        sinks: Iterable[EventSink] | EventSink = (),
        *additional_sinks: EventSink,
    ) -> None:
        if isinstance(sinks, EventSink):
            resolved = (sinks, *additional_sinks)
        else:
            if additional_sinks:
                raise TypeError("additional sinks require the first argument to be an EventSink")
            resolved = tuple(sinks)
        self._sinks = resolved

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """The sinks in dispatch order."""

        return self._sinks

    def emit(self, event: RunEvent) -> None:
        """Dispatch one event to every sink in registration order."""

        for sink in self._sinks:
            result: object = sink.emit(event)
            _require_synchronous_callback(result, "Event sink")


class SafeEventSink:
    """Isolate ordinary exceptions raised by another event sink.

    This adapter is selected explicitly by wrapping an existing sink. It
    suppresses :class:`Exception` instances after optionally reporting them to
    another synchronous callback; cancellation-like ``BaseException`` values
    remain visible to the driver.
    """

    __slots__ = ("_on_error", "_sink")

    def __init__(
        self,
        sink: EventSink,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._sink = sink
        self._on_error = on_error

    def emit(self, event: RunEvent) -> None:
        """Forward one event while containing ordinary observer failures."""

        try:
            result: object = self._sink.emit(event)
            _require_synchronous_callback(result, "Event sink")
        except Exception as error:
            if self._on_error is not None:
                result: object = self._on_error(error)
                _require_synchronous_callback(result, "SafeEventSink error callback")


__all__ = [
    "CallbackEventSink",
    "CompositeEventSink",
    "EffectEvent",
    "EffectFailed",
    "EffectProgress",
    "EffectStarted",
    "EffectSucceeded",
    "EventSink",
    "NullEventSink",
    "RunCancelled",
    "RunEvent",
    "RunFailed",
    "RunStarted",
    "RunSucceeded",
    "SafeEventSink",
]
