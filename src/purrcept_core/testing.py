"""Provide deterministic in-memory test doubles for agent orchestration.

The helpers in this module record lifecycle events and execute a fixed effect
script without introducing I/O, clocks, or global state. They are intended for
application and extension tests; production runtimes remain responsible for
real execution and event transport.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, TypeAlias, TypeVar

from .context import ExecutionContext
from .effect import Effect
from .events import RunEvent

ScriptItem: TypeAlias = tuple[Effect[Any], object]
"""Expected effect paired with the value or exception produced for it."""
EventT = TypeVar("EventT", bound=RunEvent)


class RecordingEventSink:
    """Record lifecycle events in the exact order they are emitted."""

    __slots__ = ("events",)

    def __init__(self, events: Iterable[RunEvent] = ()) -> None:
        self.events = list(events)

    def emit(self, event: RunEvent) -> None:
        """Append one event without copying or reordering it."""

        self.events.append(event)

    def clear(self) -> None:
        """Discard all recorded events."""

        self.events.clear()

    def of_type(self, event_type: type[EventT]) -> list[EventT]:
        """Return recorded events that are instances of ``event_type``."""

        return [event for event in self.events if isinstance(event, event_type)]

    def __iter__(self) -> Iterator[RunEvent]:
        """Iterate over the live recording in emission order."""

        return iter(self.events)

    def __len__(self) -> int:
        """Return the current number of recorded events."""

        return len(self.events)


class ScriptedExecutor:
    """Execute a fixed sequence of expected effects and outcomes.

    Each script item is ``(expected_effect, outcome)``. An outcome that is a
    ``BaseException`` instance is raised; every other object is returned.
    """

    __slots__ = ("_calls", "_index", "_script")

    def __init__(self, expected: Iterable[ScriptItem]) -> None:
        self._script = tuple(expected)
        self._index = 0
        self._calls: list[tuple[Effect[Any], ExecutionContext[Any]]] = []

    @property
    def calls(self) -> tuple[tuple[Effect[Any], ExecutionContext[Any]], ...]:
        """All attempted executor calls, including a mismatching call."""

        return tuple(self._calls)

    @property
    def effects(self) -> tuple[Effect[Any], ...]:
        """Effects observed so far."""

        return tuple(effect for effect, _context in self._calls)

    @property
    def contexts(self) -> tuple[ExecutionContext[Any], ...]:
        """Execution contexts observed so far."""

        return tuple(context for _effect, context in self._calls)

    @property
    def remaining(self) -> int:
        """Number of script items not yet consumed."""

        return len(self._script) - self._index

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        """Consume one matching script item and return or raise its outcome.

        Every attempted call is recorded before validation so a mismatch or
        exhausted script remains available for diagnosis. A mismatching call
        does not consume the expected item.
        """

        self._calls.append((effect, context))
        call_index = self._index

        if call_index >= len(self._script):
            raise AssertionError(
                f"Unexpected effect at script index {call_index}: {effect!r}; "
                "the script is already exhausted."
            )

        expected_effect, outcome = self._script[call_index]
        if effect != expected_effect:
            raise AssertionError(
                f"Effect mismatch at script index {call_index}: "
                f"expected {expected_effect!r}, got {effect!r}."
            )

        self._index += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def assert_finished(self) -> None:
        """Assert that every configured script item was consumed."""

        if self._index == len(self._script):
            return

        expected_effect, _outcome = self._script[self._index]
        raise AssertionError(
            f"Script has {self.remaining} unconsumed item(s); "
            f"next expected effect at index {self._index} is {expected_effect!r}."
        )


__all__ = ["RecordingEventSink", "ScriptItem", "ScriptedExecutor"]
