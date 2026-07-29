"""Define stable orchestration errors shared across the core package.

The hierarchy distinguishes invalid run operations, unsupported capability
requests, and event-infrastructure failures from application exceptions raised
by executors. The driver preserves these distinctions when deciding whether an
error may enter agent code.
"""

from typing import Any


class PurrceptError(Exception):
    """Base class for errors raised by Purrcept Core."""


class InvalidRunStateError(PurrceptError):
    """Raised when an agent run is advanced from an invalid state."""


class UnsupportedEffectError(PurrceptError):
    """Raised when an executor cannot handle an effect."""

    def __init__(self, effect: object) -> None:
        self.effect = effect
        effect_type = type(effect)
        type_name = f"{effect_type.__module__}.{effect_type.__qualname__}"
        super().__init__(f"No executor supports effect type {type_name}.")


class EventDispatchError(PurrceptError):
    """Raised when a lifecycle event cannot be delivered to its sink.

    The failed event remains available on :attr:`event`; the original sink
    failure is attached as the exception cause by the driver.
    """

    def __init__(self, event: object) -> None:
        self.event = event
        event_type = type(event)
        type_name = f"{event_type.__module__}.{event_type.__qualname__}"
        run_id: Any = getattr(event, "run_id", None)
        run_detail = "" if run_id is None else f" for run {run_id!r}"
        super().__init__(f"Failed to dispatch {type_name}{run_detail}.")


__all__ = [
    "EventDispatchError",
    "InvalidRunStateError",
    "PurrceptError",
    "UnsupportedEffectError",
]
