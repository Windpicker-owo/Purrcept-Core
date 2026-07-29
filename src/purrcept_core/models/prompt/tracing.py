"""Emit synchronous metadata-only diagnostics for prompt compilation.

Prompt sessions create these immutable events while resolving dynamic reminder
sources, accepting a compiled request, and consuming request-scoped reminders.
The trace deliberately stores keys, hashes, counts, and error type names rather
than raw prompt text. Delivery is inline and observational: it does not own
event persistence or asynchronous transport.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TypeAlias, cast

from .._utils import require_non_empty_string
from ..reminders import ReminderPlacement, ReminderScope


class PromptTraceEvent:
    """Extension marker for immutable prompt-control diagnostic snapshots."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class PromptCompiled(PromptTraceEvent):
    """One complete request was accepted for model generation."""

    model_round: int
    prompt_fingerprint: str
    reminder_keys: tuple[str | None, ...]
    reminder_scopes: tuple[ReminderScope | None, ...]
    reminder_placements: tuple[ReminderPlacement, ...]
    reminder_priorities: tuple[int, ...]
    content_hashes: tuple[str, ...]
    dynamic_source_ids: tuple[str, ...] = field(default=(), kw_only=True)
    estimated_tokens: int | None = field(default=None, kw_only=True)
    trimmed_turns: int = field(default=0, kw_only=True)


@dataclass(frozen=True, slots=True)
class ReminderConsumed(PromptTraceEvent):
    """NEXT_REQUEST reminders were consumed by a compiled request."""

    model_round: int
    reminder_keys: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class ReminderSourceResolved(PromptTraceEvent):
    """One dynamic source produced request-local reminders."""

    model_round: int
    source_id: str
    reminder_keys: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class ReminderSourceFailed(PromptTraceEvent):
    """One dynamic source failed before a request could be compiled."""

    model_round: int
    source_id: str
    error_type: str


PromptTraceSink: TypeAlias = Callable[[PromptTraceEvent], None]
"""Synchronous callback invoked in prompt lifecycle order.

Applications register a sink on ``Conversation``. It must return ``None`` and
must not return an awaitable; callback failures abort the current uncommitted
turn rather than being silently ignored.
"""


def validate_prompt_trace_sink(value: object) -> None:
    """Validate an optional trace callback."""

    if value is not None and not callable(value):
        raise TypeError("prompt_trace_sink must be callable or None.")


def emit_prompt_trace(
    sink: PromptTraceSink | None,
    event: PromptTraceEvent,
) -> None:
    """Emit one event inline, rejecting async or value-returning sinks."""

    if sink is None:
        return
    result: object = cast(Callable[[PromptTraceEvent], object], sink)(event)
    if isawaitable(result):
        if isinstance(result, Coroutine):
            result.close()
        raise TypeError("PromptTraceSink must be synchronous.")
    if result is not None:
        raise TypeError("PromptTraceSink must return None.")


def source_failure_event(
    *,
    model_round: int,
    source_id: str,
    error: BaseException,
) -> ReminderSourceFailed:
    """Create a metadata-only source failure event."""

    source_id = require_non_empty_string(source_id, field_name="source_id")
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return ReminderSourceFailed(model_round, source_id, error_type)


__all__ = [
    "PromptCompiled",
    "PromptTraceEvent",
    "PromptTraceSink",
    "ReminderConsumed",
    "ReminderSourceFailed",
    "ReminderSourceResolved",
]
