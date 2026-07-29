"""Project current runtime state into request-local reminders synchronously.

Applications implement :class:`ReminderSource` when prompt guidance must be
refreshed before every model request, including later rounds after tool calls.
Conversation construction snapshots sources and enforces unique stable IDs;
prompt compilation resolves them in registration order. Dynamic results are
never stored in :class:`ReminderState` or committed to conversation history.
"""

from __future__ import annotations

from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from ._utils import require_non_empty_string
from .context_policy import PreparedContext
from .model import Model
from .reminders import SystemReminder

if TYPE_CHECKING:
    from .conversation import ConversationState


@dataclass(frozen=True, slots=True)
class ReminderResolutionContext:
    """Immutable inputs available to one dynamic reminder projection."""

    model: Model
    conversation: ConversationState
    prepared_context: PreparedContext
    turn_index: int
    model_round: int

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.model), Model):
            raise TypeError("model must be a Model.")
        _require_conversation_state(self.conversation)
        if not isinstance(cast(object, self.prepared_context), PreparedContext):
            raise TypeError("prepared_context must be a PreparedContext.")
        _validate_positive_int(self.turn_index, field_name="turn_index")
        _validate_positive_int(self.model_round, field_name="model_round")


@runtime_checkable
class ReminderSource(Protocol):
    """Extension contract for request-time prompt-state projection.

    Applications create source instances and register them on one
    :class:`~purrcept_core.models.conversation.Conversation`. The prompt session
    calls :meth:`resolve` once per source and model round, in registration
    order, with immutable conversation and prepared-context snapshots.

    ``source_id`` must be stable and unique within the conversation. Resolution
    must be synchronous and return only :class:`SystemReminder` values; returned
    reminders are request-local. Source failures and contract violations are
    traced when configured and then propagate unchanged. The core provides no
    built-in sources, so registration is explicit.
    """

    @property
    def source_id(self) -> str:
        """Return a stable identifier unique within one Conversation."""

        ...

    def resolve(
        self,
        context: ReminderResolutionContext,
    ) -> Iterable[SystemReminder]:
        """Return current reminders without yielding Effects or awaitables."""

        ...


def normalize_reminder_sources(value: object) -> tuple[ReminderSource, ...]:
    """Validate, de-duplicate, and snapshot a conversation's sources in order."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("reminder_sources must be an iterable of ReminderSource instances.")
    sources = tuple(cast(Iterable[object], value))
    normalized: list[ReminderSource] = []
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, ReminderSource):
            raise TypeError("reminder_sources must contain only ReminderSource instances.")
        source_id = require_non_empty_string(source.source_id, field_name="source_id")
        if source_id in source_ids:
            raise ValueError(f"Reminder source {source_id!r} is registered more than once.")
        source_ids.add(source_id)
        normalized.append(source)
    return tuple(normalized)


def resolve_reminder_source(
    source: ReminderSource,
    context: ReminderResolutionContext,
) -> tuple[SystemReminder, ...]:
    """Resolve one source synchronously without hiding implementation failures."""

    source_id = require_non_empty_string(source.source_id, field_name="source_id")
    del source_id
    resolved = cast(object, source.resolve(context))
    if isawaitable(resolved):
        if isinstance(resolved, Coroutine):
            resolved.close()
        raise TypeError("ReminderSource.resolve() must be synchronous.")
    if isinstance(resolved, (str, bytes)) or not isinstance(resolved, Iterable):
        raise TypeError("ReminderSource.resolve() must return an iterable of SystemReminder.")
    reminders = tuple(cast(Iterable[object], resolved))
    for reminder in reminders:
        if isawaitable(reminder):
            if isinstance(reminder, Coroutine):
                reminder.close()
            raise TypeError("ReminderSource.resolve() must not yield awaitable values.")
        if not isinstance(reminder, SystemReminder):
            raise TypeError("ReminderSource.resolve() must return only SystemReminder instances.")
    return cast(tuple[SystemReminder, ...], reminders)


def _require_conversation_state(value: object) -> None:
    from .conversation import ConversationState

    if not isinstance(value, ConversationState):
        raise TypeError("conversation must be a ConversationState.")


def _validate_positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")


__all__ = ["ReminderResolutionContext", "ReminderSource"]
