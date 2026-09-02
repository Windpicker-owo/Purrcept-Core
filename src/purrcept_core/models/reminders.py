"""Model scoped prompt-control state independently of committed history.

``SystemReminder`` values describe guidance, lifetime, placement, and priority.
``ReminderState`` owns an immutable ordered collection between turns. Keyed
upserts replace in place to preserve stable ordering, while unkeyed reminders
append; actual per-request acquisition and consumption is owned by the prompt
compiler's turn session.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from ._utils import validate_optional_non_empty_string
from .content import ContentBlock


class ReminderScope(StrEnum):
    """How long a system reminder remains applicable."""

    NEXT_REQUEST = "next_request"
    TURN = "turn"
    CONVERSATION = "conversation"


class ReminderPlacement(StrEnum):
    """Where a provider adapter should place a reminder relative to history.

    Placement is position, not wire role. Reminders are request-local control
    text; Chat Completions adapters must emit them as user messages.
    ``SystemInstruction`` is the only system-role input.
    """

    AUTO = "auto"
    INSTRUCTIONS = "instructions"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class SystemReminder:
    """Immutable request-local guidance with lifetime, placement, and ordering.

    Reminders never become canonical history. Adapters must not serialize them
    as the system role; that role is reserved for ``SystemInstruction``.
    """

    content: tuple[ContentBlock, ...]
    key: str | None = field(default=None, kw_only=True)
    scope: ReminderScope = field(default=ReminderScope.NEXT_REQUEST, kw_only=True)
    placement: ReminderPlacement = field(default=ReminderPlacement.AUTO, kw_only=True)
    priority: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        content = _normalize_content(self.content)
        validate_optional_non_empty_string(self.key, field_name="key")
        try:
            scope = ReminderScope(self.scope)
        except ValueError as error:
            raise ValueError(f"Unsupported reminder scope: {self.scope!r}.") from error
        if scope is ReminderScope.CONVERSATION and self.key is None:
            raise ValueError("conversation-scoped reminders must have a non-empty key.")
        try:
            placement = ReminderPlacement(self.placement)
        except ValueError as error:
            raise ValueError(f"Unsupported reminder placement: {self.placement!r}.") from error
        _validate_priority(self.priority)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "placement", placement)


@dataclass(frozen=True, slots=True)
class ReminderState:
    """Immutable reminders waiting for consumption or persisting while idle.

    Construction normalizes duplicate keyed entries with the same replacement
    semantics as :meth:`upsert`. No method mutates the current instance.
    """

    reminders: tuple[SystemReminder, ...] = ()

    def __post_init__(self) -> None:
        reminders = _normalize_reminders(self.reminders)
        normalized: tuple[SystemReminder, ...] = ()
        for reminder in reminders:
            normalized = _upsert(normalized, reminder)
        object.__setattr__(self, "reminders", normalized)

    def upsert(self, reminder: SystemReminder) -> ReminderState:
        """Return a state with ``reminder`` inserted or replacing its stable key slot."""

        if not isinstance(cast(object, reminder), SystemReminder):
            raise TypeError("reminder must be a SystemReminder.")
        return ReminderState(_upsert(self.reminders, reminder))

    def remove(self, key: str) -> tuple[ReminderState, bool]:
        """Remove one keyed reminder and report whether it existed."""

        _validate_key(key)
        retained = tuple(reminder for reminder in self.reminders if reminder.key != key)
        return ReminderState(retained), len(retained) != len(self.reminders)

    def clear(self, scope: ReminderScope | str | None = None) -> ReminderState:
        """Clear all reminders or only reminders with one scope."""

        if scope is None:
            return ReminderState()
        try:
            resolved_scope = ReminderScope(scope)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported reminder scope: {scope!r}.") from error
        return ReminderState(
            tuple(reminder for reminder in self.reminders if reminder.scope is not resolved_scope)
        )

    def __iter__(self) -> Iterator[SystemReminder]:
        return iter(self.reminders)

    def __len__(self) -> int:
        return len(self.reminders)


def _normalize_content(value: object) -> tuple[ContentBlock, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("content must be an iterable of ContentBlock instances.")
    items = tuple(cast(Iterable[object], value))
    if not items:
        raise ValueError("content must contain at least one ContentBlock.")
    if not all(isinstance(item, ContentBlock) for item in items):
        raise TypeError("content must contain only ContentBlock instances.")
    return cast(tuple[ContentBlock, ...], items)


def _normalize_reminders(value: object) -> tuple[SystemReminder, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("reminders must be an iterable.")
    reminders = tuple(cast(Iterable[object], value))
    if not all(isinstance(reminder, SystemReminder) for reminder in reminders):
        raise TypeError("reminders must contain only SystemReminder instances.")
    return cast(tuple[SystemReminder, ...], reminders)


def _upsert(
    reminders: tuple[SystemReminder, ...],
    reminder: SystemReminder,
) -> tuple[SystemReminder, ...]:
    """Insert a reminder while preserving the stable slot of an existing key."""

    if reminder.key is None:
        return (*reminders, reminder)
    for index, existing in enumerate(reminders):
        if existing.key == reminder.key:
            return (*reminders[:index], reminder, *reminders[index + 1 :])
    return (*reminders, reminder)


def _validate_key(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("key must be a string.")
    if not value:
        raise ValueError("key must not be empty.")


def _validate_priority(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("priority must be an int.")


__all__ = [
    "ReminderPlacement",
    "ReminderScope",
    "ReminderState",
    "SystemReminder",
]
