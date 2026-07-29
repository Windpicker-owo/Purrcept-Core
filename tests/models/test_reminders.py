"""Verify reminder scope, placement, keyed replacement, and immutable state updates."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.content import TextBlock
from purrcept_core.models.reminders import (
    ReminderPlacement,
    ReminderScope,
    ReminderState,
    SystemReminder,
)


def test_system_reminder_normalizes_scope_placement_and_content() -> None:
    blocks = [TextBlock("cite sources")]

    reminder = SystemReminder(
        blocks,  # type: ignore[arg-type]
        key="citations",
        scope="conversation",  # type: ignore[arg-type]
        placement="tail",  # type: ignore[arg-type]
        priority=-10,
    )
    blocks.append(TextBlock("changed"))

    assert reminder.content == (TextBlock("cite sources"),)
    assert reminder.key == "citations"
    assert reminder.scope is ReminderScope.CONVERSATION
    assert reminder.placement is ReminderPlacement.TAIL
    assert reminder.priority == -10
    with pytest.raises(FrozenInstanceError):
        reminder.priority = 1  # type: ignore[misc]


def test_system_reminder_uses_explicit_lifecycle_defaults() -> None:
    reminder = SystemReminder((TextBlock("one request"),))

    assert reminder.key is None
    assert reminder.scope is ReminderScope.NEXT_REQUEST
    assert reminder.placement is ReminderPlacement.AUTO
    assert reminder.priority == 0
    assert {scope.value for scope in ReminderScope} == {
        "next_request",
        "turn",
        "conversation",
    }


def test_reminder_state_is_immutable_and_replaces_keys_in_their_stable_slots() -> None:
    first = SystemReminder((TextBlock("first"),), key="same")
    unkeyed = SystemReminder((TextBlock("unkeyed"),))
    replacement = SystemReminder(
        (TextBlock("replacement"),),
        key="same",
        scope=ReminderScope.TURN,
    )
    source = [first, unkeyed, replacement]

    state = ReminderState(source)  # type: ignore[arg-type]
    source.clear()

    assert state.reminders == (replacement, unkeyed)
    assert tuple(state) == state.reminders
    assert len(state) == 2
    assert not hasattr(state, "__dict__")
    with pytest.raises(FrozenInstanceError):
        state.reminders = ()  # type: ignore[misc]


def test_reminder_state_upsert_remove_and_clear_are_pure_transitions() -> None:
    next_request = SystemReminder((TextBlock("next"),), key="next")
    turn = SystemReminder(
        (TextBlock("turn"),),
        key="turn",
        scope=ReminderScope.TURN,
    )
    conversation = SystemReminder(
        (TextBlock("conversation"),),
        key="conversation",
        scope=ReminderScope.CONVERSATION,
    )
    state = ReminderState((next_request, turn)).upsert(conversation)

    retained, removed = state.remove("turn")
    missing, missing_removed = retained.remove("missing")

    assert state.reminders == (next_request, turn, conversation)
    assert retained.reminders == (next_request, conversation)
    assert removed is True
    assert missing is not retained
    assert missing.reminders == retained.reminders
    assert missing_removed is False
    assert state.clear("turn").reminders == (next_request, conversation)
    assert state.clear(ReminderScope.NEXT_REQUEST).reminders == (turn, conversation)
    assert state.clear().reminders == ()


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ReminderState(1),  # type: ignore[arg-type]
            TypeError,
            "reminders must be an iterable",
        ),
        (
            lambda: ReminderState((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only SystemReminder",
        ),
        (
            lambda: ReminderState().upsert(object()),  # type: ignore[arg-type]
            TypeError,
            "reminder must be a SystemReminder",
        ),
        (
            lambda: ReminderState().remove(1),  # type: ignore[arg-type]
            TypeError,
            "key must be a string",
        ),
        (
            lambda: ReminderState().remove(""),
            ValueError,
            "key must not be empty",
        ),
        (
            lambda: ReminderState().clear("forever"),
            ValueError,
            "Unsupported reminder scope",
        ),
        (
            lambda: ReminderState().clear(object()),  # type: ignore[arg-type]
            ValueError,
            "Unsupported reminder scope",
        ),
    ],
)
def test_reminder_state_validates_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
    assert {placement.value for placement in ReminderPlacement} == {
        "auto",
        "instructions",
        "tail",
    }


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: SystemReminder(1),  # type: ignore[arg-type]
            TypeError,
            "content must be an iterable",
        ),
        (lambda: SystemReminder(()), ValueError, "at least one ContentBlock"),
        (
            lambda: SystemReminder((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only ContentBlock",
        ),
        (
            lambda: SystemReminder((TextBlock("x"),), key=1),  # type: ignore[arg-type]
            TypeError,
            "key must be a string",
        ),
        (
            lambda: SystemReminder((TextBlock("x"),), key=""),
            ValueError,
            "key must not be empty",
        ),
        (
            lambda: SystemReminder(
                (TextBlock("x"),),
                scope=ReminderScope.CONVERSATION,
            ),
            ValueError,
            "conversation-scoped reminders",
        ),
        (
            lambda: SystemReminder(
                (TextBlock("x"),),
                scope="forever",  # type: ignore[arg-type]
            ),
            ValueError,
            "Unsupported reminder scope",
        ),
        (
            lambda: SystemReminder(
                (TextBlock("x"),),
                placement="middle",  # type: ignore[arg-type]
            ),
            ValueError,
            "Unsupported reminder placement",
        ),
        (
            lambda: SystemReminder(
                (TextBlock("x"),),
                priority=True,  # type: ignore[arg-type]
            ),
            TypeError,
            "priority must be an int",
        ),
        (
            lambda: SystemReminder(
                (TextBlock("x"),),
                priority=1.5,  # type: ignore[arg-type]
            ),
            TypeError,
            "priority must be an int",
        ),
    ],
)
def test_system_reminder_validates_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
