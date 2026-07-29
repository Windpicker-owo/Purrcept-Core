"""Verify dynamic reminder-source registration, resolution, and request-local state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.models import (
    ConversationState,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    PreparedContext,
    ReminderResolutionContext,
    ReminderSource,
    SystemReminder,
    TextBlock,
)
from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.reminder_sources import (
    normalize_reminder_sources,
    resolve_reminder_source,
)


class NeverBackend:
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model, emit
        raise AssertionError("backend must not run")


class StaticSource:
    def __init__(
        self,
        source_id: str,
        reminders: object,
    ) -> None:
        self.source_id = source_id
        self.reminders = reminders
        self.contexts: list[ReminderResolutionContext] = []

    def resolve(
        self,
        context: ReminderResolutionContext,
    ):
        self.contexts.append(context)
        return self.reminders


def _context() -> ReminderResolutionContext:
    prepared = PreparedContext((Message.user("current"),))
    return ReminderResolutionContext(
        Model(NeverBackend(), "source-model"),
        ConversationState(),
        prepared,
        1,
        1,
    )


def _reminder(text: str = "state") -> SystemReminder:
    return SystemReminder((TextBlock(text),), key=f"dynamic:{text}")


def test_resolution_context_is_an_immutable_explicit_snapshot() -> None:
    context = _context()

    assert context.model.name == "source-model"
    assert context.conversation == ConversationState()
    assert context.prepared_context.messages == (Message.user("current"),)
    assert context.turn_index == 1
    assert context.model_round == 1
    assert not hasattr(context, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.model_round = 2  # type: ignore[misc]


def test_sources_are_local_duplicate_free_and_resolve_in_registration_order() -> None:
    first = StaticSource("first", [_reminder("first")])
    second = StaticSource("second", (_reminder("second"),))
    source_list = [first, second]

    sources = normalize_reminder_sources(source_list)
    source_list.clear()

    assert sources == (first, second)
    assert isinstance(first, ReminderSource)
    assert resolve_reminder_source(first, _context()) == (_reminder("first"),)
    assert len(first.contexts) == 1
    assert first.contexts[0].prepared_context.messages == (Message.user("current"),)
    assert first.contexts[0].model.name == "source-model"


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ReminderResolutionContext(
                cast(Model, object()),
                ConversationState(),
                PreparedContext((Message.user("x"),)),
                1,
                1,
            ),
            TypeError,
            "model must be a Model",
        ),
        (
            lambda: ReminderResolutionContext(
                Model(NeverBackend(), "model"),
                cast(ConversationState, object()),
                PreparedContext((Message.user("x"),)),
                1,
                1,
            ),
            TypeError,
            "conversation must be a ConversationState",
        ),
        (
            lambda: ReminderResolutionContext(
                Model(NeverBackend(), "model"),
                ConversationState(),
                cast(PreparedContext, object()),
                1,
                1,
            ),
            TypeError,
            "prepared_context must be a PreparedContext",
        ),
        (
            lambda: ReminderResolutionContext(
                Model(NeverBackend(), "model"),
                ConversationState(),
                PreparedContext((Message.user("x"),)),
                True,
                1,
            ),
            TypeError,
            "turn_index must be an int",
        ),
        (
            lambda: ReminderResolutionContext(
                Model(NeverBackend(), "model"),
                ConversationState(),
                PreparedContext((Message.user("x"),)),
                0,
                1,
            ),
            ValueError,
            "turn_index must be greater than zero",
        ),
        (
            lambda: ReminderResolutionContext(
                Model(NeverBackend(), "model"),
                ConversationState(),
                PreparedContext((Message.user("x"),)),
                1,
                0,
            ),
            ValueError,
            "model_round must be greater than zero",
        ),
        (
            lambda: normalize_reminder_sources("source"),
            TypeError,
            "must be an iterable",
        ),
        (
            lambda: normalize_reminder_sources((object(),)),
            TypeError,
            "only ReminderSource",
        ),
        (
            lambda: normalize_reminder_sources((StaticSource("", ()),)),
            ValueError,
            "source_id must not be empty",
        ),
        (
            lambda: normalize_reminder_sources(
                (StaticSource("same", ()), StaticSource("same", ()))
            ),
            ValueError,
            "registered more than once",
        ),
        (
            lambda: resolve_reminder_source(StaticSource("source", 1), _context()),
            TypeError,
            "must return an iterable",
        ),
        (
            lambda: resolve_reminder_source(
                StaticSource("source", "not-reminders"),
                _context(),
            ),
            TypeError,
            "must return an iterable",
        ),
        (
            lambda: resolve_reminder_source(
                StaticSource("source", (object(),)),
                _context(),
            ),
            TypeError,
            "only SystemReminder",
        ),
    ],
)
def test_reminder_sources_validate_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_async_source_result_and_awaitable_items_are_closed_and_rejected() -> None:
    class AsyncSource:
        source_id = "async"

        async def resolve(
            self,
            context: ReminderResolutionContext,
        ) -> tuple[SystemReminder, ...]:
            del context
            return ()

    async def one_reminder() -> SystemReminder:
        return _reminder()

    async_item = one_reminder()
    item_source = StaticSource("item", (async_item,))

    with pytest.raises(TypeError, match="must be synchronous"):
        resolve_reminder_source(AsyncSource(), _context())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must not yield awaitable"):
        resolve_reminder_source(item_source, _context())

    assert async_item.cr_frame is None


def test_source_exceptions_propagate_unchanged() -> None:
    expected = LookupError("workspace unavailable")

    class FailingSource:
        source_id = "failing"

        def resolve(self, context: ReminderResolutionContext):
            del context
            raise expected

    with pytest.raises(LookupError) as caught:
        resolve_reminder_source(FailingSource(), _context())

    assert caught.value is expected
