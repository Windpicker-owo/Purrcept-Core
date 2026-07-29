"""Verify prompt tracing is synchronous, metadata-only, ordered, and failure-visible."""

from __future__ import annotations

from typing import cast

import pytest

from purrcept_core.models import (
    ConversationState,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    PreparedContext,
    PromptCompiled,
    PromptDraft,
    PromptTraceEvent,
    ReminderConsumed,
    ReminderResolutionContext,
    ReminderSourceFailed,
    ReminderSourceResolved,
    ReminderState,
    SystemReminder,
    TextBlock,
)
from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.prompt.compiler import TurnPromptSession
from purrcept_core.models.prompt.tracing import (
    emit_prompt_trace,
    validate_prompt_trace_sink,
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


class Source:
    source_id = "workspace"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def resolve(self, context: ReminderResolutionContext):
        del context
        if self.error is not None:
            raise self.error
        return (
            SystemReminder(
                (TextBlock("dynamic-sensitive-text"),),
                key="workspace:state",
            ),
        )


def _context() -> ReminderResolutionContext:
    prepared = PreparedContext((Message.user("current"),))
    return ReminderResolutionContext(
        Model(NeverBackend(), "trace-model"),
        ConversationState(),
        prepared,
        1,
        1,
    )


def test_trace_reports_compilation_resolution_and_consumption_without_raw_text() -> None:
    events: list[PromptTraceEvent] = []
    next_request = SystemReminder((TextBlock("stored-sensitive-text"),), key="next")
    session = TurnPromptSession.begin(
        ReminderState((next_request,)),
        reminder_sources=(Source(),),
        trace_sink=events.append,
    )

    session.compile(
        PromptDraft(PreparedContext((Message.user("current"),))),
        context=_context(),
        model_round=1,
    )

    assert [type(event) for event in events] == [
        ReminderSourceResolved,
        PromptCompiled,
        ReminderConsumed,
    ]
    resolved = cast(ReminderSourceResolved, events[0])
    compiled = cast(PromptCompiled, events[1])
    consumed = cast(ReminderConsumed, events[2])
    assert resolved.source_id == "workspace"
    assert resolved.reminder_keys == ("workspace:state",)
    assert compiled.reminder_keys == ("next", "workspace:state")
    assert compiled.dynamic_source_ids == ("workspace",)
    assert all(len(value) == 64 for value in compiled.content_hashes)
    assert compiled.estimated_tokens is None
    assert compiled.trimmed_turns == 0
    assert consumed.reminder_keys == ("next",)
    assert "sensitive" not in repr(events)


def test_source_failures_emit_metadata_and_preserve_the_original_error() -> None:
    expected = LookupError("secret failure detail")
    events: list[PromptTraceEvent] = []
    session = TurnPromptSession.begin(
        ReminderState(),
        reminder_sources=(Source(error=expected),),
        trace_sink=events.append,
    )

    with pytest.raises(LookupError) as caught:
        session.compile(
            PromptDraft(PreparedContext((Message.user("current"),))),
            context=_context(),
            model_round=1,
        )

    assert caught.value is expected
    assert len(events) == 1
    failed = cast(ReminderSourceFailed, events[0])
    assert failed.source_id == "workspace"
    assert failed.error_type == "builtins.LookupError"
    assert "secret failure detail" not in repr(failed)


def test_trace_sink_must_be_synchronous_and_return_none() -> None:
    validate_prompt_trace_sink(None)
    validate_prompt_trace_sink(lambda event: None)
    with pytest.raises(TypeError, match="callable or None"):
        validate_prompt_trace_sink(object())

    event = ReminderConsumed(1, ())

    async def async_sink(event: PromptTraceEvent) -> None:
        del event

    with pytest.raises(TypeError, match="must be synchronous"):
        emit_prompt_trace(async_sink, event)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return None"):
        emit_prompt_trace(
            cast(object, lambda event: "value"),  # type: ignore[arg-type]
            event,
        )
    emit_prompt_trace(None, event)


def test_trace_sink_rejects_a_non_coroutine_awaitable() -> None:
    class AwaitableResult:
        def __await__(self):
            yield
            return None

    with pytest.raises(TypeError, match="must be synchronous"):
        emit_prompt_trace(
            cast(object, lambda event: AwaitableResult()),  # type: ignore[arg-type]
            ReminderConsumed(1, ()),
        )
