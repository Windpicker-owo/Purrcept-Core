"""Verify streaming event snapshots, extension markers, and field validation."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.messages import Message
from purrcept_core.models.responses import ModelResponse, TokenUsage
from purrcept_core.models.streaming import (
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelStreamStarted,
    TextDelta,
    ToolCallDelta,
    UsageUpdate,
)


def test_stream_events_are_typed_immutable_snapshots() -> None:
    metadata: dict[str, object] = {"headers": {"request-id": "one"}}
    started = ModelStreamStarted(
        model="example-model",
        response_id="response-1",
        provider_metadata=metadata,  # type: ignore[arg-type]
    )
    text = TextDelta("hello")
    tool = ToolCallDelta(0, '{"query":', tool_call_id="call-1", name="lookup")
    usage = UsageUpdate(TokenUsage(2, 1))
    response = ModelResponse(Message.assistant("hello"))
    completed = ModelStreamCompleted(response)
    metadata.clear()

    events: tuple[ModelStreamEvent, ...] = (started, text, tool, usage, completed)

    assert len(events) == 5
    assert started.provider_metadata == {"headers": {"request-id": "one"}}
    assert text.delta == "hello"
    assert text.index == 0
    assert tool.arguments_delta == '{"query":'
    assert usage.usage.total_tokens == 3
    assert completed.response is response
    with pytest.raises(FrozenInstanceError):
        text.delta = "changed"  # type: ignore[misc]


def test_stream_events_allow_minimal_provider_fragments() -> None:
    started = ModelStreamStarted()
    text = TextDelta("hello", index=2)
    tool = ToolCallDelta(1, "")

    assert started.model is None
    assert started.response_id is None
    assert started.provider_metadata == {}
    assert text.index == 2
    assert tool.tool_call_id is None
    assert tool.name is None


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ModelStreamStarted(model=""),
            ValueError,
            "model must not be empty",
        ),
        (
            lambda: ModelStreamStarted(response_id=1),  # type: ignore[arg-type]
            TypeError,
            "response_id must be a string",
        ),
        (
            lambda: TextDelta(1),  # type: ignore[arg-type]
            TypeError,
            "delta must be a string",
        ),
        (
            lambda: TextDelta("x", index=True),  # type: ignore[arg-type]
            TypeError,
            "index must be an int",
        ),
        (lambda: TextDelta("x", index=-1), ValueError, "greater than or equal"),
        (
            lambda: ToolCallDelta(True, ""),  # type: ignore[arg-type]
            TypeError,
            "index must be an int",
        ),
        (lambda: ToolCallDelta(-1, ""), ValueError, "greater than or equal"),
        (
            lambda: ToolCallDelta(0, 1),  # type: ignore[arg-type]
            TypeError,
            "arguments_delta must be a string",
        ),
        (
            lambda: ToolCallDelta(0, "", tool_call_id=""),
            ValueError,
            "tool_call_id must not be empty",
        ),
        (
            lambda: ToolCallDelta(0, "", name=1),  # type: ignore[arg-type]
            TypeError,
            "name must be a string",
        ),
        (
            lambda: UsageUpdate(object()),  # type: ignore[arg-type]
            TypeError,
            "usage must be a TokenUsage",
        ),
        (
            lambda: ModelStreamCompleted(object()),  # type: ignore[arg-type]
            TypeError,
            "response must be a ModelResponse",
        ),
    ],
)
def test_stream_events_validate_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
