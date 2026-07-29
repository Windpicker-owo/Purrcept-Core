"""Verify model generation delegates correctly and enforces the stream protocol.

The suite covers ordered progress forwarding, delayed completion, latched
callback violations, response equality, backend failures, and cancellation.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.driver import AgentDriver
from purrcept_core.events import (
    EffectProgress,
    EffectStarted,
    EffectSucceeded,
    RunStarted,
    RunSucceeded,
)
from purrcept_core.executor import InlineExecutor
from purrcept_core.flow import AgentFlow
from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.effects import Generate
from purrcept_core.models.messages import Message
from purrcept_core.models.model import Model
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse
from purrcept_core.models.streaming import (
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelStreamStarted,
    TextDelta,
)
from purrcept_core.testing import RecordingEventSink


class RecordingBackend:
    def __init__(
        self,
        response: ModelResponse,
        *,
        stream_events: tuple[ModelStreamEvent, ...] = (),
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.stream_events = stream_events
        self.error = error
        self.calls: list[tuple[str, ModelRequest, ModelEventSink | None]] = []

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        self.calls.append((model, request, emit))
        if self.error is not None:
            raise self.error
        if emit is not None:
            for event in self.stream_events:
                emit(event)
        return self.response


class CompleteThenFailBackend:
    def __init__(self, response: ModelResponse, error: Exception) -> None:
        self.response = response
        self.error = error

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model
        if emit is not None:
            emit(ModelStreamCompleted(self.response))
        raise self.error


class SwallowingInvalidEventBackend:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model
        if emit is not None:
            try:
                emit(cast(ModelStreamEvent, object()))
            except TypeError:
                try:
                    emit(TextDelta("still invalid"))
                except TypeError:
                    pass
        return self.response


def _request() -> ModelRequest:
    return ModelRequest((Message.user("Hello"),))


def _response() -> ModelResponse:
    return ModelResponse(Message.assistant("Hello!"))


def _stream_event(delta: str = "chunk") -> ModelStreamEvent:
    return TextDelta(delta)


def _model(backend: object, name: str = "test-model") -> Model:
    return Model(backend, name)  # type: ignore[arg-type]


def _context(progress: list[object] | None = None) -> ExecutionContext[object]:
    return ExecutionContext(
        "run",
        "effect",
        0,
        object(),
        _progress_callback=None if progress is None else progress.append,
    )


def test_generate_is_an_immutable_slotted_effect() -> None:
    effect = Generate(_model(RecordingBackend(_response())), _request())

    assert effect.emit_stream_events is True
    assert not hasattr(effect, "__dict__")
    with pytest.raises(FrozenInstanceError):
        effect.emit_stream_events = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: Generate(object(), _request()),  # type: ignore[arg-type]
            "model must be a Model",
        ),
        (
            lambda: Generate(
                _model(RecordingBackend(_response())),
                object(),  # type: ignore[arg-type]
            ),
            "request must be a ModelRequest",
        ),
        (
            lambda: Generate(
                _model(RecordingBackend(_response())),
                _request(),
                emit_stream_events=1,  # type: ignore[arg-type]
            ),
            "emit_stream_events must be a bool",
        ),
    ],
)
def test_generate_validates_its_public_boundary(factory: object, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()  # type: ignore[operator]


async def test_generate_calls_backend_without_stream_callback_when_disabled() -> None:
    request = _request()
    response = _response()
    backend = RecordingBackend(response, stream_events=(_stream_event(),))
    model = _model(backend)
    progress: list[object] = []

    result = await InlineExecutor().execute(
        Generate(model, request, emit_stream_events=False),
        _context(progress),
    )

    assert result is response
    assert len(backend.calls) == 1
    called_model, called_request, emit = backend.calls[0]
    assert called_model == "test-model"
    assert called_request is request
    assert emit is None
    assert progress == []


async def test_generate_forwards_stream_events_to_progress_in_order() -> None:
    request = _request()
    response = _response()
    stream_events = (
        _stream_event("first"),
        _stream_event("second"),
        _stream_event("third"),
    )
    backend = RecordingBackend(response, stream_events=stream_events)
    model = _model(backend)
    progress: list[object] = []

    result = await InlineExecutor().execute(
        Generate(model, request),
        _context(progress),
    )

    assert result is response
    assert backend.calls[0][0] == "test-model"
    assert backend.calls[0][1] is request
    assert backend.calls[0][2] is not None
    assert progress == list(stream_events)


async def test_generate_accepts_a_well_ordered_complete_stream() -> None:
    response = _response()
    stream_events: tuple[ModelStreamEvent, ...] = (
        ModelStreamStarted(model="example"),
        TextDelta("Hello"),
        ModelStreamCompleted(response),
    )
    backend = RecordingBackend(response, stream_events=stream_events)
    model = _model(backend)
    progress: list[object] = []

    result = await InlineExecutor().execute(
        Generate(model, _request()),
        _context(progress),
    )

    assert result is response
    assert progress == list(stream_events)


@pytest.mark.parametrize(
    "stream_events",
    [
        (TextDelta("first"), ModelStreamStarted()),
        (ModelStreamStarted(), ModelStreamStarted()),
    ],
)
async def test_generate_requires_started_to_be_first_and_unique(
    stream_events: tuple[ModelStreamEvent, ...],
) -> None:
    backend = RecordingBackend(_response(), stream_events=stream_events)
    model = _model(backend)

    with pytest.raises(TypeError, match="must be the first stream event and emitted once"):
        await InlineExecutor().execute(Generate(model, _request()), _context())


async def test_generate_rejects_events_after_completion() -> None:
    response = _response()
    backend = RecordingBackend(
        response,
        stream_events=(ModelStreamCompleted(response), TextDelta("late")),
    )
    model = _model(backend)

    with pytest.raises(TypeError, match="must not emit events after ModelStreamCompleted"):
        await InlineExecutor().execute(Generate(model, _request()), _context())


async def test_generate_requires_completed_response_to_match_return_value() -> None:
    returned = _response()
    completed = ModelResponse(Message.assistant("different"))
    progress: list[object] = []
    backend = RecordingBackend(
        returned,
        stream_events=(ModelStreamCompleted(completed),),
    )
    model = _model(backend)

    with pytest.raises(TypeError, match="must equal the ModelResponse returned"):
        await InlineExecutor().execute(
            Generate(model, _request()),
            _context(progress),
        )

    assert progress == []


async def test_generate_does_not_forward_completion_before_backend_succeeds() -> None:
    response = _response()
    expected = LookupError("failed after completion")
    backend = CompleteThenFailBackend(response, expected)
    model = _model(backend)
    progress: list[object] = []

    with pytest.raises(LookupError) as caught:
        await InlineExecutor().execute(
            Generate(model, _request()),
            _context(progress),
        )

    assert caught.value is expected
    assert progress == []


async def test_generate_preserves_a_protocol_error_swallowed_by_backend() -> None:
    backend = SwallowingInvalidEventBackend(_response())
    model = _model(backend)

    with pytest.raises(TypeError, match="must emit ModelStreamEvent"):
        await InlineExecutor().execute(Generate(model, _request()), _context())


async def test_generate_preserves_backend_exceptions() -> None:
    expected = LookupError("provider failed")
    backend = RecordingBackend(_response(), error=expected)
    model = _model(backend)

    with pytest.raises(LookupError) as caught:
        await InlineExecutor().execute(Generate(model, _request()), _context())

    assert caught.value is expected


async def test_generate_preserves_backend_cancellation() -> None:
    expected = asyncio.CancelledError("cancelled")
    backend = RecordingBackend(_response(), error=expected)
    model = _model(backend)

    with pytest.raises(asyncio.CancelledError) as caught:
        await InlineExecutor().execute(Generate(model, _request()), _context())

    assert caught.value is expected


async def test_generate_rejects_invalid_backend_stream_events() -> None:
    backend = RecordingBackend(
        _response(),
        stream_events=(cast(ModelStreamEvent, object()),),
    )
    model = _model(backend)

    with pytest.raises(TypeError, match="must emit ModelStreamEvent"):
        await InlineExecutor().execute(Generate(model, _request()), _context())


async def test_generate_rejects_invalid_backend_response() -> None:
    backend = RecordingBackend(cast(ModelResponse, object()))
    model = _model(backend)

    with pytest.raises(TypeError, match="must return a ModelResponse"):
        await InlineExecutor().execute(Generate(model, _request()), _context())


async def test_generate_runs_through_executor_and_driver() -> None:
    request = _request()
    response = _response()
    stream_events = (_stream_event("first"), _stream_event("second"))
    backend = RecordingBackend(response, stream_events=stream_events)
    model = _model(backend)
    events = RecordingEventSink()

    def flow() -> AgentFlow[ModelResponse]:
        return (yield Generate(model, request))

    result = await AgentDriver(InlineExecutor(), event_sink=events).run(
        flow(),
        host=object(),
    )

    assert result is response
    assert [type(event) for event in events] == [
        RunStarted,
        EffectStarted,
        EffectProgress,
        EffectProgress,
        EffectSucceeded,
        RunSucceeded,
    ]
    progress_events = events.of_type(EffectProgress)
    assert [event.payload for event in progress_events] == list(stream_events)
    assert [event.step_index for event in progress_events] == [0, 0]
