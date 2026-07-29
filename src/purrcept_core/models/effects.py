"""Bridge a provider-neutral model request into the core effect runtime.

``Generate`` is the low-level effect created by :meth:`Model.generate`. Its
inline executor delegates to the bound backend, validates the final response,
and optionally forwards normalized stream events through effect progress.
Conversation state and tool-loop orchestration remain in the neighboring model
loop.

Streaming is validated as one synchronous backend protocol: a start event may
appear only first, completion is withheld until the backend successfully
returns the same response, and the first protocol violation remains primary
even if a backend catches the callback exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context import ExecutionContext
from ..effect import Effect
from .backend import ModelEventSink
from .model import Model
from .requests import ModelRequest
from .responses import ModelResponse
from .streaming import ModelStreamCompleted, ModelStreamEvent, ModelStreamStarted


@dataclass(frozen=True, slots=True)
class Generate(Effect[ModelResponse]):
    """Immutable effect requesting one response from a model's bound backend.

    ``InlineExecutor`` calls :meth:`execute`; runtimes may also dispatch the
    effect explicitly. Backend exceptions and cancellation propagate unchanged.
    When stream forwarding is enabled, validated intermediate events are
    reported through the surrounding execution context.
    """

    model: Model
    request: ModelRequest
    emit_stream_events: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the effect boundary before it reaches an executor."""

        _validate_generate_arguments(self.model, self.request, self.emit_stream_events)

    async def execute(self, context: ExecutionContext[object]) -> ModelResponse:
        """Delegate generation and forward optional stream events as progress."""

        forwarder = _StreamForwarder(context) if self.emit_stream_events else None
        emit: ModelEventSink | None = forwarder
        response = await self.model.backend.generate(
            self.request,
            model=self.model.name,
            emit=emit,
        )
        response = _require_model_response(response)
        if forwarder is not None:
            forwarder.validate_response(response)
        return response


def _validate_generate_arguments(
    model: object,
    request: object,
    emit_stream_events: object,
) -> None:
    if not isinstance(model, Model):
        raise TypeError("model must be a Model.")
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest.")
    if not isinstance(emit_stream_events, bool):
        raise TypeError("emit_stream_events must be a bool.")


class _StreamForwarder:
    """Enforce the backend stream protocol before exposing events as progress.

    The forwarder belongs to exactly one ``Generate.execute`` call. It latches
    the first protocol error so a backend cannot suppress its callback failure,
    and delays ``ModelStreamCompleted`` until the returned response is known to
    match.
    """

    __slots__ = ("_completed_event", "_context", "_protocol_error", "_seen_event")

    def __init__(self, context: ExecutionContext[object]) -> None:
        self._context = context
        self._seen_event = False
        self._completed_event: ModelStreamCompleted | None = None
        self._protocol_error: TypeError | None = None

    def __call__(self, event: ModelStreamEvent) -> None:
        """Validate and forward one backend event in emission order."""

        protocol_error = self._protocol_error
        if protocol_error is not None:
            raise protocol_error
        try:
            event = _require_model_stream_event(event)
            if self._completed_event is not None:
                raise TypeError("model backend must not emit events after ModelStreamCompleted.")
            if isinstance(event, ModelStreamStarted) and self._seen_event:
                raise TypeError(
                    "ModelStreamStarted must be the first stream event and emitted once."
                )
            self._seen_event = True
            if isinstance(event, ModelStreamCompleted):
                # Ordering: completion is held back until `generate` returns.
                # Forwarding it now could report success before a later backend
                # failure or before response equality can be checked.
                self._completed_event = event
                return
        except TypeError as error:
            self._protocol_error = error
            raise
        self._context.progress(event)

    def validate_response(self, response: ModelResponse) -> None:
        """Validate the final response and release a matching completion event."""

        protocol_error = self._protocol_error
        if protocol_error is not None:
            raise protocol_error
        completed_event = self._completed_event
        if completed_event is None:
            return
        if completed_event.response != response:
            protocol_error = TypeError(
                "ModelStreamCompleted.response must equal the ModelResponse returned "
                "by the backend."
            )
            self._protocol_error = protocol_error
            raise protocol_error
        self._context.progress(completed_event)


def _require_model_stream_event(value: object) -> ModelStreamEvent:
    if not isinstance(value, ModelStreamEvent):
        raise TypeError("model backend must emit ModelStreamEvent instances.")
    return value


def _require_model_response(value: object) -> ModelResponse:
    if not isinstance(value, ModelResponse):
        raise TypeError("model backend must return a ModelResponse.")
    return value


__all__ = ["Generate"]
