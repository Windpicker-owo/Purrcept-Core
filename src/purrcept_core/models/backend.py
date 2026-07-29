"""Define the provider-adapter boundary for model generation.

Purrcept Core builds immutable, provider-neutral requests and the runtime
supplies a :class:`ModelBackend` that translates them to a concrete provider.
Backends return the final response and may synchronously emit normalized stream
events during the same call. This module does not own HTTP clients, credentials,
retry policy, or provider-specific configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeAlias, runtime_checkable

from .requests import ModelRequest
from .responses import ModelResponse
from .streaming import ModelStreamEvent

ModelEventSink: TypeAlias = Callable[[ModelStreamEvent], None]
"""Synchronous callback used by a backend to emit one ordered stream event."""


@runtime_checkable
class ModelBackend(Protocol):
    """Extension contract implemented by model provider adapters.

    A runtime or provider package creates the backend and binds it to
    :class:`~purrcept_core.models.model.Model`. The
    :class:`~purrcept_core.models.effects.Generate` effect calls
    :meth:`generate` once per model round. Instances may be reused concurrently
    across conversations, so implementations own client lifetime and
    synchronization.

    Backends must return a validated :class:`ModelResponse`. When ``emit`` is
    supplied, calls to it are synchronous, ordered, and confined to the
    ``generate`` invocation. Provider failures should use the model error
    taxonomy; cancellation and existing exceptions propagate unchanged. The
    core ships no provider implementation, so selection occurs when the
    application constructs a :class:`Model`.
    """

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Return one response and optionally emit ordered stream events.

        ``model`` is the exact provider model name bound by the caller. If a
        ``ModelStreamCompleted`` event is emitted, its response must equal the
        value returned from this method.
        """

        ...


__all__ = ["ModelBackend", "ModelEventSink"]
