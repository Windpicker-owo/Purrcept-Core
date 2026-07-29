"""Verify the model backend protocol supports structural runtime checks."""

from __future__ import annotations

from purrcept_core.models.backend import ModelBackend, ModelEventSink
from purrcept_core.models.messages import Message
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse


class CompatibleBackend:
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model, emit
        return ModelResponse(Message.assistant("done"))


def test_model_backend_supports_runtime_structural_checks() -> None:
    assert isinstance(CompatibleBackend(), ModelBackend)
    assert not isinstance(object(), ModelBackend)
