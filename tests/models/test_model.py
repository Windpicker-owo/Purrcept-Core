"""Verify model bindings are immutable and create generation and conversation APIs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor, perform
from purrcept_core.models.backend import ModelBackend, ModelEventSink
from purrcept_core.models.caching import CacheMode
from purrcept_core.models.conversation import Conversation
from purrcept_core.models.effects import Generate
from purrcept_core.models.instructions import SystemInstruction
from purrcept_core.models.messages import Message
from purrcept_core.models.model import Model
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ModelRequest]] = []

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del emit
        self.calls.append((model, request))
        return ModelResponse(Message.assistant(model))


def _request() -> ModelRequest:
    return ModelRequest((Message.user("hello"),))


def test_model_binds_one_name_to_a_structural_backend() -> None:
    backend = RecordingBackend()
    model = Model(backend, "smart-model")

    assert isinstance(backend, ModelBackend)
    assert model.backend is backend
    assert model.name == "smart-model"
    assert "RecordingBackend" not in repr(model)
    assert not hasattr(model, "__dict__")
    with pytest.raises(FrozenInstanceError):
        model.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: Model(object(), "model"),  # type: ignore[arg-type]
            TypeError,
            "backend must implement ModelBackend",
        ),
        (
            lambda: Model(RecordingBackend(), 1),  # type: ignore[arg-type]
            TypeError,
            "name must be a string",
        ),
        (
            lambda: Model(RecordingBackend(), ""),
            ValueError,
            "name must not be empty",
        ),
    ],
)
def test_model_validates_its_public_boundary(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_model_generate_binds_the_model_and_request_to_an_effect() -> None:
    model = Model(RecordingBackend(), "model")
    request = _request()

    effect = model.generate(request)

    assert effect == Generate(model, request)
    assert effect.model is model
    assert effect.request is request


def test_model_conversation_is_the_high_level_bound_entrypoint() -> None:
    model = Model(RecordingBackend(), "model")

    conversation = model.conversation(
        instructions="Answer precisely.",
        cache="prefer",
        max_model_rounds=3,
    )

    assert isinstance(conversation, Conversation)
    assert conversation.model is model
    assert conversation.instructions == (SystemInstruction.from_text("Answer precisely."),)
    assert conversation.cache.mode is CacheMode.PREFER
    assert conversation.max_model_rounds == 3


async def test_multiple_models_share_a_backend_without_using_the_runtime_host() -> None:
    backend = RecordingBackend()
    fast = Model(backend, "fast-model")
    smart = Model(backend, "smart-model")
    request = _request()

    def flow() -> AgentFlow[tuple[str, str]]:
        fast_response = yield from perform(fast.generate(request))
        smart_response = yield from perform(smart.generate(request))
        return fast_response.text, smart_response.text

    result = await AgentDriver(InlineExecutor()).run(flow(), host=object())

    assert result == ("fast-model", "smart-model")
    assert backend.calls == [
        ("fast-model", request),
        ("smart-model", request),
    ]
