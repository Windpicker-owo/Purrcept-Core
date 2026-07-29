"""Provide a deterministic provider plugin and two example extension effects.

``DemoModelBackend`` translates provider-neutral requests without depending on
driver host state. ``Echo`` demonstrates an inline effect, while
``RenderTemplate`` demonstrates an effect handled by external dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from purrcept_core import Effect, ExecutionContext
from purrcept_core.models import (
    FinishReason,
    Message,
    MessageRole,
    ModelBackend,
    ModelEventSink,
    ModelRequest,
    ModelRequestError,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamStarted,
    TextDelta,
    ToolCallBlock,
)


@dataclass(frozen=True, slots=True)
class Echo(Effect[str]):
    """Return one text value through inline effect execution."""

    text: str

    def execute(self, context: ExecutionContext[object]) -> str:
        """Return the stored text without consulting runtime host state."""

        return self.text


@dataclass(frozen=True, slots=True)
class DemoModelBackend(ModelBackend):
    """Deterministic offline provider adapter used by the extension example.

    A runtime constructs and binds the backend to a ``Model``. Each call either
    requests the registered normalization tool or converts its result to a final
    greeting. The instance owns no external resources and is reusable.
    """

    default_prefix: str = "Hello, "

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Translate one request and emit a complete normalized stream when asked."""

        prefix = request.provider_options.get(
            "purrcept_demo_plugin.prefix",
            self.default_prefix,
        )
        if not isinstance(prefix, str):
            raise ModelRequestError(
                "purrcept_demo_plugin.prefix must be a string",
            )

        final_message = request.messages[-1] if request.messages else None
        tool_message = (
            final_message
            if final_message is not None and final_message.role is MessageRole.TOOL
            else None
        )
        if tool_message is None:
            prompt = _latest_user_text(request)
            tool_name = "normalize_name"
            if not any(spec.name == tool_name for spec in request.tools):
                raise ModelRequestError(
                    "the demo backend requires a normalize_name tool",
                    provider="purrcept-demo-plugin",
                    model=model,
                )
            response = ModelResponse(
                Message(
                    MessageRole.ASSISTANT,
                    (
                        ToolCallBlock(
                            "normalize-name-1",
                            tool_name,
                            arguments={"name": prompt},
                        ),
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALL,
                model=model,
                response_id="purrcept-demo-tool-call",
            )
        else:
            response = ModelResponse(
                Message.assistant(f"{prefix}{tool_message.text}!"),
                model=model,
                response_id="purrcept-demo-final",
            )

        if emit is not None:
            emit(
                ModelStreamStarted(
                    model=response.model,
                    response_id=response.response_id,
                )
            )
            if response.text:
                emit(TextDelta(response.text))
            emit(ModelStreamCompleted(response))
        return response


def _latest_user_text(request: ModelRequest) -> str:
    """Return the newest non-empty user text or a normalized request error."""

    for message in reversed(request.messages):
        if message.role is MessageRole.USER and message.text:
            return message.text
    raise ModelRequestError(
        "the demo backend requires a non-empty user message",
        provider="purrcept-demo-plugin",
    )


@dataclass(frozen=True, slots=True)
class RenderTemplate(Effect[str]):
    """Request formatting with an explicit template and value mapping."""

    template: str
    values: Mapping[str, object]


def render_template(
    effect: RenderTemplate,
    context: ExecutionContext[object],
) -> str:
    """Interpret ``RenderTemplate`` as an external dispatch handler."""

    return effect.template.format_map(effect.values)


__all__ = [
    "DemoModelBackend",
    "Echo",
    "RenderTemplate",
    "render_template",
]
