"""Run an offline model conversation with one automatic Python function tool call.

The fake backend returns a tool request on the first round and a final assistant
message after receiving the tool result. The example shows that model and tool
effects share the same driver and lifecycle event stream.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from purrcept_core import AgentDriver, AgentFlow, EffectStarted, InlineExecutor
from purrcept_core.models import (
    Conversation,
    FinishReason,
    Generate,
    InvokeTool,
    Message,
    MessageRole,
    Model,
    ModelEventSink,
    ModelRequest,
    ModelResponse,
    ModelTurnResult,
    ToolCallBlock,
)
from purrcept_core.testing import RecordingEventSink


class OfflineBackend:
    """Return a fixed tool call followed by a fixed final response."""

    __slots__ = ("requested_models",)

    def __init__(self) -> None:
        self.requested_models: list[str] = []

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Return a deterministic tool request followed by a final response."""

        del emit
        self.requested_models.append(model)
        if not any(message.role is MessageRole.TOOL for message in request.messages):
            return ModelResponse(
                Message(
                    MessageRole.ASSISTANT,
                    (
                        ToolCallBlock(
                            "add-1",
                            "add",
                            arguments={"left": 2, "right": 3},
                        ),
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALL,
                model=model,
            )

        tool_message = request.messages[-1]
        return ModelResponse(
            Message.assistant(f"The offline result is {tool_message.text}."),
            model=model,
        )


@dataclass(frozen=True, slots=True)
class RuntimeHost:
    """Example application host passed independently of model configuration."""

    environment: str


def add(left: int, right: int) -> int:
    """Add two integers.

    Args:
        left: The left operand.
        right: The right operand.
    """

    return left + right


def flow(conversation: Conversation) -> AgentFlow[ModelTurnResult]:
    """Ask one question through the conversation's transactional turn flow."""

    return (yield from conversation.ask("What is two plus three?"))


async def main() -> None:
    """Run the offline tool loop and verify its effect sequence."""

    backend = OfflineBackend()
    conversation = Model(backend, "offline-demo").conversation(tools=(add,))
    events = RecordingEventSink()
    result = await AgentDriver(
        InlineExecutor(),
        event_sink=events,
    ).run(
        flow(conversation),
        host=RuntimeHost("offline"),
    )

    effect_types = [type(event.effect) for event in events.of_type(EffectStarted)]
    assert effect_types == [Generate, InvokeTool, Generate]
    assert backend.requested_models == ["offline-demo", "offline-demo"]
    assert result.model_rounds == 2
    assert result.text == "The offline result is 5."
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
