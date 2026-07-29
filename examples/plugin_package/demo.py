"""Run the installable demo provider plugin through the core runtime.

The example composes a plugin-owned backend with an application-owned tool,
host, driver, and event recorder. Assertions cover tool-loop order and stream
progress without giving the plugin access to runtime host state.
"""

import asyncio
from dataclasses import dataclass

from purrcept_demo_plugin import DemoModelBackend

from purrcept_core import (
    AgentDriver,
    AgentFlow,
    EffectProgress,
    EffectStarted,
    InlineExecutor,
)
from purrcept_core.models import (
    Conversation,
    Generate,
    InvokeTool,
    Model,
    ModelTurnResult,
    TextDelta,
)
from purrcept_core.testing import RecordingEventSink


@dataclass(frozen=True, slots=True)
class DemoRuntimeHost:
    """Application host value that remains outside the provider adapter."""

    run_label: str


def normalize_name(name: str) -> str:
    """Normalize a name for display.

    Args:
        name: The raw name supplied by the user.
    """

    return name.strip().title()


def flow(conversation: Conversation) -> AgentFlow[ModelTurnResult]:
    """Ask the demo backend to normalize and greet one input."""

    return (yield from conversation.ask("purrcept"))


async def main() -> None:
    """Run the plugin conversation and verify effects and streamed text."""

    backend = DemoModelBackend()
    model = Model(backend, "purrcept-demo")
    conversation = model.conversation(
        instructions="Use normalize_name before greeting the user.",
        tools=(normalize_name,),
    )
    events = RecordingEventSink()
    host = DemoRuntimeHost("offline-plugin-demo")
    result = await AgentDriver(
        InlineExecutor(),
        event_sink=events,
    ).run(
        flow(conversation),
        host=host,
    )
    assert result.text == "Hello, Purrcept!"
    assert result.model_rounds == 2
    assert not hasattr(host, "model")
    effect_types = [type(event.effect) for event in events.of_type(EffectStarted)]
    assert effect_types == [Generate, InvokeTool, Generate]
    deltas = [
        event.payload.delta
        for event in events.of_type(EffectProgress)
        if isinstance(event.payload, TextDelta)
    ]
    assert deltas == ["Hello, Purrcept!"]
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
