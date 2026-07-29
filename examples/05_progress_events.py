"""Demonstrate progress events that do not create additional flow steps.

The effect reports synchronous intermediate payloads through its execution
context, while the flow receives only the final result.
"""

import asyncio
from dataclasses import dataclass

from purrcept_core import (
    AgentDriver,
    AgentFlow,
    Effect,
    EffectProgress,
    ExecutionContext,
    InlineExecutor,
    perform,
)
from purrcept_core.testing import RecordingEventSink


@dataclass(frozen=True, slots=True)
class Count(Effect[int]):
    """Count to a limit while reporting each intermediate value."""

    stop: int

    def execute(self, context: ExecutionContext[None]) -> int:
        """Emit progress synchronously and return the terminal count."""

        for value in range(1, self.stop + 1):
            context.progress({"value": value})
        return self.stop


def flow() -> AgentFlow[int]:
    """Request one progress-reporting count."""

    return (yield from perform(Count(3)))


async def main() -> None:
    """Record lifecycle events and verify the progress payload order."""

    events = RecordingEventSink()
    result = await AgentDriver(InlineExecutor(), event_sink=events).run(flow(), host=None)
    progress = [event for event in events.events if isinstance(event, EffectProgress)]
    assert result == 3
    assert [event.payload for event in progress] == [{"value": 1}, {"value": 2}, {"value": 3}]
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
