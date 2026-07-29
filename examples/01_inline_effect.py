"""Demonstrate a self-executing effect interpreted by ``InlineExecutor``.

The application owns the asyncio loop and driver; the synchronous agent flow
remains ordinary generator code.
"""

import asyncio
from dataclasses import dataclass

from purrcept_core import AgentDriver, AgentFlow, Effect, ExecutionContext, InlineExecutor, perform


@dataclass(frozen=True, slots=True)
class Add(Effect[int]):
    """Request integer addition and carry its result type to the flow."""

    left: int
    right: int

    def execute(self, context: ExecutionContext[None]) -> int:
        """Compute the effect result inline without external runtime state."""

        return self.left + self.right


def calculator() -> AgentFlow[int]:
    """Show direct ``yield`` and typed ``perform`` in the same flow."""

    # Direct yield is valid; perform() is preferred when exact result inference matters.
    first = yield Add(1, 2)
    second = yield from perform(Add(first, 4))
    return second


async def main() -> None:
    """Run the example with an application-owned asyncio entry point."""

    result = await AgentDriver(InlineExecutor()).run(calculator(), host=None)
    assert result == 7
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
