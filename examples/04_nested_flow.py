"""Demonstrate nested generator flows within one continuous agent run.

``yield from`` composes child flows without creating another driver, run ID, or
event sequence.
"""

import asyncio
from dataclasses import dataclass

from purrcept_core import AgentDriver, AgentFlow, Effect, ExecutionContext, InlineExecutor, perform


@dataclass(frozen=True, slots=True)
class Multiply(Effect[int]):
    """Request integer multiplication."""

    value: int
    by: int

    def execute(self, context: ExecutionContext[None]) -> int:
        """Compute the multiplication inline."""

        return self.value * self.by


def child(value: int) -> AgentFlow[int]:
    """Double one value through a typed effect."""

    return (yield from perform(Multiply(value, 2)))


def parent() -> AgentFlow[int]:
    """Compose two child flows under the same run."""

    first = yield from child(3)
    return (yield from child(first))


async def main() -> None:
    """Drive the nested flow to its final value."""

    result = await AgentDriver(InlineExecutor()).run(parent(), host=None)
    assert result == 12
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
