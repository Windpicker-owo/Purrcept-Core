"""Demonstrate a declarative effect interpreted by explicit type dispatch.

The effect contains only request data; application wiring selects its handler
when constructing ``DispatchExecutor``.
"""

import asyncio
from dataclasses import dataclass

from purrcept_core import (
    AgentDriver,
    AgentFlow,
    DispatchExecutor,
    Effect,
    ExecutionContext,
    perform,
)


@dataclass(frozen=True, slots=True)
class Greet(Effect[str]):
    """Request a greeting for one name."""

    name: str


def greet(effect: Greet, context: ExecutionContext[None]) -> str:
    """Interpret ``Greet`` as a synchronous application capability."""

    return f"Hello, {effect.name}!"


def flow() -> AgentFlow[str]:
    """Yield one declarative greeting request and return its result."""

    return (yield from perform(Greet("Purrcept")))


async def main() -> None:
    """Compose the handler, driver, and event loop explicitly."""

    executor = DispatchExecutor({Greet: greet})
    result = await AgentDriver(executor).run(flow(), host=None)
    assert result == "Hello, Purrcept!"
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
