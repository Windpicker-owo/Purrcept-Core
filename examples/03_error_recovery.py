"""Demonstrate ordinary Python recovery from an effect implementation failure.

The driver throws ``ZeroDivisionError`` back into the suspended flow, where a
normal ``try``/``except`` branch yields a fallback request.
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
class Divide(Effect[float]):
    """Request floating-point division."""

    numerator: float
    denominator: float


def divide(effect: Divide, context: ExecutionContext[None]) -> float:
    """Interpret division and surface invalid input as an ordinary exception."""

    if effect.denominator == 0:
        raise ZeroDivisionError("division by zero")
    return effect.numerator / effect.denominator


def resilient_flow() -> AgentFlow[float]:
    """Recover from the first failed effect by yielding a safe alternative."""

    try:
        return (yield from perform(Divide(10, 0)))
    except ZeroDivisionError:
        return (yield from perform(Divide(10, 2)))


async def main() -> None:
    """Run the recovery flow through an explicit dispatch table."""

    driver = AgentDriver(DispatchExecutor({Divide: divide}))
    result = await driver.run(resilient_flow(), host=None)
    assert result == 5
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
