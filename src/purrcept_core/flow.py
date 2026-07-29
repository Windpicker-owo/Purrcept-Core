"""Author synchronous generator flows that request typed runtime effects.

An :data:`AgentFlow` contains application control flow only: it yields effect
descriptions, receives executor results, and ultimately returns a value. The
driver supplies asynchronous execution around that synchronous state machine.
Use :func:`perform` to preserve an effect's result type while retaining normal
generator exception handling and ``finally`` semantics.
"""

from collections.abc import Generator
from typing import Any, TypeAlias, TypeVar

from .effect import Effect

FlowResultT = TypeVar("FlowResultT")
EffectResultT = TypeVar("EffectResultT")

AgentFlow: TypeAlias = Generator[Effect[Any], Any, FlowResultT]
"""A single-use synchronous generator driven through the :class:`AgentRun` protocol."""


def perform(
    effect: Effect[EffectResultT],
) -> Generator[Effect[EffectResultT], EffectResultT, EffectResultT]:
    """Yield ``effect`` and return the value sent back by the run driver.

    ``yield from perform(effect)`` preserves the result type parameter carried
    by the effect, while retaining ordinary Python generator semantics for
    exceptions and ``finally`` blocks.
    """

    result = yield effect
    return result


__all__ = ["AgentFlow", "perform"]
