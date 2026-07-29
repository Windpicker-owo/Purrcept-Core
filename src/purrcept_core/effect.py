"""Define the typed, runtime-neutral description yielded by agent flows.

Effects describe capability requests and carry only application-defined input.
The core does not prescribe how they execute: a runtime chooses an executor,
and the generic result parameter lets ``perform`` preserve the value returned
to generator code.
"""

from typing import Generic, TypeVar

EffectResultT_co = TypeVar("EffectResultT_co", covariant=True)


class Effect(Generic[EffectResultT_co]):
    """Describe one capability request whose successful result has type ``T``.

    Effects intentionally carry no execution or run state in the core model.
    Applications and extension packages create concrete subclasses, commonly
    as immutable slotted dataclasses. Executors or inline ``execute`` methods
    implement the corresponding capability and may raise ordinary exceptions
    for the agent flow to handle.
    """

    __slots__ = ()


__all__ = ["Effect"]
