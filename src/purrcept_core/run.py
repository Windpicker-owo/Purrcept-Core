"""Represent agent execution as a small synchronous state-machine protocol.

The asynchronous driver consumes :class:`AgentRun` rather than depending
directly on Python generators. This module owns that protocol, its explicit
``Yielded`` and ``Returned`` states, and :class:`GeneratorRun`, the adapter used
for ordinary generator-based flows. It does not execute effects or schedule
async work.

Each run advances synchronously by exactly one state transition. A
``GeneratorRun`` is single-use, records terminal state even when generator code
raises, and closes idempotently so driver cleanup cannot resume it.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast, overload, runtime_checkable

from .effect import Effect
from .errors import InvalidRunStateError
from .flow import AgentFlow

RunResultT_co = TypeVar("RunResultT_co", covariant=True)
RunResultT = TypeVar("RunResultT")


@dataclass(frozen=True, slots=True)
class Yielded:
    """A run state that requests execution of one effect."""

    effect: Effect[Any]


@dataclass(frozen=True, slots=True)
class Returned(Generic[RunResultT_co]):
    """A run state that contains the final result."""

    value: RunResultT_co


RunState: TypeAlias = Yielded | Returned[RunResultT_co]
"""Exactly one effect request or terminal value returned by a run transition."""


@runtime_checkable
class AgentRun(Protocol[RunResultT_co]):
    """Extension contract consumed synchronously by an agent driver.

    Applications may provide a custom run object or use the built-in
    :class:`GeneratorRun` adapter. The driver creates no copies: it calls
    :meth:`start` once, then :meth:`send` or :meth:`throw` after each effect,
    and :meth:`close` on abnormal exit. Every advancement must return
    :class:`Yielded` or :class:`Returned` and must not perform asynchronous
    work.

    A run owns its local state and resources and is not reused after a terminal
    result or failure. Exceptions raised by state transitions propagate to the
    driver. Structural runtime checks select custom implementations before
    generator adaptation.
    """

    def start(self) -> RunState[RunResultT_co]:
        """Start the run and return its first state."""

        ...

    def send(self, value: object) -> RunState[RunResultT_co]:
        """Send a successful effect result into the run."""

        ...

    def throw(self, error: BaseException) -> RunState[RunResultT_co]:
        """Send an effect failure into the run."""

        ...

    def close(self) -> None:
        """Close the run and release its local resources."""

        ...


class _GeneratorRunState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    RETURNED = "RETURNED"
    CLOSED = "CLOSED"


class GeneratorRun(Generic[RunResultT]):
    """Adapt one synchronous :class:`AgentFlow` to the :class:`AgentRun` contract.

    The adapter owns the generator from construction through return, failure,
    or explicit close. It enforces a single linear lifecycle and preserves
    Python's normal ``send``, ``throw``, ``finally``, and return-value
    semantics.
    """

    __slots__ = ("_flow", "_state")

    def __init__(self, flow: AgentFlow[RunResultT]) -> None:
        self._flow = flow
        self._state = _GeneratorRunState.CREATED

    def start(self) -> RunState[RunResultT]:
        """Start a newly created generator run."""

        self._require_state(_GeneratorRunState.CREATED, "start")
        return self._advance(lambda: next(self._flow))

    def send(self, value: object) -> RunState[RunResultT]:
        """Send an effect result to a running generator."""

        self._require_state(_GeneratorRunState.RUNNING, "send a value to")
        return self._advance(lambda: self._flow.send(value))

    def throw(self, error: BaseException) -> RunState[RunResultT]:
        """Throw an effect failure into a running generator."""

        self._require_state(_GeneratorRunState.RUNNING, "throw an error into")
        return self._advance(lambda: self._flow.throw(error))

    def close(self) -> None:
        """Close a created or running generator.

        Closing is idempotent. A normally returned run remains in the
        ``RETURNED`` state because its generator is already exhausted.
        """

        if self._state in {_GeneratorRunState.CLOSED, _GeneratorRunState.RETURNED}:
            return

        try:
            self._flow.close()
        finally:
            self._state = _GeneratorRunState.CLOSED

    def _advance(self, advance: Callable[[], Effect[Any]]) -> RunState[RunResultT]:
        """Advance once while recording terminal state before propagation."""

        try:
            effect = advance()
        except StopIteration as returned:
            self._state = _GeneratorRunState.RETURNED
            return Returned(cast(RunResultT, returned.value))
        except BaseException:
            self._state = _GeneratorRunState.CLOSED
            raise

        self._state = _GeneratorRunState.RUNNING
        return Yielded(effect)

    def _require_state(self, expected: _GeneratorRunState, operation: str) -> None:
        """Reject transitions that violate the generator's linear lifecycle."""

        if self._state is expected:
            return

        raise InvalidRunStateError(
            f"Cannot {operation} GeneratorRun while it is in "
            f"{self._state.value} state; expected {expected.value}."
        )


@overload
def as_agent_run(run: AgentRun[RunResultT]) -> AgentRun[RunResultT]:
    """Return an existing run implementation unchanged."""

    ...


@overload
def as_agent_run(run: AgentFlow[RunResultT]) -> GeneratorRun[RunResultT]:
    """Wrap a synchronous generator flow in :class:`GeneratorRun`."""

    ...


def as_agent_run(run: object) -> AgentRun[Any]:
    """Return an ``AgentRun``, adapting a synchronous generator when needed."""

    if isinstance(run, AgentRun):
        return cast(AgentRun[Any], run)
    if isinstance(run, Generator):
        return GeneratorRun(cast(AgentFlow[Any], run))

    raise TypeError(
        "Expected an AgentRun or synchronous generator, "
        f"got {type(run).__module__}.{type(run).__qualname__}."
    )


__all__ = [
    "AgentRun",
    "GeneratorRun",
    "Returned",
    "RunState",
    "Yielded",
    "as_agent_run",
]
