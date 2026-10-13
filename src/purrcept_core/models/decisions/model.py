"""Bind one decision capability to the generic Effect execution boundary.

DecisionModel creates Decide effects without I/O. InlineExecutor executes them
through the injected backend and validates returned results. This module owns
neither application routing policies nor backend resources, history, or retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from ...context import ExecutionContext
from ...effect import Effect
from ._validation import text
from .requests import DecisionRequest
from .responses import DecisionResponse, DecisionResponseError


@runtime_checkable
class DecisionBackend(Protocol):
    """Provider extension for one non-streaming decision request.

    The runtime creates and owns implementations, binding them to DecisionModel.
    Decide calls decide once with a validated immutable request and exact model
    name. Backends return a response matching all question names, types, and order;
    refusals are results, while transport failures use the existing ModelError
    taxonomy. Cancellation propagates unchanged. Instances may serve concurrent
    calls and own synchronization, client reuse, and resource shutdown. Core ships
    no implementation; applications select one through explicit model construction.
    """

    async def decide(self, request: DecisionRequest, *, model: str) -> DecisionResponse:
        """Return independent answers over shared evidence without hidden follow-up calls."""
        ...


@dataclass(frozen=True, slots=True)
class DecisionModel:
    """Immutable name/backend binding borrowing all runtime-owned resources."""

    backend: DecisionBackend = field(repr=False)
    name: str

    def __post_init__(self) -> None:
        """Check the structural provider boundary before effects are created."""
        if not isinstance(cast(object, self.backend), DecisionBackend):
            raise TypeError("backend must implement DecisionBackend.")
        text(self.name, "name")

    def decide(self, request: DecisionRequest) -> Decide:
        """Describe one call without starting I/O or applying a business threshold."""
        return Decide(self, request)


@dataclass(frozen=True, slots=True)
class Decide(Effect[DecisionResponse]):
    """One visible decision operation, executable by the existing InlineExecutor.

    Executors and middleware retain responsibility for retries and observation.
    No synthetic token-stream events are emitted for a non-streaming capability.
    """

    model: DecisionModel
    request: DecisionRequest

    def __post_init__(self) -> None:
        """Reject malformed effect inputs before dispatch."""
        if not isinstance(cast(object, self.model), DecisionModel) or not isinstance(
            cast(object, self.request), DecisionRequest
        ):
            raise TypeError("Decide requires a DecisionModel and DecisionRequest.")

    async def execute(self, context: ExecutionContext[object]) -> DecisionResponse:
        """Await one backend call and validate its result; cancellation is not intercepted."""
        response = await self.model.backend.decide(self.request, model=self.model.name)
        if not isinstance(cast(object, response), DecisionResponse):
            raise DecisionResponseError("Decision backend must return DecisionResponse.")
        response.validate_for(self.request)
        return response
