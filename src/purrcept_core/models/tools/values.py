"""Expose Pydantic-free tool metadata, runtime context, results, and policies.

Applications use ``ToolParameter`` inside ``Annotated`` signatures and may
receive ``ToolContext`` through dependency injection. ``FunctionTool`` returns
immutable ``ToolResult`` values and interprets the error and synchronous
execution policies defined here. Schema generation remains an internal adapter
detail.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, cast

from ...context import ExecutionContext
from ..content import ContentBlock, ToolCallBlock, ToolResultBlock


def _validate_optional_number(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")


def _validate_optional_length(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Immutable schema metadata attached to one parameter through ``Annotated``.

    The function-tool adapter consumes at most one instance per model-visible
    parameter and translates supported fields to validation constraints.
    """

    description: str | None = field(default=None, kw_only=True)
    ge: int | float | None = field(default=None, kw_only=True)
    le: int | float | None = field(default=None, kw_only=True)
    gt: int | float | None = field(default=None, kw_only=True)
    lt: int | float | None = field(default=None, kw_only=True)
    min_length: int | None = field(default=None, kw_only=True)
    max_length: int | None = field(default=None, kw_only=True)
    pattern: str | None = field(default=None, kw_only=True)
    strict: bool | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.description is not None:
            if not isinstance(cast(object, self.description), str):
                raise TypeError("description must be a string.")
            if not self.description:
                raise ValueError("description must not be empty.")
        for field_name in ("ge", "le", "gt", "lt"):
            _validate_optional_number(getattr(self, field_name), field_name=field_name)
        _validate_optional_length(self.min_length, field_name="min_length")
        _validate_optional_length(self.max_length, field_name="max_length")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length must not be greater than max_length.")
        if self.pattern is not None and not isinstance(
            cast(object, self.pattern),
            str,
        ):
            raise TypeError("pattern must be a string.")
        if self.strict is not None and not isinstance(
            cast(object, self.strict),
            bool,
        ):
            raise TypeError("strict must be a bool.")


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Runtime-only context injected into an exactly annotated tool parameter.

    The adapter excludes this parameter from model-facing JSON Schema and
    creates the value for each invocation. It borrows the surrounding execution
    context and tool call; retaining it does not extend the effect's progress
    lifetime.
    """

    execution: ExecutionContext[Any]
    call: ToolCallBlock

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.execution), ExecutionContext):
            raise TypeError("execution must be an ExecutionContext.")
        if not isinstance(cast(object, self.call), ToolCallBlock):
            raise TypeError("call must be a ToolCallBlock.")

    @property
    def run_id(self) -> str:
        """The current agent run identifier."""

        return self.execution.run_id

    @property
    def effect_id(self) -> str:
        """The current effect identifier."""

        return self.execution.effect_id

    @property
    def step_index(self) -> int:
        """The current effect's zero-based step index."""

        return self.execution.step_index

    @property
    def host(self) -> Any:
        """The runtime host supplied to the agent driver."""

        return self.execution.host

    @property
    def metadata(self) -> Mapping[str, Any]:
        """The current run's read-only metadata snapshot."""

        return self.execution.metadata

    def progress(self, payload: object) -> None:
        """Report progress for the surrounding InvokeTool effect."""

        self.execution.progress(payload)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Immutable provider-neutral content produced by a Python function tool."""

    content: tuple[ContentBlock, ...] = field(default=(), kw_only=True)
    is_error: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        raw_content = cast(object, self.content)
        if not isinstance(raw_content, Iterable):
            raise TypeError("content must be an iterable of ContentBlock instances.")
        content = tuple(cast(Iterable[object], raw_content))
        if not all(isinstance(block, ContentBlock) for block in content):
            raise TypeError("content must contain only ContentBlock instances.")
        if not isinstance(cast(object, self.is_error), bool):
            raise TypeError("is_error must be a bool.")
        object.__setattr__(self, "content", cast(tuple[ContentBlock, ...], content))

    def to_block(self, tool_call_id: str) -> ToolResultBlock:
        """Bind this result to a model tool call."""

        return ToolResultBlock(
            tool_call_id,
            content=self.content,
            is_error=self.is_error,
        )


class ToolErrorPolicy(StrEnum):
    """How Python exceptions raised by a tool callable are handled."""

    PROPAGATE = "propagate"
    RETURN_TO_MODEL = "return_to_model"


class SyncToolPolicy(StrEnum):
    """Where a synchronous tool callable executes."""

    INLINE = "inline"
    THREAD = "thread"


__all__ = [
    "SyncToolPolicy",
    "ToolContext",
    "ToolErrorPolicy",
    "ToolParameter",
    "ToolResult",
]
