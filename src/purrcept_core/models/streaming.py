"""Define immutable provider-neutral events for incremental model output.

Backends synchronously emit these values through ``ModelEventSink`` while
``generate`` is active. The ``Generate`` effect validates event ordering and
forwards accepted events as effect progress; this module owns only event shapes
and field validation, not stream aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from ._utils import (
    JsonValue,
    empty_json_object,
    freeze_json_object,
    validate_optional_non_empty_string,
)
from .responses import ModelResponse, TokenUsage


class ModelStreamEvent:
    """Extension marker for immutable snapshots emitted during generation.

    Provider packages may define additional event subclasses. Events must remain
    immutable because observers may retain them after the backend call returns.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ModelStreamStarted(ModelStreamEvent):
    """The provider accepted a generation and began streaming."""

    model: str | None = field(default=None, kw_only=True)
    response_id: str | None = field(default=None, kw_only=True)
    provider_metadata: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        validate_optional_non_empty_string(self.model, field_name="model")
        validate_optional_non_empty_string(self.response_id, field_name="response_id")
        object.__setattr__(
            self,
            "provider_metadata",
            freeze_json_object(self.provider_metadata, field_name="provider_metadata"),
        )


@dataclass(frozen=True, slots=True)
class TextDelta(ModelStreamEvent):
    """A text fragment targeting one top-level response content index."""

    delta: str
    index: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        _validate_string(self.delta, field_name="delta")
        _validate_index_type(self.index)
        if self.index < 0:
            raise ValueError("index must be greater than or equal to zero.")


@dataclass(frozen=True, slots=True)
class ReasoningDelta(ModelStreamEvent):
    """Readable provider reasoning, kept separate from assistant answer text.

    Backends emit only text the provider actually exposes, never opaque
    signatures or encrypted continuation state. ``index`` identifies a block
    within this reasoning channel; observers key it with the event type rather
    than assuming it is an index into a subsequently normalized final message.
    ``is_summary`` identifies provider-supplied summaries. These previews do not
    commit conversation history or replace the final response's reasoning state.
    """

    delta: str
    index: int = field(default=0, kw_only=True)
    is_summary: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Validate text and channel identity before observers retain the event."""

        _validate_string(self.delta, field_name="delta")
        _validate_index_type(self.index)
        if self.index < 0:
            raise ValueError("index must be greater than or equal to zero.")
        if not isinstance(cast(object, self.is_summary), bool):
            raise TypeError("is_summary must be a bool.")


@dataclass(frozen=True, slots=True)
class ToolCallDelta(ModelStreamEvent):
    """A tool-call fragment targeting one top-level response content index."""

    index: int
    arguments_delta: str
    tool_call_id: str | None = field(default=None, kw_only=True)
    name: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_index_type(self.index)
        if self.index < 0:
            raise ValueError("index must be greater than or equal to zero.")
        _validate_string(self.arguments_delta, field_name="arguments_delta")
        validate_optional_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        validate_optional_non_empty_string(self.name, field_name="name")


@dataclass(frozen=True, slots=True)
class UsageUpdate(ModelStreamEvent):
    """A replacement usage snapshot reported during streaming."""

    usage: TokenUsage

    def __post_init__(self) -> None:
        if not _is_token_usage(self.usage):
            raise TypeError("usage must be a TokenUsage.")


@dataclass(frozen=True, slots=True)
class ModelStreamCompleted(ModelStreamEvent):
    """The final response for a completed model stream."""

    response: ModelResponse

    def __post_init__(self) -> None:
        if not _is_model_response(self.response):
            raise TypeError("response must be a ModelResponse.")


def _validate_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")


def _validate_index_type(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("index must be an int.")


def _is_token_usage(value: object) -> bool:
    return isinstance(value, TokenUsage)


def _is_model_response(value: object) -> bool:
    return isinstance(value, ModelResponse)


__all__ = [
    "ModelStreamCompleted",
    "ModelStreamEvent",
    "ModelStreamStarted",
    "ReasoningDelta",
    "TextDelta",
    "ToolCallDelta",
    "UsageUpdate",
]
