"""Represent final successful model output independently of provider transport.

Backends create :class:`ModelResponse` values after translating provider data.
Construction requires an assistant message and freezes usage, continuation, and
provider metadata. Streaming events may refer to the same final response, but
stream assembly and protocol validation live outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from ._utils import JsonValue, empty_json_object, freeze_json_object
from .content import ToolCallBlock
from .continuation import ModelContinuation
from .messages import Message, MessageRole


class FinishReason(StrEnum):
    """Why a successfully returned model response stopped producing output."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Input and output token counts reported by a provider."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = field(default=0, kw_only=True)
    cache_write_input_tokens: int = field(default=0, kw_only=True)
    reasoning_tokens: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        _validate_token_count(self.input_tokens, field_name="input_tokens")
        _validate_token_count(self.output_tokens, field_name="output_tokens")
        _validate_token_count(
            self.cached_input_tokens,
            field_name="cached_input_tokens",
        )
        _validate_token_count(
            self.cache_write_input_tokens,
            field_name="cache_write_input_tokens",
        )
        _validate_token_count(
            self.reasoning_tokens,
            field_name="reasoning_tokens",
        )

    @property
    def total_tokens(self) -> int:
        """The total number of reported input and output tokens."""

        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """The immutable final successful response returned by a model backend."""

    message: Message
    usage: TokenUsage | None = field(default=None, kw_only=True)
    finish_reason: FinishReason = field(default=FinishReason.STOP, kw_only=True)
    model: str | None = field(default=None, kw_only=True)
    response_id: str | None = field(default=None, kw_only=True)
    continuation: ModelContinuation | None = field(default=None, kw_only=True)
    provider_metadata: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_instance(self.message, Message, field_name="message")
        if self.message.role is not MessageRole.ASSISTANT:
            raise ValueError("message role must be assistant for a model response.")
        if self.usage is not None:
            _validate_instance(self.usage, TokenUsage, field_name="usage")
        try:
            finish_reason = FinishReason(self.finish_reason)
        except ValueError as error:
            raise ValueError(f"Unsupported finish reason: {self.finish_reason!r}.") from error
        _validate_optional_identifier(self.model, field_name="model")
        _validate_optional_identifier(self.response_id, field_name="response_id")
        if self.continuation is not None:
            _validate_instance(
                self.continuation,
                ModelContinuation,
                field_name="continuation",
            )
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(
            self,
            "provider_metadata",
            freeze_json_object(self.provider_metadata, field_name="provider_metadata"),
        )

    @property
    def text(self) -> str:
        """The concatenated text from the response message."""

        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        """Tool calls present in the top-level response content."""

        return tuple(block for block in self.message.content if isinstance(block, ToolCallBlock))


def _validate_token_count(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")


def _validate_optional_identifier(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value:
        raise ValueError(f"{field_name} must not be empty.")


def _validate_instance(value: object, expected_type: type[object], *, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}.")


__all__ = ["FinishReason", "ModelResponse", "TokenUsage"]
