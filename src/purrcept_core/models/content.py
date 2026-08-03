"""Define immutable provider-neutral blocks used in messages and tool exchange.

Core request and response values compose these blocks without assuming a
provider wire format. The built-ins cover visible text, hidden model reasoning,
images, tool calls, and tool results. Provider or application extensions may
subclass :class:`ContentBlock`, but conversion to a provider payload remains
the backend's responsibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from ._utils import (
    JsonValue,
    empty_json_object,
    freeze_json_object,
    require_non_empty_string,
    validate_optional_non_empty_string,
)


class ContentBlock:
    """Extension marker for immutable values that may appear in message content.

    Applications and provider packages may define subclasses. Instances can be
    snapshotted into messages and compared across stream/final responses, so
    extensions must behave as immutable values and must not own live resources.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TextBlock(ContentBlock):
    """Plain text content."""

    text: str

    def __post_init__(self) -> None:
        _validate_text(self.text)


@dataclass(frozen=True, slots=True)
class ReasoningBlock(ContentBlock):
    """Hidden assistant reasoning that may be replayed to a thinking model.

    Backends create this block from a provider's structured reasoning field,
    and conversation checkpoints may retain it so a later request can reproduce
    the provider's exact assistant turn. It is deliberately distinct from
    :class:`TextBlock`: user interfaces, ``Message.text``, world projections,
    and ordinary application logs must not expose it as visible assistant text.

    The value contains only the model-returned reasoning text. Opaque provider
    cursors, signatures, and transport state remain ``ModelContinuation`` data
    and are governed by their separate persistence policy.
    """

    text: str

    def __post_init__(self) -> None:
        _validate_text(self.text)


@dataclass(frozen=True, slots=True)
class ImageUrl:
    """An image addressed by URL."""

    url: str

    def __post_init__(self) -> None:
        require_non_empty_string(self.url, field_name="url")


@dataclass(frozen=True, slots=True)
class ImageBytes:
    """Raw image bytes and their media type."""

    data: bytes
    media_type: str

    def __post_init__(self) -> None:
        _validate_image_bytes(self.data)
        require_non_empty_string(self.media_type, field_name="media_type")


@dataclass(frozen=True, slots=True)
class ImageBlock(ContentBlock):
    """Image content backed by a URL or an in-memory byte snapshot."""

    source: ImageUrl | ImageBytes
    alt_text: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_image_source(self.source)
        validate_optional_non_empty_string(self.alt_text, field_name="alt_text")


@dataclass(frozen=True, slots=True)
class ToolCallBlock(ContentBlock):
    """A model request to invoke one named tool."""

    id: str
    name: str
    arguments: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        require_non_empty_string(self.id, field_name="id")
        require_non_empty_string(self.name, field_name="name")
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, field_name="arguments"),
        )


@dataclass(frozen=True, slots=True)
class ToolResultBlock(ContentBlock):
    """The content returned by a prior tool call."""

    tool_call_id: str
    content: tuple[ContentBlock, ...] = field(default=(), kw_only=True)
    is_error: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        content = _normalize_content(self.content)
        _validate_is_error(self.is_error)
        object.__setattr__(self, "content", content)


def _normalize_content(value: object) -> tuple[ContentBlock, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("content must be an iterable of ContentBlock instances.")
    content = tuple(cast(Iterable[object], value))
    if not all(isinstance(block, ContentBlock) for block in content):
        raise TypeError("content must contain only ContentBlock instances.")
    return cast(tuple[ContentBlock, ...], content)


def _validate_text(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("text must be a string.")


def _validate_image_bytes(value: object) -> None:
    if not isinstance(value, bytes):
        raise TypeError("data must be bytes.")


def _validate_image_source(value: object) -> None:
    if not isinstance(value, (ImageUrl, ImageBytes)):
        raise TypeError("source must be an ImageUrl or ImageBytes.")


def _validate_is_error(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("is_error must be a bool.")


__all__ = [
    "ContentBlock",
    "ImageBlock",
    "ImageBytes",
    "ImageUrl",
    "ReasoningBlock",
    "TextBlock",
    "ToolCallBlock",
    "ToolResultBlock",
]
