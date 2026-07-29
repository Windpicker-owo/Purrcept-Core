"""Compose typed content blocks into immutable conversational messages.

Messages preserve semantic role, optional participant name, and recursively
frozen metadata while leaving provider serialization to a backend. Convenience
constructors create common text and tool-result messages; the ``text`` view is
a read-only projection and does not discard non-text content.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from ._utils import (
    JsonValue,
    empty_json_object,
    freeze_json_object,
    validate_optional_non_empty_string,
)
from .content import ContentBlock, TextBlock, ToolResultBlock


class MessageRole(StrEnum):
    """The semantic role of a model message."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    """An immutable message composed of one or more typed content blocks.

    Construction snapshots the content iterable and metadata. Role-specific
    semantic constraints for requests and responses are enforced by their
    enclosing value objects rather than by this reusable container.
    """

    role: MessageRole
    content: tuple[ContentBlock, ...]
    name: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        try:
            role = MessageRole(self.role)
        except ValueError as error:
            raise ValueError(f"Unsupported message role: {self.role!r}.") from error
        content = _normalize_content(self.content)
        validate_optional_non_empty_string(self.name, field_name="name")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(self.metadata, field_name="metadata"),
        )

    @classmethod
    def from_text(
        cls,
        role: MessageRole | str,
        text: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Message:
        """Create a one-block text message."""

        try:
            resolved_role = MessageRole(role)
        except ValueError as error:
            raise ValueError(f"Unsupported message role: {role!r}.") from error
        return cls(
            resolved_role,
            (TextBlock(text),),
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def user(
        cls,
        text: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Message:
        """Create a user text message."""

        return cls.from_text(MessageRole.USER, text, name=name, metadata=metadata)

    @classmethod
    def assistant(
        cls,
        text: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Message:
        """Create an assistant text message."""

        return cls.from_text(MessageRole.ASSISTANT, text, name=name, metadata=metadata)

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        content: str | Iterable[ContentBlock],
        *,
        is_error: bool = False,
        name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Message:
        """Create a tool message tied to a prior tool call."""

        blocks = (TextBlock(content),) if isinstance(content, str) else tuple(content)
        result = ToolResultBlock(tool_call_id, content=blocks, is_error=is_error)
        return cls(
            MessageRole.TOOL,
            (result,),
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @property
    def text(self) -> str:
        """Concatenate text blocks, including text nested in tool results."""

        return "".join(_iter_text(self.content))


def _iter_text(blocks: Iterable[ContentBlock]) -> Iterable[str]:
    """Yield textual leaves without flattening or mutating the block tree."""

    for block in blocks:
        if isinstance(block, TextBlock):
            yield block.text
        elif isinstance(block, ToolResultBlock):
            yield from _iter_text(block.content)


def _normalize_content(value: object) -> tuple[ContentBlock, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("content must be an iterable of ContentBlock instances.")
    content = tuple(cast(Iterable[object], value))
    if not content:
        raise ValueError("content must contain at least one ContentBlock.")
    if not all(isinstance(block, ContentBlock) for block in content):
        raise TypeError("content must contain only ContentBlock instances.")
    return cast(tuple[ContentBlock, ...], content)


__all__ = ["Message", "MessageRole"]
