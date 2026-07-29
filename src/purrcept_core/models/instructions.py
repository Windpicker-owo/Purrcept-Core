"""Represent immutable system instructions outside conversational history.

Prompt compilation carries :class:`SystemInstruction` values separately from
messages so provider adapters can place and cache them appropriately. Each
instance snapshots typed content, an optional stable key, and its expected
cache stability; it does not determine final provider placement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from ._utils import validate_optional_non_empty_string
from .caching import PromptStability
from .content import ContentBlock, TextBlock


@dataclass(frozen=True, slots=True)
class SystemInstruction:
    """Stable system guidance supplied outside the conversational transcript.

    Construction validates and snapshots content. A non-empty ``key`` gives
    prompt compilation a stable identity, while ``stability`` remains a hint to
    provider adapters.
    """

    content: tuple[ContentBlock, ...]
    key: str | None = field(default=None, kw_only=True)
    stability: PromptStability = field(default=PromptStability.STABLE, kw_only=True)

    def __post_init__(self) -> None:
        content = _normalize_content(self.content)
        validate_optional_non_empty_string(self.key, field_name="key")
        try:
            stability = PromptStability(self.stability)
        except ValueError as error:
            raise ValueError(f"Unsupported prompt stability: {self.stability!r}.") from error
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "stability", stability)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        key: str | None = None,
        stability: PromptStability = PromptStability.STABLE,
    ) -> SystemInstruction:
        """Create a one-block text instruction."""

        return cls((TextBlock(text),), key=key, stability=stability)


def _normalize_content(value: object) -> tuple[ContentBlock, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("content must be an iterable of ContentBlock instances.")
    items = tuple(cast(Iterable[object], value))
    if not items:
        raise ValueError("content must contain at least one ContentBlock.")
    if not all(isinstance(item, ContentBlock) for item in items):
        raise TypeError("content must contain only ContentBlock instances.")
    return cast(tuple[ContentBlock, ...], items)


__all__ = ["SystemInstruction"]
