"""Transform prompt placeholder values through small synchronous policies.

``RenderPolicy`` wraps one value-to-text transform and composes transforms in
left-to-right order. Factory functions provide common absence, whitespace,
section, wrapping, joining, and length behavior. Policies are invoked only
during local template rendering; they do not resolve runtime state or perform
asynchronous work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set, Sized
from dataclasses import dataclass
from typing import TypeAlias, cast

PromptTransform: TypeAlias = Callable[[object], str]
"""Synchronous value-to-text function wrapped by a ``RenderPolicy``."""


def is_effectively_empty(value: object) -> bool:
    """Return whether a value should be treated as absent by prompt policies."""

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Mapping, Sequence, Set)):
        return len(cast(Sized, value)) == 0
    return False


@dataclass(frozen=True, slots=True)
class RenderPolicy:
    """An immutable synchronous placeholder transform that can be chained.

    Prompt authors construct policies directly or through this module's
    factories and attach them by field name to a ``PromptTemplate``. A transform
    may raise an ordinary exception; the template layer translates it to a
    field-specific rendering error. Returning anything other than ``str`` is a
    contract violation.
    """

    transform: PromptTransform

    def __post_init__(self) -> None:
        if not callable(cast(object, self.transform)):
            raise TypeError("transform must be callable.")

    def __call__(self, value: object) -> str:
        transform = cast(Callable[[object], object], self.transform)
        rendered = transform(value)
        if not isinstance(rendered, str):
            raise TypeError("prompt render policies must return a string.")
        return rendered

    def then(self, other: RenderPolicy) -> RenderPolicy:
        """Return a policy that applies this transform followed by ``other``."""

        if not isinstance(cast(object, other), RenderPolicy):
            raise TypeError("other must be a RenderPolicy.")
        return RenderPolicy(lambda value: other(self(value)))


def optional(empty: str = "") -> RenderPolicy:
    """Render effectively empty values as ``empty``."""

    _require_string(empty, field_name="empty")
    return RenderPolicy(lambda value: empty if is_effectively_empty(value) else str(value))


def trim() -> RenderPolicy:
    """Convert a value to text and remove leading and trailing whitespace."""

    return RenderPolicy(lambda value: "" if value is None else str(value).strip())


def header(title: str, separator: str = "\n") -> RenderPolicy:
    """Prefix non-empty content with a title and separator."""

    _require_string(title, field_name="title")
    _require_string(separator, field_name="separator")

    def transform(value: object) -> str:
        """Add the header only when both input and rendered text are non-empty."""

        if is_effectively_empty(value):
            return ""
        rendered = str(value)
        if is_effectively_empty(rendered):
            return ""
        return f"{title}{separator}{rendered}"

    return RenderPolicy(transform)


def wrap(prefix: str = "", suffix: str = "") -> RenderPolicy:
    """Wrap non-empty content with a prefix and suffix."""

    _require_string(prefix, field_name="prefix")
    _require_string(suffix, field_name="suffix")

    def transform(value: object) -> str:
        """Add delimiters only when both input and rendered text are non-empty."""

        if is_effectively_empty(value):
            return ""
        rendered = str(value)
        if is_effectively_empty(rendered):
            return ""
        return f"{prefix}{rendered}{suffix}"

    return RenderPolicy(transform)


def join_blocks(separator: str = "\n\n") -> RenderPolicy:
    """Join non-empty list or tuple items while rendering other values directly."""

    _require_string(separator, field_name="separator")

    def transform(value: object) -> str:
        """Normalize block items individually before joining retained text."""

        if is_effectively_empty(value):
            return ""
        if isinstance(value, (list, tuple)):
            blocks: list[str] = []
            for item in cast(list[object] | tuple[object, ...], value):
                if is_effectively_empty(item):
                    continue
                rendered = str(item).strip()
                if rendered:
                    blocks.append(rendered)
            return separator.join(blocks)
        return str(value)

    return RenderPolicy(transform)


def min_len(length: int) -> RenderPolicy:
    """Suppress text whose stripped length is below ``length``."""

    raw_length = cast(object, length)
    if isinstance(raw_length, bool) or not isinstance(raw_length, int):
        raise TypeError("length must be an int.")
    if length < 0:
        raise ValueError("length must be greater than or equal to zero.")

    def transform(value: object) -> str:
        """Retain original text only when its stripped form reaches the threshold."""

        if value is None:
            return ""
        rendered = str(value)
        return "" if len(rendered.strip()) < length else rendered

    return RenderPolicy(transform)


def _require_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")


__all__ = [
    "PromptTransform",
    "RenderPolicy",
    "header",
    "is_effectively_empty",
    "join_blocks",
    "min_len",
    "optional",
    "trim",
    "wrap",
]
