"""Expose structured failures from prompt definition, lookup, and compilation.

Authoring errors identify an invalid reusable template, render errors retain the
template and field context for one value set, and compile errors describe
conflicts among otherwise valid prompt-control inputs. Original policy or source
failures remain available through normal exception chaining.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from ...errors import PurrceptError
from .._utils import require_non_empty_string, validate_optional_non_empty_string


class PromptError(PurrceptError):
    """Base class for prompt definition, lookup, and rendering failures."""


class PromptDefinitionError(PromptError):
    """A prompt template or prompt library definition is invalid."""

    def __init__(self, message: str, *, template_name: str | None = None) -> None:
        message = require_non_empty_string(message, field_name="message")
        validate_optional_non_empty_string(template_name, field_name="template_name")
        self.template_name = template_name
        super().__init__(message)


class PromptRenderError(PromptError):
    """A prompt template could not be rendered with the supplied values."""

    def __init__(
        self,
        message: str,
        *,
        template_name: str,
        field_name: str | None = None,
        missing_fields: Iterable[str] = (),
    ) -> None:
        message = require_non_empty_string(message, field_name="message")
        self.template_name = require_non_empty_string(
            template_name,
            field_name="template_name",
        )
        validate_optional_non_empty_string(field_name, field_name="field_name")
        raw_missing = cast(object, missing_fields)
        if isinstance(raw_missing, (str, bytes)) or not isinstance(raw_missing, Iterable):
            raise TypeError("missing_fields must be an iterable of strings.")
        missing = tuple(cast(Iterable[object], raw_missing))
        if not all(isinstance(item, str) for item in missing):
            raise TypeError("missing_fields must contain only strings.")
        if any(not item for item in cast(tuple[str, ...], missing)):
            raise ValueError("missing_fields must not contain empty strings.")
        self.field_name = field_name
        self.missing_fields = cast(tuple[str, ...], missing)
        super().__init__(message)


class PromptCompileError(PromptError):
    """Current prompt-control inputs cannot form one unambiguous request."""

    def __init__(self, message: str, *, reminder_key: str | None = None) -> None:
        message = require_non_empty_string(message, field_name="message")
        validate_optional_non_empty_string(reminder_key, field_name="reminder_key")
        self.reminder_key = reminder_key
        super().__init__(message)


class UnknownPromptError(PromptError):
    """A requested template is not present in a local prompt library."""

    def __init__(self, name: str) -> None:
        self.name = require_non_empty_string(name, field_name="name")
        super().__init__(f"Unknown prompt template {self.name!r}.")


__all__ = [
    "PromptCompileError",
    "PromptDefinitionError",
    "PromptError",
    "PromptRenderError",
    "UnknownPromptError",
]
