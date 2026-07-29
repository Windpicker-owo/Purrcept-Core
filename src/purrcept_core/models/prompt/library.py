"""Store reusable prompt templates in an immutable application-local registry.

``PromptLibrary`` snapshots templates in registration order, rejects duplicate
names, and supports strict or optional lookup. It intentionally has no global
registry, import-time discovery, or mutation API, so applications own library
composition and lifecycle explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from .errors import PromptDefinitionError, UnknownPromptError
from .template import PromptTemplate


@dataclass(frozen=True, slots=True, init=False)
class PromptLibrary:
    """A duplicate-free prompt registry with no global or import-time state.

    Construction copies the iterable into a read-only name mapping. Template
    objects are already immutable, so resolved values may be safely reused
    across conversations and render calls.
    """

    _templates: Mapping[str, PromptTemplate] = field(repr=False)

    def __init__(self, templates: Iterable[PromptTemplate] = ()) -> None:
        raw_templates = cast(object, templates)
        if isinstance(raw_templates, (str, bytes)) or not isinstance(raw_templates, Iterable):
            raise TypeError("templates must be an iterable of PromptTemplate instances.")
        by_name: dict[str, PromptTemplate] = {}
        for value in cast(Iterable[object], raw_templates):
            if not isinstance(value, PromptTemplate):
                raise TypeError("templates must contain only PromptTemplate instances.")
            if value.name in by_name:
                raise PromptDefinitionError(
                    f"Prompt template {value.name!r} is registered more than once.",
                    template_name=value.name,
                )
            by_name[value.name] = value
        object.__setattr__(self, "_templates", MappingProxyType(by_name))

    @property
    def templates(self) -> Mapping[str, PromptTemplate]:
        """Return the immutable name-to-template mapping."""

        return self._templates

    def resolve(self, name: str) -> PromptTemplate:
        """Resolve a template or raise :class:`UnknownPromptError`."""

        template = self.get(name)
        if template is None:
            raise UnknownPromptError(name)
        return template

    def get(self, name: str) -> PromptTemplate | None:
        """Resolve a template, returning ``None`` when it is absent."""

        if not isinstance(cast(object, name), str):
            raise TypeError("name must be a string.")
        return self._templates.get(name)

    def __iter__(self) -> Iterator[PromptTemplate]:
        return iter(self._templates.values())

    def __len__(self) -> int:
        return len(self._templates)


__all__ = ["PromptLibrary"]
