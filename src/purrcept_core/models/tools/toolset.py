"""Store locally invokable function tools in an immutable ordered registry.

``ToolSet`` adapts plain callables, rejects duplicate names, and snapshots both
the invocation mapping and model-facing specifications. It performs no global
registration or discovery. The conversation model loop uses optional lookup so
an unknown model request can become a tool error result rather than an
orchestration failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Self, cast

from ..requests import ToolSpec
from .errors import ToolDefinitionError, UnknownToolError
from .function import FunctionTool, ToolCallable, ToolLike


@dataclass(frozen=True, slots=True, init=False)
class ToolSet:
    """An immutable duplicate-free registry preserving registration order.

    Applications build a set once and may share it across conversations.
    Callable objects remain caller-owned through their ``FunctionTool`` wrappers;
    the registry itself never mutates them.
    """

    _tools: Mapping[str, FunctionTool] = field(repr=False)
    _specs: tuple[ToolSpec, ...]

    def __init__(self, tools: Iterable[ToolLike] = ()) -> None:
        raw_tools = cast(object, tools)
        if isinstance(raw_tools, (str, bytes)) or not isinstance(
            raw_tools,
            Iterable,
        ):
            raise TypeError("tools must be an iterable of FunctionTool or callable values.")

        by_name: dict[str, FunctionTool] = {}
        for value in cast(Iterable[object], raw_tools):
            if isinstance(value, FunctionTool):
                resolved = value
            elif callable(value):
                resolved = FunctionTool(value)
            else:
                raise TypeError("tools must contain only FunctionTool or callable values.")
            if resolved.name in by_name:
                raise ToolDefinitionError(
                    f"Tool name {resolved.name!r} is registered more than once.",
                    tool_name=resolved.name,
                )
            by_name[resolved.name] = resolved

        object.__setattr__(self, "_tools", MappingProxyType(by_name))
        object.__setattr__(
            self,
            "_specs",
            tuple(value.spec for value in by_name.values()),
        )

    @property
    def tools(self) -> Mapping[str, FunctionTool]:
        """Return the immutable name-to-tool mapping."""

        return self._tools

    @classmethod
    def from_callables(cls, callables: Iterable[ToolCallable]) -> Self:
        """Create a tool set from undecorated Python callables."""

        return cls(callables)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return model-facing specifications in registration order."""

        return self._specs

    def resolve(self, name: str) -> FunctionTool:
        """Resolve a registered tool or raise UnknownToolError."""

        tool = self.get(name)
        if tool is None:
            raise UnknownToolError(name)
        return tool

    def get(self, name: str) -> FunctionTool | None:
        """Resolve a registered tool, returning None when it is absent."""

        if not isinstance(cast(object, name), str):
            raise TypeError("name must be a string.")
        return self._tools.get(name)

    def __iter__(self) -> Iterator[FunctionTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


__all__ = ["ToolSet"]
