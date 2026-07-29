"""Define stable tool definition, invocation, and encoding failures.

Definition errors belong to application setup. Argument and unknown-tool errors
describe model-provided calls and can be converted to ``ToolResult`` values by
the invocation effect. Encoding errors identify an unsupported Python return
type. Arbitrary exceptions raised inside a tool remain application exceptions
unless its explicit error policy converts them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from ...errors import PurrceptError


def _require_tool_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("tool_name must be a string.")
    if not value:
        raise ValueError("tool_name must not be empty.")
    return value


class UnknownToolError(PurrceptError):
    """A requested tool is not present in the local tool set."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = _require_tool_name(tool_name)
        super().__init__(f"Unknown tool {self.tool_name!r}.")


class ToolArgumentsError(PurrceptError):
    """Model-provided arguments failed validation for one tool."""

    def __init__(self, tool_name: str, issues: Iterable[str]) -> None:
        self.tool_name = _require_tool_name(tool_name)
        raw_issues = cast(object, issues)
        if isinstance(raw_issues, (str, bytes)) or not isinstance(
            raw_issues,
            Iterable,
        ):
            raise TypeError("issues must be an iterable of strings.")
        issue_items = tuple(cast(Iterable[object], raw_issues))
        if not all(isinstance(issue, str) for issue in issue_items):
            raise TypeError("issues must contain only strings.")
        self.issues = cast(tuple[str, ...], issue_items)
        detail = "; ".join(self.issues) if self.issues else "arguments are invalid"
        super().__init__(f"Invalid arguments for tool {self.tool_name!r}: {detail}.")


class ToolInputValidationError(ToolArgumentsError):
    """Internal marker distinguishing adapter validation from tool exceptions."""


class ToolDefinitionError(PurrceptError):
    """A Python callable cannot be exposed as a model tool."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        if not isinstance(cast(object, message), str):
            raise TypeError("message must be a string.")
        if not message:
            raise ValueError("message must not be empty.")
        if tool_name is not None:
            tool_name = _require_tool_name(tool_name)
        self.tool_name = tool_name
        super().__init__(message)


class ToolResultEncodingError(PurrceptError):
    """A Python return value cannot be represented as model tool content."""

    def __init__(self, tool_name: str, value: object) -> None:
        self.tool_name = _require_tool_name(tool_name)
        value_type = type(value)
        self.value_type = value_type
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        super().__init__(f"Tool {self.tool_name!r} returned unsupported value of type {type_name}.")


__all__ = [
    "ToolArgumentsError",
    "ToolDefinitionError",
    "ToolResultEncodingError",
    "UnknownToolError",
]
