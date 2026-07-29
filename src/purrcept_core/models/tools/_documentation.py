"""Extract conservative model-facing documentation from Python callables.

``FunctionTool`` calls :func:`resolve_documentation` while defining a tool.
Explicit descriptions win; otherwise the callable's first docstring paragraph
or a humanized tool name becomes the summary. A small Google-style parser maps
argument descriptions into generated JSON Schema.

This module intentionally does not implement a complete docstring grammar.
Unknown sections and malformed entries are ignored so documentation quality
cannot change invocation semantics or prevent an otherwise valid tool from
being registered.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from inspect import getdoc

_SECTION_NAMES = frozenset(
    {
        "args",
        "arguments",
        "attributes",
        "examples",
        "keyword args",
        "keyword arguments",
        "notes",
        "parameters",
        "raises",
        "returns",
        "warnings",
        "yields",
    }
)
_ARGUMENT_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)(?:\s*\([^)]*\))?\s*:\s*(.*)$")


def resolve_documentation(
    function: Callable[..., object],
    *,
    tool_name: str,
    explicit_description: str | None,
) -> tuple[str, Mapping[str, str]]:
    """Resolve the tool summary and Google-style argument descriptions."""

    docstring = getdoc(function)
    summary = _first_paragraph(docstring) if docstring is not None else None
    description = (
        explicit_description
        if explicit_description is not None
        else summary or _humanize_name(tool_name)
    )
    parameter_descriptions: Mapping[str, str]
    if docstring is None:
        parameter_descriptions = {}
    else:
        parameter_descriptions = _google_parameter_descriptions(docstring)
    return description, parameter_descriptions


def _first_paragraph(docstring: str) -> str | None:
    lines: list[str] = []
    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped or _is_section_header(stripped):
            break
        lines.append(stripped)
    summary = " ".join(lines)
    return summary or None


def _google_parameter_descriptions(docstring: str) -> Mapping[str, str]:
    """Extract the first Google-style argument section into flat descriptions.

    Continuation lines are joined with spaces. Parsing stops at the next
    recognized top-level section so return or exception prose is never attached
    to the final parameter.
    """

    lines = docstring.splitlines()
    section_index = _find_argument_section(lines)
    if section_index is None:
        return {}

    descriptions: dict[str, str] = {}
    current_name: str | None = None
    current_parts: list[str] = []
    for line in lines[section_index + 1 :]:
        stripped = line.strip()
        if stripped and _is_section_header(stripped) and not line[:1].isspace():
            break

        match = _ARGUMENT_PATTERN.match(line)
        if match is not None:
            _store_description(descriptions, current_name, current_parts)
            current_name = match.group(1)
            initial_description = match.group(2).strip()
            current_parts = [initial_description] if initial_description else []
            continue

        if current_name is not None and stripped:
            current_parts.append(stripped)

    _store_description(descriptions, current_name, current_parts)
    return descriptions


def _find_argument_section(lines: list[str]) -> int | None:
    """Return the first recognized argument-section header."""

    for index, line in enumerate(lines):
        section_name = _section_name(line.strip())
        if section_name in {"args", "arguments", "parameters"}:
            return index
    return None


def _store_description(
    descriptions: dict[str, str],
    name: str | None,
    parts: list[str],
) -> None:
    """Commit one accumulated parameter description when it is non-empty."""

    description = " ".join(parts)
    if name is not None and description:
        descriptions[name] = description


def _is_section_header(value: str) -> bool:
    return _section_name(value) in _SECTION_NAMES


def _section_name(value: str) -> str:
    return value[:-1].strip().lower() if value.endswith(":") else ""


def _humanize_name(tool_name: str) -> str:
    words = " ".join(tool_name.strip("_").replace("_", " ").split())
    if not words:
        words = tool_name
    return words[:1].upper() + words[1:]


__all__: list[str] = []
