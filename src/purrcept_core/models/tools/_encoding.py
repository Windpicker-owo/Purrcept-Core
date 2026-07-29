"""Convert supported Python tool results into provider-neutral model content.

``FunctionTool`` delegates successful return values here after invocation.
Existing ``ToolResult`` and content blocks pass through directly; strings and
``None`` use compact native representations; dataclasses and JSON-compatible
containers are recursively normalized and serialized deterministically.

Unsupported values, non-string mapping keys, non-finite numbers, and reference
cycles are translated to ``ToolResultEncodingError`` at the public tool
boundary. This module does not serialize arbitrary objects through ``repr`` or
expose Pydantic model types.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from math import isfinite
from typing import Any, cast

from ..content import ContentBlock, TextBlock
from .errors import ToolResultEncodingError
from .values import ToolResult


def encode_tool_result(value: object, *, tool_name: str) -> ToolResult:
    """Encode one supported Python return value without exposing Pydantic."""

    if isinstance(value, ToolResult):
        return value
    if isinstance(value, ContentBlock):
        return ToolResult(content=(value,))
    if isinstance(value, str):
        return ToolResult(content=(TextBlock(value),))
    if value is None:
        return ToolResult()

    try:
        json_value = _to_json_value(value, active_container_ids=set())
        text = json.dumps(
            json_value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise ToolResultEncodingError(tool_name, value) from None
    return ToolResult(content=(TextBlock(text),))


def _to_json_value(value: object, *, active_container_ids: set[int]) -> object:
    """Normalize one supported value while tracking recursion-path ownership."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _convert_dataclass(value, active_container_ids=active_container_ids)
    if isinstance(value, Mapping):
        return _convert_mapping(
            cast(Mapping[object, object], value),
            active_container_ids=active_container_ids,
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return _convert_sequence(
            cast(Sequence[object], value),
            active_container_ids=active_container_ids,
        )
    raise TypeError(f"Unsupported JSON value {type(value).__name__}.")


def _convert_dataclass(
    value: object,
    *,
    active_container_ids: set[int],
) -> dict[str, object]:
    """Convert declared dataclass fields without traversing incidental attributes."""

    container_id = _enter_container(value, active_container_ids)
    try:
        return {
            field.name: _to_json_value(
                getattr(value, field.name),
                active_container_ids=active_container_ids,
            )
            for field in fields(cast(Any, value))
        }
    finally:
        active_container_ids.remove(container_id)


def _convert_mapping(
    value: Mapping[object, object],
    *,
    active_container_ids: set[int],
) -> dict[str, object]:
    """Convert a mapping while enforcing JSON's string-key constraint."""

    container_id = _enter_container(value, active_container_ids)
    try:
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            converted[key] = _to_json_value(
                item,
                active_container_ids=active_container_ids,
            )
        return converted
    finally:
        active_container_ids.remove(container_id)


def _convert_sequence(
    value: Sequence[object],
    *,
    active_container_ids: set[int],
) -> list[object]:
    """Convert a non-string sequence while preserving element order."""

    container_id = _enter_container(value, active_container_ids)
    try:
        return [_to_json_value(item, active_container_ids=active_container_ids) for item in value]
    finally:
        active_container_ids.remove(container_id)


def _enter_container(value: object, active_container_ids: set[int]) -> int:
    """Mark a container active and reject only cycles on the current path.

    Callers remove the returned identifier in ``finally``. Reusing the same
    acyclic container in separate branches is therefore supported.
    """

    container_id = id(value)
    if container_id in active_container_ids:
        raise ValueError("JSON values must not contain reference cycles.")
    active_container_ids.add(container_id)
    return container_id


def error_tool_result(error: Exception) -> ToolResult:
    """Represent an expected invocation error as model-readable text."""

    return ToolResult(content=(TextBlock(str(error)),), is_error=True)


__all__: list[str] = []
