"""Centralize immutable JSON snapshots and repeated model-boundary validation.

Public model value objects call these helpers during construction. The module
accepts JSON-compatible caller containers, rejects cycles and non-finite
numbers, and recursively replaces mutable mappings and sequences with
read-only mappings and tuples. It does not serialize values or accept
provider-specific Python objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias, cast

JsonPrimitive: TypeAlias = str | int | float | bool | None
"""Scalar values representable by the provider-neutral JSON boundary."""

JsonValue: TypeAlias = JsonPrimitive | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
"""Recursive JSON value accepted from mutable callers and frozen on entry."""

JsonObject: TypeAlias = Mapping[str, JsonValue]
"""Read-only object shape exposed by frozen model metadata."""


def empty_json_object() -> JsonObject:
    """Create a fresh empty JSON object for a dataclass default."""

    return {}


def freeze_json_object(value: Mapping[str, JsonValue], *, field_name: str) -> JsonObject:
    """Copy and recursively freeze a JSON object."""

    raw_value = _require_mapping(value, field_name=field_name)
    frozen = _freeze_json(raw_value, path=field_name, active_container_ids=set())
    return cast(JsonObject, frozen)


def require_non_empty_string(value: object, *, field_name: str) -> str:
    """Return a non-empty string or raise a field-specific error."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    return value


def validate_optional_non_empty_string(value: object, *, field_name: str) -> None:
    """Validate an optional non-empty string."""

    if value is not None:
        require_non_empty_string(value, field_name=field_name)


def _require_mapping(value: object, *, field_name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return cast(Mapping[object, object], value)


def _freeze_json(
    value: object,
    *,
    path: str,
    active_container_ids: set[int],
) -> JsonValue:
    """Recursively freeze one JSON value while detecting active-path cycles.

    Only container identifiers on the current recursion path are tracked. The
    same acyclic container may therefore appear in multiple branches and is
    copied independently, while a true self-reference is rejected.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers.")
        return value
    if isinstance(value, Mapping):
        container_id = id(cast(object, value))
        if container_id in active_container_ids:
            raise ValueError(f"{path} must not contain a reference cycle.")
        active_container_ids.add(container_id)
        source = cast(Mapping[object, object], value)
        frozen_mapping: dict[str, JsonValue] = {}
        try:
            for key, item in source.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} must contain only string object keys.")
                frozen_mapping[key] = _freeze_json(
                    item,
                    path=f"{path}.{key}",
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return MappingProxyType(frozen_mapping)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        container_id = id(cast(object, value))
        if container_id in active_container_ids:
            raise ValueError(f"{path} must not contain a reference cycle.")
        active_container_ids.add(container_id)
        source = cast(Sequence[object], value)
        try:
            return tuple(
                _freeze_json(
                    item,
                    path=f"{path}[{index}]",
                    active_container_ids=active_container_ids,
                )
                for index, item in enumerate(source)
            )
        finally:
            active_container_ids.remove(container_id)
    raise TypeError(f"{path} contains unsupported JSON value of type {type(value).__name__}.")


__all__ = ["JsonObject", "JsonPrimitive", "JsonValue"]
