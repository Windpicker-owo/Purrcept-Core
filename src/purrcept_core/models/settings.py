"""Describe common immutable generation controls for provider adapters.

The core validates and normalizes shared settings but does not promise that
every provider supports each value. Backends decide how to translate supported
controls and how to report unsupported combinations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import cast


class ToolChoice(StrEnum):
    """How a model may choose from the request's tools."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Optional generation controls shared by model providers."""

    temperature: float | None = field(default=None, kw_only=True)
    max_output_tokens: int | None = field(default=None, kw_only=True)
    stop_sequences: tuple[str, ...] = field(default=(), kw_only=True)
    tool_choice: ToolChoice = field(default=ToolChoice.AUTO, kw_only=True)
    parallel_tool_calls: bool | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        temperature = _normalize_temperature(self.temperature)
        _validate_max_output_tokens(self.max_output_tokens)
        stop_sequences = _normalize_stop_sequences(self.stop_sequences)
        try:
            tool_choice = ToolChoice(self.tool_choice)
        except ValueError as error:
            raise ValueError(f"Unsupported tool choice: {self.tool_choice!r}.") from error
        _validate_optional_bool(
            self.parallel_tool_calls,
            field_name="parallel_tool_calls",
        )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "stop_sequences", stop_sequences)
        object.__setattr__(self, "tool_choice", tool_choice)


def _normalize_temperature(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("temperature must be a number.")
    if not isfinite(value):
        raise ValueError("temperature must be finite.")
    if value < 0:
        raise ValueError("temperature must be greater than or equal to zero.")
    return float(value)


def _validate_max_output_tokens(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_output_tokens must be an int.")
    if value <= 0:
        raise ValueError("max_output_tokens must be greater than zero.")


def _normalize_stop_sequences(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("stop_sequences must be an iterable.")
    items = tuple(cast(Iterable[object], value))
    if not all(isinstance(item, str) for item in items):
        raise TypeError("stop_sequences must contain only strings.")
    sequences = cast(tuple[str, ...], items)
    if any(not sequence for sequence in sequences):
        raise ValueError("stop_sequences must not contain empty strings.")
    return sequences


def _validate_optional_bool(value: object, *, field_name: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool or None.")


__all__ = ["ModelSettings", "ToolChoice"]
