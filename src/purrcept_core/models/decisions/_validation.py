"""Validate immutable decision values without provider-specific limits.

Constructors use these helpers before handing snapshots to backends. Numerical
validation deliberately rejects booleans, even though Python treats them as
integers; choice values retain their distinct string and boolean identities.
"""

from collections.abc import Iterable
from math import isfinite
from typing import cast


def text(value: object, name: str) -> None:
    """Require meaningful text while preserving the caller's whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def optional_text(value: object, name: str) -> None:
    """Validate an optional description or diagnostic identifier."""
    if value is not None:
        text(value, name)


def choice_value(value: object) -> None:
    """Reject numeric choices rather than coercing them to bool or str."""
    if not isinstance(value, (str, bool)):
        raise TypeError("choice value must be a string or boolean.")


def number(value: object, name: str, *, upper: float | None = None) -> None:
    """Validate a finite nonnegative score or bounded probability."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its finite range.")


def snapshot(value: object, types: tuple[type, ...], name: str) -> tuple[object, ...]:
    """Copy an iterable and reject empty or heterogeneous boundary values."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of decision values.")
    items = tuple(cast(Iterable[object], value))
    if not items:
        raise ValueError(f"{name} must not be empty.")
    if not all(isinstance(item, types) for item in items):
        raise TypeError(f"{name} contains an unsupported value.")
    return items


def unique(values: Iterable[object], name: str) -> None:
    """Reject duplicate names or typed option identities."""
    items = tuple(values)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must be unique.")
