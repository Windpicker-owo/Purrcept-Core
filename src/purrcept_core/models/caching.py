"""Describe provider-neutral prompt-cache intent without implementing a cache.

Applications attach :class:`PromptCachePolicy` to model requests, and provider
adapters translate supported hints to their native API. The values are frozen
and normalized at construction; cache storage, key derivation, and provider
capability negotiation remain outside the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

from ._utils import validate_optional_non_empty_string


class CacheMode(StrEnum):
    """The requested prompt caching behavior."""

    AUTO = "auto"
    DISABLED = "disabled"
    PREFER = "prefer"
    EXPLICIT = "explicit"


class PromptStability(StrEnum):
    """How a prompt segment is expected to evolve between requests."""

    STABLE = "stable"
    GROWING = "growing"
    VOLATILE = "volatile"


@dataclass(frozen=True, slots=True)
class PromptCachePolicy:
    """Immutable hints controlling provider prompt-cache use.

    ``strict`` lets an adapter reject a request when it cannot honor the policy.
    The optional key and TTL are transport hints only; the core neither creates
    cache entries nor promises provider support.
    """

    mode: CacheMode = field(default=CacheMode.AUTO, kw_only=True)
    key: str | None = field(default=None, kw_only=True)
    ttl: timedelta | None = field(default=None, kw_only=True)
    strict: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        try:
            mode = CacheMode(self.mode)
        except ValueError as error:
            raise ValueError(f"Unsupported prompt cache mode: {self.mode!r}.") from error
        validate_optional_non_empty_string(self.key, field_name="key")
        _validate_ttl(self.ttl)
        _validate_strict(self.strict)
        object.__setattr__(self, "mode", mode)


def _validate_ttl(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, timedelta):
        raise TypeError("ttl must be a timedelta or None.")
    if value <= timedelta(0):
        raise ValueError("ttl must be greater than zero.")


def _validate_strict(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("strict must be a bool.")


__all__ = ["CacheMode", "PromptCachePolicy", "PromptStability"]
