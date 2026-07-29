"""Carry opaque provider continuation state across model requests.

Backends return :class:`ModelContinuation` snapshots and conversation
orchestration decides whether to send them on a later request according to
:class:`ContinuationPolicy`. The core recursively freezes the provider payload
but never interprets its keys or updates it in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ._utils import JsonValue, freeze_json_object, require_non_empty_string


class ContinuationPolicy(StrEnum):
    """Who manages model continuation state across requests."""

    CLIENT_MANAGED = "client_managed"
    PROVIDER_MANAGED = "provider_managed"
    AUTO = "auto"


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """An immutable snapshot whose payload semantics belong to one provider."""

    provider: str
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        require_non_empty_string(self.provider, field_name="provider")
        object.__setattr__(
            self,
            "data",
            freeze_json_object(self.data, field_name="data"),
        )


__all__ = ["ContinuationPolicy", "ModelContinuation"]
