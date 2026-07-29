"""Carry immutable run identity and runtime-owned services into one effect call.

The driver creates an :class:`ExecutionContext` for each yielded effect and
passes it to the selected executor. The context exposes the runtime host,
read-only run metadata, and an inline progress callback; it does not own effect
execution or survive as an active progress channel after the effect finishes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeAlias, TypeVar

HostT = TypeVar("HostT")

ProgressCallback: TypeAlias = Callable[[object], None]
"""Synchronous runtime callback receiving one progress payload inline."""


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class ExecutionContext(Generic[HostT]):
    """Per-effect information supplied by the runtime.

    ``metadata`` is copied into a read-only mapping when the context is created.
    The copy prevents later mutations of a caller-owned dictionary from changing
    the context observed by an effect. The driver also invalidates the injected
    progress callback after execution, so retaining a context does not grant a
    way to emit late lifecycle events.
    """

    run_id: str
    effect_id: str
    step_index: int
    host: HostT
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    _progress_callback: ProgressCallback | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        """Take an immutable snapshot of caller-provided metadata."""

        self.metadata = MappingProxyType(dict(self.metadata))

    def progress(self, payload: object) -> None:
        """Report incremental progress synchronously without advancing the flow.

        With no injected callback this is a no-op. Callback failures propagate
        immediately so the driver can treat observer failure as infrastructure
        failure rather than an effect result.
        """

        callback = self._progress_callback
        if callback is not None:
            callback(payload)


__all__ = ["ExecutionContext", "ProgressCallback"]
