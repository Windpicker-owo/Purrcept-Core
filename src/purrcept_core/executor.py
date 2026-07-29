"""Define the runtime boundary that turns effect descriptions into values.

Agent flows yield :class:`~purrcept_core.effect.Effect` objects, and the driver
delegates each one to an :class:`EffectExecutor`. This module owns the executor
contract and two local selection strategies; it does not advance flows, emit
events, or translate executor exceptions.

``InlineExecutor`` selects an ``execute(context)`` method on the effect itself.
``DispatchExecutor`` selects the nearest registered class in the effect's
actual MRO and can delegate misses to another executor. Both accept synchronous
or awaitable handler results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import isawaitable
from typing import Any, Protocol, TypeAlias, TypeVar, cast

from .context import ExecutionContext
from .effect import Effect
from .errors import UnsupportedEffectError

HostT = TypeVar("HostT")

# ``Callable[..., Any]`` intentionally permits handlers to annotate their first
# parameter as a concrete Effect subtype. A heterogeneous handler mapping cannot
# express that relationship more precisely with Python's current type system.
EffectHandler: TypeAlias = Callable[..., Any]
"""Synchronous or awaitable handler selected for a concrete effect type."""


class EffectExecutor(Protocol[HostT]):
    """Extension contract for executing effects on behalf of an agent driver.

    A runtime creates an implementation and supplies it to
    :class:`~purrcept_core.driver.AgentDriver`. The driver may call one instance
    from multiple run tasks, so implementations own synchronization and
    resource lifetime. ``execute`` must either return the effect result or raise
    the original capability failure; the driver owns event reporting and
    delivery of ordinary exceptions back to agent code.

    Built-in implementations are :class:`InlineExecutor` and
    :class:`DispatchExecutor`. Middleware composition is provided by
    :class:`~purrcept_core.middleware.MiddlewareExecutor`, and selection is
    always explicit at driver construction.
    """

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[HostT],
    ) -> Any:
        """Execute ``effect`` in ``context``."""


async def _resolve_result(value: Any) -> Any:
    """Resolve an optionally-awaitable executor or handler result."""

    if isawaitable(value):
        return await value
    return value


class InlineExecutor:
    """Execute effects that provide their own ``execute(context)`` method.

    Missing or non-callable methods raise
    :class:`~purrcept_core.errors.UnsupportedEffectError`. Exceptions from a
    valid method propagate unchanged.
    """

    __slots__ = ()

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        """Call the effect's synchronous or asynchronous execute method."""

        execute = getattr(effect, "execute", None)
        if not callable(execute):
            raise UnsupportedEffectError(effect)
        return await _resolve_result(execute(context))


class DispatchExecutor:
    """Dispatch effects through an instance-local mapping of explicit handlers.

    Registration is snapshotted at construction. For each call, the executor
    walks the concrete effect type's MRO and invokes the first registered
    handler; an optional fallback executor receives misses. Handler and fallback
    exceptions propagate unchanged.
    """

    __slots__ = ("_fallback", "_handlers")

    def __init__(
        self,
        handlers: Mapping[type[Effect[Any]], EffectHandler] | None = None,
        fallback: EffectExecutor[Any] | None = None,
    ) -> None:
        # Copy the mapping so later caller mutations cannot implicitly register or
        # replace handlers on this executor.
        self._handlers = dict(handlers or {})
        self._fallback = fallback

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        """Execute the most specific handler in the effect type's actual MRO."""

        handler = self._find_handler(effect)
        if handler is not None:
            return await _resolve_result(handler(effect, context))

        fallback = self._fallback
        if fallback is not None:
            return await fallback.execute(effect, context)

        raise UnsupportedEffectError(effect)

    def _find_handler(self, effect: Effect[Any]) -> EffectHandler | None:
        """Return the nearest handler according to the effect's actual MRO."""

        for candidate in type(effect).__mro__:
            handler = self._handlers.get(cast(type[Effect[Any]], candidate))
            if handler is not None:
                return handler
        return None


__all__ = [
    "DispatchExecutor",
    "EffectExecutor",
    "EffectHandler",
    "InlineExecutor",
]
