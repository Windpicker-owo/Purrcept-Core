"""Compose ordered wrappers around the effect-execution boundary.

The module owns local executor composition plus timeout and retry policies. A
:class:`MiddlewareExecutor` snapshots middleware order and builds one callable
chain around a caller-owned base executor; it does not schedule agent runs or
classify domain-specific effect failures.

Calls enter middleware in registration order and unwind in reverse order.
Cancellation and event-dispatch failures always bypass retries because
re-executing an effect after either outcome could duplicate side effects or
hide an infrastructure failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from typing import Any, Protocol, TypeAlias, cast

from .context import ExecutionContext
from .effect import Effect
from .errors import EventDispatchError
from .executor import EffectExecutor

NextExecutor: TypeAlias = Callable[
    [Effect[Any], ExecutionContext[Any]],
    Awaitable[Any],
]
"""Bound continuation that invokes the next middleware or base executor."""

MiddlewareFunction: TypeAlias = Callable[
    [Effect[Any], ExecutionContext[Any], NextExecutor],
    Any,
]
"""Plain synchronous or asynchronous function adapted by ``FunctionMiddleware``."""

RetryPredicate: TypeAlias = Callable[[Exception], bool]
"""Selector returning whether one ordinary effect failure may be retried."""

RetryDelay: TypeAlias = Callable[[int, Exception], float]
"""Strategy returning delay seconds for a one-based failed attempt number."""

SleepFunction: TypeAlias = Callable[[float], Awaitable[None]]
"""Awaitable delay hook, injectable for deterministic retry tests."""


class EffectMiddleware(Protocol):
    """Extension contract for one wrapper around the next effect executor.

    A runtime constructs middleware and passes it to
    :class:`MiddlewareExecutor`. The executor calls each instance once per
    effect with a reusable ``call_next`` continuation. Middleware may inspect
    or replace results and ordinary exceptions, but it must await or return the
    downstream call according to its own policy. Instances own mutable state
    and any concurrency guarantees when shared across runs.

    Built-in implementations are :class:`FunctionMiddleware`,
    :class:`TimeoutMiddleware`, and :class:`RetryMiddleware`; selection is
    explicit through constructor order.
    """

    async def __call__(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
        call_next: NextExecutor,
    ) -> Any:
        """Execute middleware logic around ``call_next``."""


def _wrap_middleware(
    middleware: EffectMiddleware,
    call_next: NextExecutor,
) -> NextExecutor:
    """Bind one middleware instance to the already-built inner chain."""

    async def wrapped(
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        """Invoke the bound middleware with its fixed downstream continuation."""

        return await middleware(effect, context, call_next)

    return wrapped


class MiddlewareExecutor:
    """Wrap an executor in a local, ordered middleware chain.

    For ``[A, B, C]``, calls enter ``A``, then ``B``, then ``C``, then the
    base executor. Results and exceptions travel through the reverse order.
    """

    __slots__ = ("_call",)

    def __init__(
        self,
        base: EffectExecutor[Any],
        middlewares: Sequence[EffectMiddleware] = (),
    ) -> None:
        async def call_base(
            effect: Effect[Any],
            context: ExecutionContext[Any],
        ) -> Any:
            """Terminate the middleware chain at the caller-owned executor."""

            return await base.execute(effect, context)

        # Ordering: wrapping in reverse makes the first configured middleware
        # the outermost call while preserving a straight-through continuation.
        call_next: NextExecutor = call_base
        for middleware in reversed(tuple(middlewares)):
            call_next = _wrap_middleware(middleware, call_next)
        self._call = call_next

    async def execute(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
    ) -> Any:
        """Execute an effect through the configured middleware chain."""

        return await self._call(effect, context)


class FunctionMiddleware:
    """Adapt a plain function into :class:`EffectMiddleware`.

    Both synchronous and asynchronous functions are accepted. The latter is the
    usual form; accepting synchronous functions keeps small instrumentation
    wrappers concise.
    """

    __slots__ = ("_function",)

    def __init__(self, function: MiddlewareFunction) -> None:
        self._function = function

    async def __call__(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
        call_next: NextExecutor,
    ) -> Any:
        """Invoke the wrapped middleware function."""

        result = self._function(effect, context, call_next)
        if isawaitable(result):
            return await result
        return result


class TimeoutMiddleware:
    """Limit the total duration of the wrapped effect call."""

    __slots__ = ("timeout",)

    def __init__(self, timeout: float | None) -> None:
        self.timeout = timeout

    async def __call__(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
        call_next: NextExecutor,
    ) -> Any:
        """Raise :class:`TimeoutError` if the call exceeds ``timeout``."""

        async with asyncio.timeout(self.timeout):
            return await call_next(effect, context)


class RetryMiddleware:
    """Retry explicitly selected ordinary exceptions.

    ``max_attempts`` includes the initial call. ``retry_on`` and ``predicate``
    are combined with OR semantics. With neither configured, no exception is
    retried. A delay strategy receives the one-based number of the failed
    attempt and the exception.
    """

    __slots__ = (
        "_delay",
        "_max_attempts",
        "_predicate",
        "_retry_on",
        "_sleep",
    )

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        retry_on: type[Exception] | tuple[type[Exception], ...] = (),
        predicate: RetryPredicate | None = None,
        delay: float | RetryDelay = 0.0,
        sleep: SleepFunction = asyncio.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        retry_types = _normalize_retry_types(retry_on)
        if not callable(delay) and delay < 0:
            raise ValueError("delay must not be negative")

        self._max_attempts = max_attempts
        self._retry_on = retry_types
        self._predicate = predicate
        self._delay = delay
        self._sleep = sleep

    async def __call__(
        self,
        effect: Effect[Any],
        context: ExecutionContext[Any],
        call_next: NextExecutor,
    ) -> Any:
        """Call the next executor until it succeeds or retries are exhausted."""

        attempt = 1
        while True:
            try:
                return await call_next(effect, context)
            except asyncio.CancelledError:
                # Never let a broad user-supplied retry policy swallow task
                # cancellation, even if CancelledError changes hierarchy later.
                raise
            except EventDispatchError:
                # Observer failures are infrastructure failures, not retryable
                # effect failures. Retrying here could duplicate side effects.
                raise
            except Exception as error:
                if attempt >= self._max_attempts or not self._should_retry(error):
                    raise

                delay = self._delay
                seconds = delay(attempt, error) if callable(delay) else delay
                if seconds < 0:
                    raise ValueError("retry delay must not be negative") from error
                if seconds:
                    await self._sleep(seconds)
                attempt += 1

    def _should_retry(self, error: Exception) -> bool:
        """Apply the configured type and predicate selectors with OR semantics."""

        if isinstance(error, self._retry_on):
            return True
        predicate = self._predicate
        return predicate(error) if predicate is not None else False


def _normalize_retry_types(value: object) -> tuple[type[Exception], ...]:
    """Validate retry classes without accepting cancellation-like base errors."""

    if isinstance(value, type):
        candidates: tuple[object, ...] = (value,)
    elif isinstance(value, tuple):
        candidates = cast(tuple[object, ...], value)
    else:
        raise TypeError("retry_on must contain Exception subclasses")

    if any(
        not isinstance(candidate, type) or not issubclass(candidate, Exception)
        for candidate in candidates
    ):
        raise TypeError("retry_on must contain Exception subclasses")
    return cast(tuple[type[Exception], ...], candidates)


__all__ = [
    "EffectMiddleware",
    "FunctionMiddleware",
    "MiddlewareExecutor",
    "MiddlewareFunction",
    "NextExecutor",
    "RetryDelay",
    "RetryMiddleware",
    "RetryPredicate",
    "SleepFunction",
    "TimeoutMiddleware",
]
