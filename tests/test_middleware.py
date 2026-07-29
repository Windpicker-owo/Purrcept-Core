"""Verify middleware nesting, timeout, retry selection, delay, and error exclusions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.effect import Effect
from purrcept_core.errors import EventDispatchError
from purrcept_core.events import RunStarted
from purrcept_core.middleware import (
    FunctionMiddleware,
    MiddlewareExecutor,
    NextExecutor,
    RetryMiddleware,
    TimeoutMiddleware,
)


class SampleEffect(Effect[str]):
    pass


def context() -> ExecutionContext[None]:
    return ExecutionContext("run", "effect", 0, None)


class FunctionExecutor:
    def __init__(
        self,
        function: Any,
    ) -> None:
        self.function = function

    async def execute(
        self,
        effect: Effect[Any],
        execution_context: ExecutionContext[Any],
    ) -> Any:
        result = self.function(effect, execution_context)
        if isinstance(result, Awaitable):
            return await result
        return result


async def test_middlewares_are_nested_with_the_first_outermost() -> None:
    trace: list[str] = []

    def middleware(name: str) -> FunctionMiddleware:
        async def invoke(
            effect: Effect[Any],
            execution_context: ExecutionContext[Any],
            call_next: NextExecutor,
        ) -> str:
            trace.append(f"{name}:before")
            result = await call_next(effect, execution_context)
            trace.append(f"{name}:after")
            return f"{result}:{name}"

        return FunctionMiddleware(invoke)

    async def base(_: Effect[Any], __: ExecutionContext[Any]) -> str:
        trace.append("base")
        return "result"

    executor = MiddlewareExecutor(
        FunctionExecutor(base),
        [middleware("A"), middleware("B"), middleware("C")],
    )

    assert await executor.execute(SampleEffect(), context()) == "result:C:B:A"
    assert trace == [
        "A:before",
        "B:before",
        "C:before",
        "base",
        "C:after",
        "B:after",
        "A:after",
    ]


async def test_exceptions_travel_out_through_the_reverse_order() -> None:
    trace: list[str] = []
    expected = RuntimeError("failed")

    def middleware(name: str) -> FunctionMiddleware:
        async def invoke(
            effect: Effect[Any],
            execution_context: ExecutionContext[Any],
            call_next: NextExecutor,
        ) -> Any:
            trace.append(f"{name}:before")
            try:
                return await call_next(effect, execution_context)
            except RuntimeError:
                trace.append(f"{name}:error")
                raise

        return FunctionMiddleware(invoke)

    def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        trace.append("base")
        raise expected

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [middleware("outer"), middleware("inner")],
    )

    with pytest.raises(RuntimeError) as caught:
        await executor.execute(SampleEffect(), context())

    assert caught.value is expected
    assert trace == [
        "outer:before",
        "inner:before",
        "base",
        "inner:error",
        "outer:error",
    ]


async def test_function_middleware_accepts_a_synchronous_function() -> None:
    def replace(
        _: Effect[Any],
        __: ExecutionContext[Any],
        ___: NextExecutor,
    ) -> str:
        return "replacement"

    executor = MiddlewareExecutor(
        FunctionExecutor(lambda _effect, _context: "base"),
        [FunctionMiddleware(replace)],
    )

    assert await executor.execute(SampleEffect(), context()) == "replacement"


async def test_middleware_sequence_is_snapshotted() -> None:
    async def replace(
        _: Effect[Any],
        __: ExecutionContext[Any],
        ___: NextExecutor,
    ) -> str:
        return "replacement"

    middlewares = [FunctionMiddleware(replace)]
    executor = MiddlewareExecutor(
        FunctionExecutor(lambda _effect, _context: "base"),
        middlewares,
    )
    middlewares.clear()

    assert await executor.execute(SampleEffect(), context()) == "replacement"


async def test_timeout_cancels_the_wrapped_await() -> None:
    finalized = asyncio.Event()

    async def wait_forever(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    executor = MiddlewareExecutor(
        FunctionExecutor(wait_forever),
        [TimeoutMiddleware(0.01)],
    )

    with pytest.raises(TimeoutError):
        await executor.execute(SampleEffect(), context())

    assert finalized.is_set()


async def test_external_cancellation_is_not_converted_to_timeout() -> None:
    entered = asyncio.Event()

    async def wait_forever(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        entered.set()
        await asyncio.Event().wait()

    executor = MiddlewareExecutor(
        FunctionExecutor(wait_forever),
        [TimeoutMiddleware(60)],
    )
    task = asyncio.create_task(executor.execute(SampleEffect(), context()))
    await entered.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_retry_retries_only_a_configured_exception() -> None:
    attempts = 0

    def flaky(_: Effect[Any], __: ExecutionContext[Any]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LookupError("temporary")
        return "ok"

    executor = MiddlewareExecutor(
        FunctionExecutor(flaky),
        [RetryMiddleware(max_attempts=3, retry_on=LookupError)],
    )

    assert await executor.execute(SampleEffect(), context()) == "ok"
    assert attempts == 3


async def test_retry_does_not_retry_an_unconfigured_exception() -> None:
    attempts = 0

    def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("not selected")

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [RetryMiddleware(max_attempts=5, retry_on=LookupError)],
    )

    with pytest.raises(ValueError):
        await executor.execute(SampleEffect(), context())

    assert attempts == 1


async def test_retry_without_a_policy_retries_nothing() -> None:
    attempts = 0

    def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [RetryMiddleware(max_attempts=5)],
    )

    with pytest.raises(RuntimeError):
        await executor.execute(SampleEffect(), context())

    assert attempts == 1


async def test_retry_predicate_can_select_errors() -> None:
    attempts = 0

    def flaky(_: Effect[Any], __: ExecutionContext[Any]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry me")
        return "ok"

    executor = MiddlewareExecutor(
        FunctionExecutor(flaky),
        [
            RetryMiddleware(
                max_attempts=2,
                predicate=lambda error: str(error) == "retry me",
            )
        ],
    )

    assert await executor.execute(SampleEffect(), context()) == "ok"
    assert attempts == 2


async def test_retry_attempt_count_and_delay_are_deterministic() -> None:
    attempts = 0
    delay_calls: list[tuple[int, Exception]] = []
    sleeps: list[float] = []

    def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise LookupError(f"attempt {attempts}")

    def delay(attempt: int, error: Exception) -> float:
        delay_calls.append((attempt, error))
        return attempt / 10

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [
            RetryMiddleware(
                max_attempts=3,
                retry_on=LookupError,
                delay=delay,
                sleep=sleep,
            )
        ],
    )

    with pytest.raises(LookupError, match="attempt 3"):
        await executor.execute(SampleEffect(), context())

    assert attempts == 3
    assert [(attempt, str(error)) for attempt, error in delay_calls] == [
        (1, "attempt 1"),
        (2, "attempt 2"),
    ]
    assert sleeps == [0.1, 0.2]


async def test_retry_never_catches_cancellation() -> None:
    attempts = 0

    async def cancel(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    executor = MiddlewareExecutor(
        FunctionExecutor(cancel),
        [
            RetryMiddleware(
                max_attempts=5,
                predicate=lambda _error: True,
            )
        ],
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(SampleEffect(), context())

    assert attempts == 1


async def test_retry_never_catches_event_dispatch_errors() -> None:
    attempts = 0
    dispatch_error = EventDispatchError(
        RunStarted(
            run_id="run",
            occurred_at=datetime.now(UTC),
        )
    )

    async def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise dispatch_error

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [RetryMiddleware(max_attempts=5, predicate=lambda _error: True)],
    )

    with pytest.raises(EventDispatchError) as caught:
        await executor.execute(SampleEffect(), context())

    assert caught.value is dispatch_error
    assert attempts == 1


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_retry_rejects_invalid_attempt_counts(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryMiddleware(max_attempts=max_attempts)


def test_retry_rejects_base_exception_types() -> None:
    with pytest.raises(TypeError, match="Exception subclasses"):
        RetryMiddleware(retry_on=BaseException)  # type: ignore[arg-type]


def test_retry_rejects_negative_fixed_delay() -> None:
    with pytest.raises(ValueError, match="delay"):
        RetryMiddleware(delay=-0.1)


async def test_retry_rejects_a_negative_strategy_delay() -> None:
    expected = LookupError("retryable")

    def fail(_: Effect[Any], __: ExecutionContext[Any]) -> None:
        raise expected

    executor = MiddlewareExecutor(
        FunctionExecutor(fail),
        [
            RetryMiddleware(
                retry_on=LookupError,
                delay=lambda _attempt, _error: -0.1,
            )
        ],
    )

    with pytest.raises(ValueError, match="retry delay") as caught:
        await executor.execute(SampleEffect(), context())

    assert caught.value.__cause__ is expected
