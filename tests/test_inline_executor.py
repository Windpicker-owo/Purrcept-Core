"""Verify inline effects support sync and async execution without error wrapping."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.effect import Effect
from purrcept_core.errors import UnsupportedEffectError
from purrcept_core.executor import InlineExecutor


def context() -> ExecutionContext[dict[str, int]]:
    return ExecutionContext("run", "effect", 0, {"offset": 4})


@dataclass(frozen=True, slots=True)
class SyncAdd(Effect[int]):
    value: int

    def execute(self, execution_context: ExecutionContext[dict[str, int]]) -> int:
        return self.value + execution_context.host["offset"]


@dataclass(frozen=True, slots=True)
class AsyncAdd(Effect[int]):
    value: int

    async def execute(
        self,
        execution_context: ExecutionContext[dict[str, int]],
    ) -> int:
        await asyncio.sleep(0)
        return self.value + execution_context.host["offset"]


class DeclarativeEffect(Effect[str]):
    pass


class NonCallableExecute(Effect[None]):
    execute = "not callable"


async def test_executes_a_synchronous_inline_effect() -> None:
    assert await InlineExecutor().execute(SyncAdd(3), context()) == 7


async def test_executes_an_asynchronous_inline_effect() -> None:
    assert await InlineExecutor().execute(AsyncAdd(3), context()) == 7


@pytest.mark.parametrize("effect", [DeclarativeEffect(), NonCallableExecute()])
async def test_rejects_an_effect_without_a_callable_execute(
    effect: Effect[object],
) -> None:
    with pytest.raises(UnsupportedEffectError) as caught:
        await InlineExecutor().execute(effect, context())

    assert caught.value.effect is effect
    assert type(effect).__name__ in str(caught.value)


async def test_does_not_wrap_inline_exceptions() -> None:
    expected = LookupError("missing")

    class FailingEffect(Effect[None]):
        def execute(self, execution_context: ExecutionContext[object]) -> None:
            del execution_context
            raise expected

    with pytest.raises(LookupError) as caught:
        await InlineExecutor().execute(FailingEffect(), context())

    assert caught.value is expected


async def test_does_not_swallow_cancellation() -> None:
    class CancelledEffect(Effect[None]):
        async def execute(
            self,
            execution_context: ExecutionContext[object],
        ) -> None:
            del execution_context
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await InlineExecutor().execute(CancelledEffect(), context())
