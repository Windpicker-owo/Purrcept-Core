"""Verify explicit effect dispatch follows MRO order, isolation, and fallback rules."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from purrcept_core.context import ExecutionContext
from purrcept_core.effect import Effect
from purrcept_core.errors import UnsupportedEffectError
from purrcept_core.executor import DispatchExecutor, InlineExecutor


def context() -> ExecutionContext[str]:
    return ExecutionContext("run", "effect", 0, "host")


@dataclass(frozen=True, slots=True)
class ParentEffect(Effect[str]):
    value: str


class ChildEffect(ParentEffect):
    pass


class GrandchildEffect(ChildEffect):
    pass


async def test_dispatches_to_a_synchronous_handler() -> None:
    def handle(effect: ParentEffect, execution_context: ExecutionContext[str]) -> str:
        return f"{effect.value}:{execution_context.host}"

    executor = DispatchExecutor({ParentEffect: handle})

    assert await executor.execute(ParentEffect("value"), context()) == "value:host"


async def test_dispatches_to_an_asynchronous_handler() -> None:
    async def handle(
        effect: ParentEffect,
        execution_context: ExecutionContext[str],
    ) -> str:
        await asyncio.sleep(0)
        return f"{effect.value}:{execution_context.host}"

    executor = DispatchExecutor({ParentEffect: handle})

    assert await executor.execute(ParentEffect("value"), context()) == "value:host"


async def test_uses_the_nearest_handler_in_the_effect_mro() -> None:
    calls: list[str] = []

    def handle_parent(effect: ParentEffect, _: ExecutionContext[Any]) -> str:
        calls.append(f"parent:{effect.value}")
        return "parent"

    def handle_child(effect: ChildEffect, _: ExecutionContext[Any]) -> str:
        calls.append(f"child:{effect.value}")
        return "child"

    executor = DispatchExecutor(
        {
            ParentEffect: handle_parent,
            ChildEffect: handle_child,
        }
    )

    assert await executor.execute(GrandchildEffect("value"), context()) == "child"
    assert calls == ["child:value"]


async def test_uses_a_parent_handler_for_a_subclass() -> None:
    executor = DispatchExecutor({ParentEffect: lambda effect, _: effect.value})

    assert await executor.execute(GrandchildEffect("inherited"), context()) == "inherited"


async def test_uses_fallback_when_no_handler_matches() -> None:
    @dataclass(frozen=True, slots=True)
    class InlineEffect(Effect[str]):
        value: str

        def execute(self, execution_context: ExecutionContext[str]) -> str:
            return f"{self.value}:{execution_context.host}"

    executor = DispatchExecutor({}, fallback=InlineExecutor())

    assert await executor.execute(InlineEffect("fallback"), context()) == "fallback:host"


async def test_raises_when_no_handler_or_fallback_supports_the_effect() -> None:
    effect = ParentEffect("unsupported")

    with pytest.raises(UnsupportedEffectError) as caught:
        await DispatchExecutor().execute(effect, context())

    assert caught.value.effect is effect


async def test_handler_mapping_is_copied_at_construction() -> None:
    handlers: dict[type[Effect[Any]], Any] = {ParentEffect: lambda _effect, _context: "original"}
    executor = DispatchExecutor(handlers)
    handlers[ParentEffect] = lambda _effect, _context: "mutated"

    assert await executor.execute(ParentEffect("value"), context()) == "original"


async def test_executor_instances_do_not_share_handlers() -> None:
    first = DispatchExecutor({ParentEffect: lambda _effect, _context: "first"})
    second = DispatchExecutor({ParentEffect: lambda _effect, _context: "second"})

    assert await first.execute(ParentEffect("value"), context()) == "first"
    assert await second.execute(ParentEffect("value"), context()) == "second"


async def test_handler_cancellation_is_not_wrapped() -> None:
    async def cancel(
        effect: ParentEffect,
        execution_context: ExecutionContext[str],
    ) -> str:
        del effect, execution_context
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await DispatchExecutor({ParentEffect: cancel}).execute(
            ParentEffect("value"),
            context(),
        )
