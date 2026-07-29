"""Verify generator flows preserve typed results and ordinary Python control flow."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from purrcept_core.effect import Effect
from purrcept_core.flow import AgentFlow, perform


@dataclass(frozen=True, slots=True)
class ReadNumber(Effect[int]):
    name: str


@dataclass(frozen=True, slots=True)
class FormatNumber(Effect[str]):
    value: int


def test_perform_yields_effect_and_returns_sent_value() -> None:
    effect = ReadNumber("answer")
    operation = perform(effect)

    assert next(operation) is effect

    with pytest.raises(StopIteration) as returned:
        operation.send(42)

    assert returned.value.value == 42


def test_flow_can_receive_multiple_result_types() -> None:
    def flow() -> AgentFlow[str]:
        number = yield from perform(ReadNumber("answer"))
        text = yield from perform(FormatNumber(number))
        return text

    agent = flow()
    assert next(agent) == ReadNumber("answer")
    assert agent.send(42) == FormatNumber(42)

    with pytest.raises(StopIteration) as returned:
        agent.send("forty-two")

    assert returned.value.value == "forty-two"


def test_flow_supports_direct_yield() -> None:
    def flow() -> AgentFlow[int]:
        result = yield ReadNumber("direct")
        return result

    agent = flow()
    assert next(agent) == ReadNumber("direct")

    with pytest.raises(StopIteration) as returned:
        agent.send(7)

    assert returned.value.value == 7


def test_perform_preserves_python_exception_handling() -> None:
    def flow() -> AgentFlow[str]:
        try:
            yield from perform(ReadNumber("missing"))
        except LookupError:
            return "fallback"
        return "unexpected"

    agent = flow()
    assert next(agent) == ReadNumber("missing")

    with pytest.raises(StopIteration) as returned:
        agent.throw(LookupError("not found"))

    assert returned.value.value == "fallback"


def test_flow_supports_condition_loops_and_finally() -> None:
    finalized: list[bool] = []

    def flow() -> AgentFlow[int]:
        total = 0
        try:
            for index in range(3):
                value = yield from perform(ReadNumber(str(index)))
                if value > 0:
                    total += value
            return total
        finally:
            finalized.append(True)

    agent = flow()
    assert next(agent) == ReadNumber("0")
    assert agent.send(-1) == ReadNumber("1")
    assert agent.send(2) == ReadNumber("2")

    with pytest.raises(StopIteration) as returned:
        agent.send(3)

    assert returned.value.value == 5
    assert finalized == [True]
