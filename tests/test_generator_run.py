"""Verify generator-run state transitions, terminal states, and cleanup semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from purrcept_core.effect import Effect
from purrcept_core.errors import InvalidRunStateError
from purrcept_core.flow import AgentFlow, perform
from purrcept_core.run import (
    AgentRun,
    GeneratorRun,
    Returned,
    Yielded,
    as_agent_run,
)


@dataclass(frozen=True, slots=True)
class ReadValue(Effect[int]):
    key: str


def one_step() -> AgentFlow[int]:
    value = yield from perform(ReadValue("first"))
    return value + 1


def test_start_returns_first_effect() -> None:
    run = GeneratorRun(one_step())

    assert run.start() == Yielded(ReadValue("first"))


def test_send_can_return_next_effect_then_final_value() -> None:
    def flow() -> AgentFlow[int]:
        first = yield from perform(ReadValue("first"))
        second = yield from perform(ReadValue("second"))
        return first + second

    run = GeneratorRun(flow())
    assert run.start() == Yielded(ReadValue("first"))
    assert run.send(2) == Yielded(ReadValue("second"))
    assert run.send(3) == Returned(5)


def test_empty_flow_returns_immediately() -> None:
    def flow() -> AgentFlow[str]:
        if False:
            yield ReadValue("never")
        return "done"

    assert GeneratorRun(flow()).start() == Returned("done")


def test_throw_can_be_caught_by_flow() -> None:
    def flow() -> AgentFlow[int]:
        try:
            yield from perform(ReadValue("primary"))
        except LookupError:
            value = yield from perform(ReadValue("fallback"))
            return value
        return -1

    run = GeneratorRun(flow())
    assert run.start() == Yielded(ReadValue("primary"))
    assert run.throw(LookupError("missing")) == Yielded(ReadValue("fallback"))
    assert run.send(9) == Returned(9)


def test_uncaught_error_propagates_and_closes_run() -> None:
    error = LookupError("missing")
    run = GeneratorRun(one_step())
    run.start()

    with pytest.raises(LookupError) as raised:
        run.throw(error)

    assert raised.value is error
    with pytest.raises(InvalidRunStateError, match="CLOSED"):
        run.send(1)


def test_error_raised_by_flow_after_send_propagates_and_closes_run() -> None:
    def flow() -> AgentFlow[None]:
        yield from perform(ReadValue("value"))
        raise RuntimeError("flow failed")

    run = GeneratorRun(flow())
    run.start()

    with pytest.raises(RuntimeError, match="flow failed"):
        run.send(1)
    with pytest.raises(InvalidRunStateError, match="CLOSED"):
        run.throw(ValueError("too late"))


def test_error_before_first_yield_propagates_and_closes_run() -> None:
    def flow() -> AgentFlow[None]:
        if False:
            yield ReadValue("never")
        raise RuntimeError("start failed")

    run = GeneratorRun(flow())

    with pytest.raises(RuntimeError, match="start failed"):
        run.start()
    with pytest.raises(InvalidRunStateError, match="CLOSED"):
        run.start()


def test_repeated_start_reports_running_state() -> None:
    run = GeneratorRun(one_step())
    run.start()

    with pytest.raises(InvalidRunStateError, match=r"RUNNING.*CREATED"):
        run.start()


@pytest.mark.parametrize("operation", ["send", "throw"])
def test_advancing_before_start_reports_created_state(operation: str) -> None:
    run = GeneratorRun(one_step())

    with pytest.raises(InvalidRunStateError, match=r"CREATED.*RUNNING"):
        if operation == "send":
            run.send(1)
        else:
            run.throw(ValueError("early"))


@pytest.mark.parametrize("operation", ["start", "send", "throw"])
def test_completed_run_cannot_be_advanced(operation: str) -> None:
    run = GeneratorRun(one_step())
    run.start()
    assert run.send(1) == Returned(2)

    with pytest.raises(InvalidRunStateError, match="RETURNED"):
        if operation == "start":
            run.start()
        elif operation == "send":
            run.send(2)
        else:
            run.throw(ValueError("late"))


def test_close_executes_finally_and_prevents_advancement() -> None:
    finalized: list[bool] = []

    def flow() -> AgentFlow[None]:
        try:
            yield from perform(ReadValue("value"))
        finally:
            finalized.append(True)

    run = GeneratorRun(flow())
    run.start()
    run.close()
    run.close()

    assert finalized == [True]
    with pytest.raises(InvalidRunStateError, match="CLOSED"):
        run.send(1)


def test_close_created_run_is_idempotent() -> None:
    run = GeneratorRun(one_step())
    run.close()
    run.close()

    with pytest.raises(InvalidRunStateError, match="CLOSED"):
        run.start()


def test_close_after_return_preserves_returned_state() -> None:
    run = GeneratorRun(one_step())
    run.start()
    run.send(1)
    run.close()

    with pytest.raises(InvalidRunStateError, match="RETURNED"):
        run.send(2)


def test_as_agent_run_wraps_generator() -> None:
    run = as_agent_run(one_step())

    assert isinstance(run, GeneratorRun)
    assert run.start() == Yielded(ReadValue("first"))


def test_as_agent_run_preserves_protocol_implementation() -> None:
    original = GeneratorRun(one_step())
    run = as_agent_run(cast(AgentRun[int], original))

    assert run is original
    assert isinstance(run, AgentRun)


def test_as_agent_run_rejects_other_values() -> None:
    with pytest.raises(TypeError, match=r"AgentRun or synchronous generator.*int"):
        as_agent_run(cast(AgentRun[object], 1))
