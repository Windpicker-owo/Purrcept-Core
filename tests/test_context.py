"""Verify execution contexts snapshot metadata and forward progress synchronously."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from purrcept_core.context import ExecutionContext


def test_context_preserves_identifiers_and_host() -> None:
    host = object()

    context = ExecutionContext(
        run_id="run-1",
        effect_id="effect-2",
        step_index=3,
        host=host,
    )

    assert context.run_id == "run-1"
    assert context.effect_id == "effect-2"
    assert context.step_index == 3
    assert context.host is host
    assert context.metadata == {}


def test_metadata_is_a_read_only_snapshot() -> None:
    original = {"request": "initial"}
    context = ExecutionContext("run", "effect", 0, None, original)

    original["request"] = "changed"

    assert isinstance(context.metadata, MappingProxyType)
    assert context.metadata == {"request": "initial"}
    with pytest.raises(TypeError):
        context.metadata["new"] = "value"  # type: ignore[index]


def test_progress_calls_the_injected_callback_synchronously() -> None:
    payloads: list[object] = []
    context = ExecutionContext(
        "run",
        "effect",
        0,
        None,
        _progress_callback=payloads.append,
    )

    payload = {"completed": 4}
    result = context.progress(payload)

    assert result is None
    assert payloads == [payload]


def test_progress_without_callback_is_a_no_op() -> None:
    context = ExecutionContext("run", "effect", 0, None)

    assert context.progress("ignored") is None


def test_progress_does_not_hide_callback_errors() -> None:
    expected = RuntimeError("event sink failed")

    def fail(_: object) -> None:
        raise expected

    context = ExecutionContext(
        "run",
        "effect",
        0,
        None,
        _progress_callback=fail,
    )

    with pytest.raises(RuntimeError) as caught:
        context.progress("payload")

    assert caught.value is expected
