"""Verify tool argument validation, invocation policy, and result encoding.

The suite exercises synchronous, threaded, and awaitable calls; context
injection; model-readable argument failures; deterministic JSON conversion;
cycle detection; and callable exception policy.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, cast

import pytest

from purrcept_core import Effect, ExecutionContext, InlineExecutor
from purrcept_core.models import ContentBlock, TextBlock, ToolCallBlock
from purrcept_core.models.tools import (
    FunctionTool,
    InvokeTool,
    SyncToolPolicy,
    ToolArgumentsError,
    ToolContext,
    ToolErrorPolicy,
    ToolParameter,
    ToolResult,
    ToolResultEncodingError,
    UnknownToolError,
)


def _context() -> ExecutionContext[object]:
    return ExecutionContext("run", "effect", 0, object())


def _call(
    name: str,
    arguments: dict[str, object] | None = None,
) -> ToolCallBlock:
    return ToolCallBlock(
        "call-1",
        name,
        arguments={} if arguments is None else arguments,  # type: ignore[arg-type]
    )


def _text(result: ToolResult) -> str:
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextBlock)
    return block.text


async def _invoke_return_value(
    value: object,
    *,
    error_policy: ToolErrorPolicy = ToolErrorPolicy.PROPAGATE,
) -> ToolResult:
    def return_value() -> object:
        return value

    function_tool = FunctionTool(return_value, error_policy=error_policy)
    return await function_tool.invoke(_call("return_value"), _context())


async def test_invoke_tool_validates_converts_defaults_and_injects_context() -> None:
    observed: list[tuple[int, bool, ToolContext]] = []

    def calculate(
        value: int,
        enabled: bool = False,
        *,
        context: ToolContext,
    ) -> dict[str, object]:
        observed.append((value, enabled, context))
        context.progress("working")
        return {
            "call": context.call.id,
            "enabled": enabled,
            "run": context.run_id,
            "value": value + 1,
        }

    progress: list[object] = []
    execution = ExecutionContext(
        "run",
        "effect",
        0,
        object(),
        _progress_callback=progress.append,
    )
    function_tool = FunctionTool(calculate)
    effect = InvokeTool(
        function_tool,
        _call("calculate", {"value": "4", "enabled": "true"}),
    )

    result = await InlineExecutor().execute(effect, execution)

    assert isinstance(effect, Effect)
    assert result.is_error is False
    assert json.loads(_text(result)) == {
        "call": "call-1",
        "enabled": True,
        "run": "run",
        "value": 5,
    }
    assert observed[0][0:2] == (4, True)
    assert observed[0][2].execution is execution
    assert progress == ["working"]


async def test_invoke_tool_uses_validated_defaults() -> None:
    def defaulted(value: int = "4") -> int:  # type: ignore[assignment]
        return value + 1

    result = await InvokeTool(
        FunctionTool(defaulted),
        _call("defaulted"),
    ).execute(_context())

    assert _text(result) == "5"


@pytest.mark.parametrize(
    ("arguments", "expected_fragment"),
    [
        ({}, "value: Field required"),
        ({"value": 1, "extra": 2}, "extra: Extra inputs are not permitted"),
        ({"value": 0}, "greater than or equal to 1"),
        ({"value": "1"}, "valid integer"),
        ({"value": [1, "bad"]}, "valid integer"),
    ],
)
async def test_invoke_tool_returns_model_argument_errors(
    arguments: dict[str, object],
    expected_fragment: str,
) -> None:
    function_tool = FunctionTool(
        _strict_integer,
        name="constrained",
    )

    result = await InvokeTool(
        function_tool,
        _call("constrained", arguments),
    ).execute(_context())

    assert result.is_error is True
    assert expected_fragment in _text(result)


def _strict_integer(
    value: Annotated[int, ToolParameter(strict=True, ge=1)],
) -> int:
    return value


async def test_direct_invoke_exposes_structured_argument_errors() -> None:
    def nested(values: list[int]) -> list[int]:
        return values

    function_tool = FunctionTool(nested)

    with pytest.raises(ToolArgumentsError) as caught:
        await function_tool.invoke(
            _call("nested", {"values": [1, "bad"]}),
            _context(),
        )

    assert caught.value.tool_name == "nested"
    assert any("values.1" in issue for issue in caught.value.issues)
    assert all("pydantic" not in issue.lower() for issue in caught.value.issues)


async def test_unknown_or_mismatched_tools_are_returned_to_the_model() -> None:
    def available() -> None:
        return None

    unknown = await InvokeTool(None, _call("missing")).execute(_context())
    mismatched = await InvokeTool(
        FunctionTool(available),
        _call("different"),
    ).execute(_context())

    assert unknown.is_error is True
    assert _text(unknown) == "Unknown tool 'missing'."
    assert mismatched.is_error is True
    assert _text(mismatched) == "Unknown tool 'different'."


async def test_function_tool_rejects_a_mismatched_direct_call() -> None:
    def available() -> None:
        return None

    with pytest.raises(UnknownToolError, match="different"):
        await FunctionTool(available).invoke(_call("different"), _context())


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: InvokeTool(cast(FunctionTool, object()), _call("lookup")),
            "tool must be a FunctionTool or None",
        ),
        (
            lambda: InvokeTool(None, cast(ToolCallBlock, object())),
            "call must be a ToolCallBlock",
        ),
    ],
)
def test_invoke_tool_validates_its_boundary(factory: object, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()  # type: ignore[operator]


async def test_function_tool_invoke_validates_its_boundary() -> None:
    def available() -> None:
        return None

    function_tool = FunctionTool(available)
    with pytest.raises(TypeError, match="call must be a ToolCallBlock"):
        await function_tool.invoke(cast(ToolCallBlock, object()), _context())
    with pytest.raises(TypeError, match="execution must be an ExecutionContext"):
        await function_tool.invoke(
            _call("available"),
            cast(ExecutionContext[object], object()),
        )


async def test_async_and_awaitable_returning_tools_are_awaited() -> None:
    async def asynchronous(value: int) -> int:
        await asyncio.sleep(0)
        return value + 1

    async def delayed() -> str:
        await asyncio.sleep(0)
        return "resolved"

    def returns_awaitable() -> object:
        return delayed()

    async_result = await FunctionTool(
        asynchronous,
        sync_policy=SyncToolPolicy.THREAD,
    ).invoke(_call("asynchronous", {"value": 1}), _context())
    awaitable_result = await FunctionTool(returns_awaitable).invoke(
        _call("returns_awaitable"),
        _context(),
    )

    assert _text(async_result) == "2"
    assert _text(awaitable_result) == "resolved"


async def test_thread_policy_runs_synchronous_tools_off_the_event_loop_thread() -> None:
    caller_thread = threading.get_ident()

    def identify_thread() -> int:
        return threading.get_ident()

    result = await FunctionTool(
        identify_thread,
        sync_policy="thread",
    ).invoke(_call("identify_thread"), _context())

    assert int(_text(result)) != caller_thread


async def test_tool_exceptions_propagate_by_default() -> None:
    expected = LookupError("database unavailable")

    def fail() -> None:
        raise expected

    with pytest.raises(LookupError) as caught:
        await InvokeTool(FunctionTool(fail), _call("fail")).execute(_context())

    assert caught.value is expected


async def test_explicit_error_policy_returns_tool_exceptions_to_the_model() -> None:
    def fail() -> None:
        raise LookupError("database unavailable")

    result = await InvokeTool(
        FunctionTool(fail, error_policy="return_to_model"),
        _call("fail"),
    ).execute(_context())

    assert result.is_error is True
    assert _text(result) == "database unavailable"


async def test_tool_raised_public_argument_error_is_not_mistaken_for_model_input() -> None:
    expected = ToolArgumentsError("fail", ("raised by Python tool",))

    def fail() -> None:
        raise expected

    with pytest.raises(ToolArgumentsError) as caught:
        await InvokeTool(FunctionTool(fail), _call("fail")).execute(_context())

    assert caught.value is expected


async def test_cancellation_always_propagates_even_with_return_policy() -> None:
    expected = asyncio.CancelledError("stop")

    async def cancel() -> None:
        raise expected

    with pytest.raises(asyncio.CancelledError) as caught:
        await FunctionTool(
            cancel,
            error_policy=ToolErrorPolicy.RETURN_TO_MODEL,
        ).invoke(_call("cancel"), _context())

    assert caught.value is expected


async def test_supported_return_types_encode_to_provider_neutral_content() -> None:
    existing = ToolResult(content=(TextBlock("existing"),), is_error=True)
    block = TextBlock("block")

    assert await _invoke_return_value(existing) is existing
    assert (await _invoke_return_value(block)).content == (block,)
    assert _text(await _invoke_return_value("plain 🐈")) == "plain 🐈"
    assert await _invoke_return_value(None) == ToolResult()
    assert _text(await _invoke_return_value(True)) == "true"
    assert _text(await _invoke_return_value(12)) == "12"
    assert _text(await _invoke_return_value(1.25)) == "1.25"


@dataclass(slots=True)
class Payload:
    name: str
    values: tuple[int, ...]
    metadata: dict[str, object]


async def test_json_and_dataclass_results_are_compact_sorted_and_unicode_safe() -> None:
    shared = [1, 2]
    value = Payload(
        "猫",
        (3, 4),
        {
            "z": MappingProxyType({"nested": shared}),
            "a": shared,
        },
    )

    result = await _invoke_return_value(value)

    assert _text(result) == (
        '{"metadata":{"a":[1,2],"z":{"nested":[1,2]}},"name":"猫","values":[3,4]}'
    )


@pytest.mark.parametrize(
    "value",
    [
        object(),
        b"bytes",
        bytearray(b"bytes"),
        {1, 2},
        float("inf"),
        {"bad": float("nan")},
        {1: "non-string key"},
    ],
)
async def test_unsupported_results_raise_encoding_error(value: object) -> None:
    with pytest.raises(ToolResultEncodingError) as caught:
        await _invoke_return_value(
            value,
            error_policy=ToolErrorPolicy.RETURN_TO_MODEL,
        )

    assert caught.value.tool_name == "return_value"
    assert caught.value.value_type is type(value)


@dataclass(slots=True)
class CyclicPayload:
    child: object | None = None


@pytest.mark.parametrize("kind", ["mapping", "sequence", "dataclass"])
async def test_cyclic_results_raise_encoding_error(kind: str) -> None:
    if kind == "mapping":
        value: object = {}
        cast(dict[str, object], value)["self"] = value
    elif kind == "sequence":
        value = []
        cast(list[object], value).append(value)
    else:
        payload = CyclicPayload()
        payload.child = payload
        value = payload

    with pytest.raises(ToolResultEncodingError):
        await _invoke_return_value(value)


def test_content_block_marker_is_not_treated_as_a_dataclass_payload() -> None:
    assert isinstance(TextBlock("text"), ContentBlock)
