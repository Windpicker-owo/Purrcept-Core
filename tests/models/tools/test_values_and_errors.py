"""Verify Pydantic-free tool values, runtime context, policies, and public errors."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core import ExecutionContext, PurrceptError
from purrcept_core.models import TextBlock, ToolCallBlock, ToolResultBlock
from purrcept_core.models.tools import (
    SyncToolPolicy,
    ToolArgumentsError,
    ToolContext,
    ToolDefinitionError,
    ToolErrorPolicy,
    ToolParameter,
    ToolResult,
    ToolResultEncodingError,
    UnknownToolError,
)


def _context(progress: list[object] | None = None) -> ExecutionContext[object]:
    return ExecutionContext(
        "run-1",
        "effect-1",
        3,
        {"provider": "test"},
        {"trace": True},
        _progress_callback=None if progress is None else progress.append,
    )


def _call() -> ToolCallBlock:
    return ToolCallBlock("call-1", "lookup")


def test_tool_parameter_is_a_frozen_slotted_metadata_value() -> None:
    parameter = ToolParameter(
        description="A constrained value.",
        ge=1,
        le=10,
        gt=0,
        lt=11,
        min_length=1,
        max_length=20,
        pattern=r"^\w+$",
        strict=True,
    )

    assert parameter.description == "A constrained value."
    assert parameter.ge == 1
    assert parameter.le == 10
    assert parameter.gt == 0
    assert parameter.lt == 11
    assert parameter.min_length == 1
    assert parameter.max_length == 20
    assert parameter.pattern == r"^\w+$"
    assert parameter.strict is True
    assert not hasattr(parameter, "__dict__")
    with pytest.raises(FrozenInstanceError):
        parameter.ge = 2  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["ge", "le", "gt", "lt"])
@pytest.mark.parametrize("bad_value", [True, "1", object()])
def test_tool_parameter_rejects_non_numeric_constraints(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be a number"):
        ToolParameter(**{field_name: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["ge", "le", "gt", "lt"])
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_tool_parameter_rejects_non_finite_constraints(
    field_name: str,
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        ToolParameter(**{field_name: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["min_length", "max_length"])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1"])
def test_tool_parameter_rejects_non_integer_lengths(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be an int"):
        ToolParameter(**{field_name: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["min_length", "max_length"])
def test_tool_parameter_rejects_negative_lengths(field_name: str) -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        ToolParameter(**{field_name: -1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ToolParameter(description=cast(str, 1)),
            TypeError,
            "description must be a string",
        ),
        (
            lambda: ToolParameter(description=""),
            ValueError,
            "description must not be empty",
        ),
        (
            lambda: ToolParameter(min_length=2, max_length=1),
            ValueError,
            "min_length must not be greater",
        ),
        (
            lambda: ToolParameter(pattern=cast(str, 1)),
            TypeError,
            "pattern must be a string",
        ),
        (
            lambda: ToolParameter(strict=cast(bool, 1)),
            TypeError,
            "strict must be a bool",
        ),
    ],
)
def test_tool_parameter_validates_other_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_tool_context_exposes_execution_and_call_without_copying() -> None:
    progress: list[object] = []
    execution = _context(progress)
    call = _call()
    context = ToolContext(execution, call)

    context.progress({"stage": "working"})

    assert context.execution is execution
    assert context.call is call
    assert context.run_id == "run-1"
    assert context.effect_id == "effect-1"
    assert context.step_index == 3
    assert context.host == {"provider": "test"}
    assert context.metadata == {"trace": True}
    assert progress == [{"stage": "working"}]
    assert not hasattr(context, "__dict__")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: ToolContext(cast(ExecutionContext[object], object()), _call()),
            "execution must be an ExecutionContext",
        ),
        (
            lambda: ToolContext(_context(), cast(ToolCallBlock, object())),
            "call must be a ToolCallBlock",
        ),
    ],
)
def test_tool_context_validates_its_boundary(factory: object, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()  # type: ignore[operator]


def test_tool_result_freezes_content_and_binds_to_a_call() -> None:
    source = [TextBlock("done")]
    result = ToolResult(content=source, is_error=True)  # type: ignore[arg-type]
    source.append(TextBlock("changed"))

    block = result.to_block("call-1")

    assert result.content == (TextBlock("done"),)
    assert result.is_error is True
    assert block == ToolResultBlock(
        "call-1",
        content=(TextBlock("done"),),
        is_error=True,
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.is_error = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: ToolResult(content=cast(tuple[TextBlock, ...], 1)),
            "content must be an iterable",
        ),
        (
            lambda: ToolResult(content=cast(tuple[TextBlock, ...], (object(),))),
            "only ContentBlock",
        ),
        (
            lambda: ToolResult(is_error=cast(bool, "yes")),
            "is_error must be a bool",
        ),
    ],
)
def test_tool_result_validates_its_boundary(factory: object, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        factory()  # type: ignore[operator]


def test_public_tool_errors_are_purrcept_errors_with_structured_details() -> None:
    unknown = UnknownToolError("missing")
    arguments = ToolArgumentsError("lookup", ("x: field required", "extra: forbidden"))
    definition = ToolDefinitionError("bad definition", tool_name="lookup")
    value = object()
    encoding = ToolResultEncodingError("lookup", value)

    assert isinstance(unknown, PurrceptError)
    assert unknown.tool_name == "missing"
    assert str(unknown) == "Unknown tool 'missing'."
    assert arguments.tool_name == "lookup"
    assert arguments.issues == ("x: field required", "extra: forbidden")
    assert "x: field required; extra: forbidden" in str(arguments)
    assert definition.tool_name == "lookup"
    assert str(definition) == "bad definition"
    assert encoding.tool_name == "lookup"
    assert encoding.value_type is object
    assert "builtins.object" in str(encoding)


def test_tool_arguments_error_accepts_a_generator_and_has_an_empty_fallback() -> None:
    generated = ToolArgumentsError("lookup", (item for item in ("first", "second")))
    empty = ToolArgumentsError("lookup", ())

    assert generated.issues == ("first", "second")
    assert "arguments are invalid" in str(empty)


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: UnknownToolError(cast(str, 1)),
            TypeError,
            "tool_name must be a string",
        ),
        (
            lambda: UnknownToolError(""),
            ValueError,
            "tool_name must not be empty",
        ),
        (
            lambda: ToolArgumentsError("tool", cast(object, "issue")),
            TypeError,
            "issues must be an iterable",
        ),
        (
            lambda: ToolArgumentsError("tool", cast(object, 1)),
            TypeError,
            "issues must be an iterable",
        ),
        (
            lambda: ToolArgumentsError("tool", cast(object, ("ok", 1))),
            TypeError,
            "issues must contain only strings",
        ),
        (
            lambda: ToolDefinitionError(cast(str, 1)),
            TypeError,
            "message must be a string",
        ),
        (
            lambda: ToolDefinitionError(""),
            ValueError,
            "message must not be empty",
        ),
        (
            lambda: ToolDefinitionError("bad", tool_name=""),
            ValueError,
            "tool_name must not be empty",
        ),
    ],
)
def test_tool_errors_validate_their_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_tool_policies_have_stable_string_values() -> None:
    assert str(ToolErrorPolicy.PROPAGATE) == "propagate"
    assert str(ToolErrorPolicy.RETURN_TO_MODEL) == "return_to_model"
    assert str(SyncToolPolicy.INLINE) == "inline"
    assert str(SyncToolPolicy.THREAD) == "thread"
