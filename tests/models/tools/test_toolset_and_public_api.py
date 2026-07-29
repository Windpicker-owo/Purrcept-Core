"""Verify ordered tool registration, lookup, exports, and dependency boundaries."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import purrcept_core.models as models
import purrcept_core.models.tools as tools_api
from purrcept_core.models.tools import (
    FunctionTool,
    ToolDefinitionError,
    ToolSet,
    UnknownToolError,
    tool,
)


def lookup(query: str) -> str:
    """Look up one query."""

    return query


@tool(name="count")
def count_items(values: list[str]) -> int:
    return len(values)


def test_toolset_normalizes_tools_and_preserves_registration_order() -> None:
    source = [lookup, count_items]

    toolset = ToolSet(source)
    source.clear()

    assert len(toolset) == 2
    assert tuple(item.name for item in toolset) == ("lookup", "count")
    assert toolset.resolve("lookup").function is lookup
    assert toolset.resolve("count") is count_items
    assert toolset.get("missing") is None
    assert toolset.specs == (
        toolset.resolve("lookup").spec,
        count_items.spec,
    )
    assert tuple(toolset.tools) == ("lookup", "count")


def test_toolset_from_callables_is_a_discoverable_normalization_entrypoint() -> None:
    toolset = ToolSet.from_callables(item for item in (lookup,))

    assert len(toolset) == 1
    assert isinstance(toolset.resolve("lookup"), FunctionTool)


def test_toolset_is_frozen_slotted_and_exposes_an_immutable_mapping() -> None:
    toolset = ToolSet((lookup,))

    assert not hasattr(toolset, "__dict__")
    with pytest.raises(TypeError):
        toolset.tools["other"] = FunctionTool(lookup)
    with pytest.raises(FrozenInstanceError):
        toolset._specs = ()  # type: ignore[misc]


def test_toolset_rejects_duplicate_names() -> None:
    def another_lookup(query: str) -> str:
        return query

    another_lookup.__name__ = "lookup"

    with pytest.raises(ToolDefinitionError, match="registered more than once") as caught:
        ToolSet((lookup, another_lookup))

    assert caught.value.tool_name == "lookup"


@pytest.mark.parametrize("value", [1, "lookup", b"lookup"])
def test_toolset_rejects_non_iterable_or_string_construction_inputs(value: object) -> None:
    with pytest.raises(TypeError, match="tools must be an iterable"):
        ToolSet(cast(object, value))  # type: ignore[arg-type]


def test_toolset_rejects_non_callable_elements_clearly() -> None:
    with pytest.raises(TypeError, match="only FunctionTool or callable"):
        ToolSet((lookup, cast(object, 1)))  # type: ignore[arg-type]


def test_toolset_resolution_reports_unknown_and_invalid_names() -> None:
    toolset = ToolSet()

    with pytest.raises(UnknownToolError, match="missing"):
        toolset.resolve("missing")
    with pytest.raises(TypeError, match="name must be a string"):
        toolset.get(cast(str, 1))
    with pytest.raises(ValueError, match="tool_name must not be empty"):
        toolset.resolve("")


def test_tools_subpackage_has_the_complete_pydantic_free_public_surface() -> None:
    expected = {
        "FunctionTool",
        "InvokeTool",
        "SyncToolPolicy",
        "ToolArgumentsError",
        "ToolContext",
        "ToolDefinitionError",
        "ToolErrorPolicy",
        "ToolLike",
        "ToolParameter",
        "ToolResult",
        "ToolResultEncodingError",
        "ToolSet",
        "UnknownToolError",
        "tool",
    }

    assert set(tools_api.__all__) == expected
    for name in tools_api.__all__:
        value = getattr(tools_api, name)
        assert "pydantic" not in repr(value).lower()
        if inspect.isclass(value) or inspect.isfunction(value):
            assert "pydantic" not in str(inspect.signature(value)).lower()


def test_tools_are_aggregated_by_models_without_polluting_kernel_top_level() -> None:
    assert set(tools_api.__all__) <= set(models.__all__)
    assert models.FunctionTool is FunctionTool
    assert not hasattr(__import__("purrcept_core"), "FunctionTool")


def test_model_facing_tool_schema_contains_only_core_and_builtin_values() -> None:
    function_tool = FunctionTool(lookup)

    assert isinstance(function_tool.spec, models.ToolSpec)
    assert type(function_tool.spec.parameters).__module__ == "builtins"
    assert all(
        "pydantic" not in type(value).__module__ for value in function_tool.spec.parameters.values()
    )
