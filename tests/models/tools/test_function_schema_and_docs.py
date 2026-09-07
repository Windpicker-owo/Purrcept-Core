"""Verify callable inspection produces stable tool documentation and JSON Schema.

The scenarios cover supported signature shapes, context injection, annotation
metadata, docstring fallbacks, compatibility diagnostics, immutable definitions,
and rejection of ambiguous callables.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Annotated, NotRequired, TypedDict, cast

import pytest

from purrcept_core.models import ToolSpec
from purrcept_core.models.tools import (
    FunctionTool,
    SyncToolPolicy,
    ToolContext,
    ToolDefinitionError,
    ToolErrorPolicy,
    ToolParameter,
    tool,
)
from purrcept_core.models.tools._schema import _definition_failure_detail


class UnsupportedParameter:
    pass


class SearchInput(TypedDict):
    query: str
    limit: NotRequired[int]


def test_function_tool_supports_typed_dict_schema_or_guides_python_311() -> None:
    def search(options: SearchInput) -> str:
        return options["query"]

    if sys.version_info < (3, 12):
        with pytest.raises(
            ToolDefinitionError,
            match="import TypedDict from typing_extensions",
        ):
            FunctionTool(search)
        return

    function_tool = FunctionTool(search)
    schema = function_tool.spec.parameters
    definitions = schema["$defs"]
    properties = schema["properties"]

    assert isinstance(definitions, dict | type(schema))
    assert isinstance(properties, dict | type(schema))
    assert properties["options"]["$ref"] == "#/$defs/SearchInput"
    assert definitions["SearchInput"]["type"] == "object"
    assert definitions["SearchInput"]["required"] == ("query",)
    assert definitions["SearchInput"]["properties"]["limit"]["type"] == "integer"


def test_typed_dict_compatibility_guidance_is_added_only_when_relevant() -> None:
    ordinary = ValueError("ordinary schema error")
    pydantic_guidance = ValueError(
        "Please use typing_extensions.TypedDict instead of typing.TypedDict."
    )

    assert _definition_failure_detail(ordinary) == "ordinary schema error"
    assert "import TypedDict from typing_extensions" in _definition_failure_detail(
        pydantic_guidance
    )


def test_function_tool_builds_a_forbid_extra_json_schema_from_type_hints() -> None:
    def constrained(
        query: Annotated[
            str,
            ToolParameter(
                description="Explicit query description.",
                min_length=2,
                max_length=20,
                pattern=r"^[a-z]+$",
                strict=True,
            ),
        ],
        count: Annotated[
            int,
            "preserved metadata",
            ToolParameter(ge=1, le=10, gt=0, lt=11, strict=True),
        ] = 3,
        *,
        context: ToolContext,
    ) -> str:
        del count, context
        return query

    function_tool = FunctionTool(constrained, description="Run a constrained query.")
    schema = function_tool.spec.parameters
    properties = schema["properties"]

    assert isinstance(function_tool.spec, ToolSpec)
    assert function_tool.name == "constrained"
    assert function_tool.description == "Run a constrained query."
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ("query",)
    assert isinstance(properties, dict | type(schema))
    assert properties["query"] == {
        "description": "Explicit query description.",
        "maxLength": 20,
        "minLength": 2,
        "pattern": r"^[a-z]+$",
        "title": "Query",
        "type": "string",
    }
    assert properties["count"] == {
        "default": 3,
        "exclusiveMaximum": 11,
        "exclusiveMinimum": 0,
        "maximum": 10,
        "minimum": 1,
        "title": "Count",
        "type": "integer",
    }
    assert "context" not in properties


def test_description_precedence_and_google_style_parameter_docs() -> None:
    def documented(
        place: str,
        days: Annotated[int, ToolParameter(description="Explicit day count.")],
        units: str = "metric",
    ) -> str:
        """Look up a forecast
        for one place.

        Args:
            place (str): City or region to inspect.
                Unicode names are accepted.
            days:
                Number of forecast days from the docstring.
            units (str):
            ignored: This is not an exposed parameter.

        Returns:
            A compact forecast.
        """
        del days, units
        return place

    inferred = FunctionTool(documented)
    overridden = FunctionTool(documented, description="Explicit tool summary.")
    properties = inferred.spec.parameters["properties"]

    assert inferred.description == "Look up a forecast for one place."
    assert overridden.description == "Explicit tool summary."
    assert isinstance(properties, dict | type(inferred.spec.parameters))
    assert properties["place"]["description"] == (
        "City or region to inspect. Unicode names are accepted."
    )
    assert properties["days"]["description"] == "Explicit day count."
    assert "description" not in properties["units"]


@pytest.mark.parametrize("section_name", ["Arguments", "Parameters"])
def test_google_style_argument_section_aliases_are_supported(section_name: str) -> None:
    def aliased(value: str) -> str:
        return value

    aliased.__doc__ = f"""Resolve a value.

{section_name}:
    value: Value documented under an alias.
"""

    function_tool = FunctionTool(aliased)
    properties = function_tool.spec.parameters["properties"]

    assert isinstance(properties, dict | type(function_tool.spec.parameters))
    assert properties["value"]["description"] == "Value documented under an alias."


def test_tool_description_falls_back_to_a_humanized_function_name() -> None:
    def find_favorite_cat(limit: int = 1) -> int:
        return limit

    find_favorite_cat.__doc__ = None

    function_tool = FunctionTool(find_favorite_cat)
    only_underscores = FunctionTool(
        find_favorite_cat,
        name="___",
    )

    assert function_tool.description == "Find favorite cat"
    assert only_underscores.description == "___"


def test_docstring_starting_with_a_section_uses_the_name_fallback() -> None:
    def lookup(value: str) -> str:
        """Args:
        value: A lookup value.
        """
        return value

    function_tool = FunctionTool(lookup)
    properties = function_tool.spec.parameters["properties"]

    assert function_tool.description == "Lookup"
    assert isinstance(properties, dict | type(function_tool.spec.parameters))
    assert properties["value"]["description"] == "A lookup value."


def test_function_tool_and_decorator_capture_stable_configuration() -> None:
    @tool
    def direct(value: int) -> int:
        return value

    @tool(
        name="renamed",
        description="A renamed tool.",
        error_policy="return_to_model",
        sync_policy="thread",
    )
    def configured(value: int) -> int:
        return value

    assert isinstance(direct, FunctionTool)
    assert direct.name == "direct"
    assert configured.name == "renamed"
    assert configured.description == "A renamed tool."
    assert configured.error_policy is ToolErrorPolicy.RETURN_TO_MODEL
    assert configured.sync_policy is SyncToolPolicy.THREAD


def test_function_tool_hides_the_callable_from_repr_and_is_frozen_slotted() -> None:
    resource_marker = object()

    def closure(value: int) -> int:
        return value if resource_marker is not None else 0

    function_tool = FunctionTool(closure)
    rendered = repr(function_tool)

    assert "function=" not in rendered
    assert "closure at " not in rendered
    assert not hasattr(function_tool, "__dict__")
    with pytest.raises(FrozenInstanceError):
        function_tool.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("function", "match"),
    [
        (
            lambda value, /: value,
            "positional-only",
        ),
        (
            lambda *values: values,
            "variadic positional",
        ),
        (
            lambda **values: values,
            "variadic keyword",
        ),
    ],
)
def test_function_tool_rejects_unsupported_parameter_kinds(
    function: object,
    match: str,
) -> None:
    function.__annotations__ = {"value": int, "values": int}  # type: ignore[attr-defined]

    with pytest.raises(ToolDefinitionError, match=match):
        FunctionTool(function)  # type: ignore[arg-type]


def test_function_tool_rejects_unannotated_and_unresolvable_parameters() -> None:
    def unannotated(value):  # type: ignore[no-untyped-def]
        return value

    def unresolved(value: object) -> object:
        return value

    unresolved.__annotations__["value"] = "TypeThatDoesNotExist"

    with pytest.raises(ToolDefinitionError, match="must have a type annotation"):
        FunctionTool(unannotated)
    with pytest.raises(ToolDefinitionError, match="TypeThatDoesNotExist"):
        FunctionTool(unresolved)


def test_function_tool_rejects_duplicate_or_annotated_context_parameters() -> None:
    def duplicate(first: ToolContext, second: ToolContext) -> None:
        del first, second

    def decorated(context: Annotated[ToolContext, "metadata"]) -> None:
        del context

    with pytest.raises(ToolDefinitionError, match="more than one ToolContext"):
        FunctionTool(duplicate)
    with pytest.raises(ToolDefinitionError, match="annotate ToolContext exactly"):
        FunctionTool(decorated)


def test_function_tool_rejects_duplicate_tool_parameter_metadata() -> None:
    def duplicate(
        value: Annotated[
            int,
            ToolParameter(ge=1),
            ToolParameter(le=10),
        ],
    ) -> int:
        return value

    with pytest.raises(ToolDefinitionError, match="more than one ToolParameter"):
        FunctionTool(duplicate)


def test_function_tool_wraps_schema_and_signature_failures() -> None:
    def unsupported(value: UnsupportedParameter) -> None:
        del value

    class InvalidSignature:
        __signature__ = "not a signature"

        def __call__(self, value: int) -> int:
            return value

    with pytest.raises(ToolDefinitionError, match="Cannot define tool 'unsupported'"):
        FunctionTool(unsupported)
    with pytest.raises(ToolDefinitionError, match="Cannot define tool 'invalid'"):
        FunctionTool(InvalidSignature(), name="invalid")


def test_callable_objects_are_supported_when_given_an_explicit_name() -> None:
    class Multiplier:
        """Multiply a value."""

        def __init__(self, factor: int) -> None:
            self.factor = factor

        def __call__(self, value: int) -> int:
            return value * self.factor

    function_tool = FunctionTool(Multiplier(2), name="multiply")

    assert function_tool.name == "multiply"
    assert function_tool.spec.parameters["required"] == ("value",)
    properties = function_tool.spec.parameters["properties"]
    assert isinstance(properties, Mapping)
    assert properties["value"]["type"] == "integer"


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: FunctionTool(cast(object, 1)),  # type: ignore[arg-type]
            TypeError,
            "function must be callable",
        ),
        (
            lambda: FunctionTool(cast(object, object()), name="object"),  # type: ignore[arg-type]
            TypeError,
            "function must be callable",
        ),
        (
            lambda: FunctionTool(lambda value: value),
            ToolDefinitionError,
            "must have a type annotation",
        ),
        (
            lambda: FunctionTool(
                cast(object, type("CallableWithoutName", (), {"__call__": lambda self: None})())
            ),  # type: ignore[arg-type]
            ToolDefinitionError,
            "require an explicit name",
        ),
        (
            lambda: FunctionTool(lambda: None, name=cast(str, 1)),
            TypeError,
            "name must be a string",
        ),
        (
            lambda: FunctionTool(lambda: None, name=""),
            ValueError,
            "name must not be empty",
        ),
        (
            lambda: FunctionTool(lambda: None, description=""),
            ToolDefinitionError,
            "description must not be empty",
        ),
        (
            lambda: FunctionTool(lambda: None, description=cast(str, 1)),
            ToolDefinitionError,
            "description must be a string",
        ),
        (
            lambda: FunctionTool(
                lambda: None,
                error_policy=cast(ToolErrorPolicy, 1),
            ),
            TypeError,
            "error_policy must be",
        ),
        (
            lambda: FunctionTool(lambda: None, error_policy="unknown"),
            ValueError,
            "Unsupported tool error policy",
        ),
        (
            lambda: FunctionTool(
                lambda: None,
                sync_policy=cast(SyncToolPolicy, 1),
            ),
            TypeError,
            "sync_policy must be",
        ),
        (
            lambda: FunctionTool(lambda: None, sync_policy="unknown"),
            ValueError,
            "Unsupported sync tool policy",
        ),
    ],
)
def test_function_tool_validates_definition_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
