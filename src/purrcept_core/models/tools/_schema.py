"""Adapt Python callable signatures to JSON Schema and validated call arguments.

``FunctionTool`` constructs one :class:`ArgumentAdapter` per callable. The
adapter resolves annotations, separates an injected ``ToolContext`` parameter,
combines ``Annotated`` metadata with docstring descriptions, and creates a
private Pydantic model. Only the resulting JSON Schema and plain validated
argument dictionary cross back into the Pydantic-free public tool API.

Definition-time failures become ``ToolDefinitionError``. Invocation-time
validation failures become ``ToolInputValidationError`` with stable,
input-free issue strings; the original Pydantic error is intentionally not
exposed as part of the public contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import Parameter, Signature, isroutine, signature
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .._utils import JsonValue
from .errors import ToolDefinitionError, ToolInputValidationError
from .values import ToolContext, ToolParameter

_ToolCallable = Callable[..., object]


class ArgumentAdapter:
    """Private lifetime adapter shared by one immutable ``FunctionTool``.

    Construction performs all signature and schema work once. ``validate`` may
    then be called for many model tool calls; Pydantic model state is local to
    each call, while the adapter itself retains only immutable definition data.
    """

    __slots__ = ("_context_parameter", "_model", "_parameter_names", "_schema")

    def __init__(
        self,
        function: _ToolCallable,
        tool_name: str,
        parameter_descriptions: Mapping[str, str],
    ) -> None:
        try:
            resolved_signature = signature(function)
            type_hints = _resolve_type_hints(function)
            definitions, parameter_names, context_parameter = _build_definitions(
                resolved_signature,
                type_hints,
                tool_name=tool_name,
                parameter_descriptions=parameter_descriptions,
            )
            model_factory = cast(Callable[..., type[BaseModel]], create_model)
            model = model_factory(
                f"{tool_name}Arguments",
                __config__=ConfigDict(extra="forbid", validate_default=True),
                **definitions,
            )
            raw_schema = model.model_json_schema(mode="validation")
        except ToolDefinitionError:
            raise
        except Exception as error:
            raise ToolDefinitionError(
                f"Cannot define tool {tool_name!r}: {_definition_failure_detail(error)}",
                tool_name=tool_name,
            ) from None

        self._model: type[BaseModel] = model
        self._parameter_names = parameter_names
        self._context_parameter = context_parameter
        self._schema = cast(Mapping[str, JsonValue], raw_schema)

    @property
    def schema(self) -> Mapping[str, JsonValue]:
        """Return the model-facing validation schema generated at construction."""

        return self._schema

    def validate(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
        *,
        tool_name: str,
    ) -> dict[str, object]:
        """Validate JSON arguments and inject the runtime-only tool context."""

        try:
            validated = self._model.model_validate(dict(arguments))
        except ValidationError as error:
            raise ToolInputValidationError(
                tool_name,
                _format_validation_issues(error),
            ) from None

        values = {
            parameter_name: cast(object, getattr(validated, parameter_name))
            for parameter_name in self._parameter_names
        }
        if self._context_parameter is not None:
            values[self._context_parameter] = context
        return values


def _resolve_type_hints(function: _ToolCallable) -> Mapping[str, Any]:
    """Resolve annotations for a function or callable object's ``__call__``."""

    # Compatibility: Python 3.14 can return an empty mapping for callable
    # instances instead of raising TypeError. Resolve the method that owns
    # their parameter annotations explicitly, independently of that behavior.
    target = function if isroutine(function) else cast(_ToolCallable, function.__call__)
    return get_type_hints(target, include_extras=True)


def _definition_failure_detail(error: Exception) -> str:
    """Add an actionable Python 3.11 TypedDict compatibility explanation."""

    detail = str(error)
    if "typing_extensions.TypedDict" not in detail:
        return detail
    return (
        "TypedDict parameters on Python 3.11 must import TypedDict from "
        f"typing_extensions instead of typing. Original error: {detail}"
    )


def _build_definitions(
    resolved_signature: Signature,
    type_hints: Mapping[str, Any],
    *,
    tool_name: str,
    parameter_descriptions: Mapping[str, str],
) -> tuple[dict[str, tuple[Any, Any]], tuple[str, ...], str | None]:
    """Translate supported parameters into Pydantic fields and injection metadata.

    Positional-only and variadic parameters are rejected because model calls
    arrive as named JSON objects. Exactly one unwrapped ``ToolContext`` may be
    omitted from the public schema and injected after validation.
    """

    definitions: dict[str, tuple[Any, Any]] = {}
    parameter_names: list[str] = []
    context_parameter: str | None = None

    for parameter in resolved_signature.parameters.values():
        if parameter.kind in (
            Parameter.POSITIONAL_ONLY,
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            raise ToolDefinitionError(
                f"Tool {tool_name!r} parameter {parameter.name!r} uses unsupported "
                f"kind {parameter.kind.description!r}.",
                tool_name=tool_name,
            )
        if parameter.annotation is Parameter.empty:
            raise ToolDefinitionError(
                f"Tool {tool_name!r} parameter {parameter.name!r} must have a type annotation.",
                tool_name=tool_name,
            )

        annotation = type_hints.get(parameter.name)
        if annotation is None:
            raise ToolDefinitionError(
                f"Cannot resolve annotation for tool {tool_name!r} parameter {parameter.name!r}.",
                tool_name=tool_name,
            )
        if annotation is ToolContext:
            if context_parameter is not None:
                raise ToolDefinitionError(
                    f"Tool {tool_name!r} must not declare more than one ToolContext parameter.",
                    tool_name=tool_name,
                )
            context_parameter = parameter.name
            continue

        exposed_annotation, parameter_metadata = _split_parameter_metadata(
            annotation,
            tool_name=tool_name,
            parameter_name=parameter.name,
        )
        default = ... if parameter.default is Parameter.empty else parameter.default
        definitions[parameter.name] = (
            exposed_annotation,
            _create_field(
                default,
                parameter_metadata,
                fallback_description=parameter_descriptions.get(parameter.name),
            ),
        )
        parameter_names.append(parameter.name)

    return definitions, tuple(parameter_names), context_parameter


def _split_parameter_metadata(
    annotation: Any,
    *,
    tool_name: str,
    parameter_name: str,
) -> tuple[Any, ToolParameter | None]:
    """Separate core ``ToolParameter`` metadata while preserving other annotations."""

    if get_origin(annotation) is not Annotated:
        return annotation, None

    base, *metadata = get_args(annotation)
    tool_parameters = tuple(item for item in metadata if isinstance(item, ToolParameter))
    if len(tool_parameters) > 1:
        raise ToolDefinitionError(
            f"Tool {tool_name!r} parameter {parameter_name!r} has more than one "
            "ToolParameter annotation.",
            tool_name=tool_name,
        )
    if base is ToolContext:
        raise ToolDefinitionError(
            f"Tool {tool_name!r} parameter {parameter_name!r} must annotate "
            "ToolContext exactly, without Annotated metadata.",
            tool_name=tool_name,
        )

    remaining_metadata = tuple(item for item in metadata if not isinstance(item, ToolParameter))
    exposed_annotation: Any = base
    if remaining_metadata:
        exposed_annotation = Annotated[base, *remaining_metadata]
    parameter_metadata = tool_parameters[0] if tool_parameters else None
    return exposed_annotation, parameter_metadata


def _create_field(
    default: object,
    metadata: ToolParameter | None,
    *,
    fallback_description: str | None,
) -> Any:
    """Merge docstring fallback text with explicit ``ToolParameter`` constraints."""

    options: dict[str, Any] = {}
    if fallback_description is not None:
        options["description"] = fallback_description
    if metadata is None:
        return Field(default, **options)
    for name in (
        "description",
        "ge",
        "le",
        "gt",
        "lt",
        "min_length",
        "max_length",
        "pattern",
        "strict",
    ):
        value = getattr(metadata, name)
        if value is not None:
            options[name] = value
    return Field(default, **options)


def _format_validation_issues(error: ValidationError) -> tuple[str, ...]:
    """Project Pydantic details into stable model-readable location messages."""

    issues: list[str] = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in detail["loc"])
        message = detail["msg"]
        issues.append(f"{location}: {message}" if location else message)
    return tuple(issues)


__all__: list[str] = []
