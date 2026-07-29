"""Expose typed Python callables as validated provider-neutral model tools.

``FunctionTool`` resolves a stable name and documentation, builds one private
argument adapter, invokes the callable with validated keyword arguments, and
encodes its return value into model content. The ``tool`` helper supports both
direct construction and decorator syntax. Public values remain Pydantic-free;
schema implementation details stay in the private adapter module.

Synchronous callables execute inline by default or in ``asyncio.to_thread`` by
explicit policy. Awaitable results are always awaited. Argument errors are
handled by ``InvokeTool``; callable exceptions either propagate or become error
results according to the tool's configured policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from inspect import isawaitable, iscoroutinefunction
from typing import Any, TypeAlias, cast, overload

from ...context import ExecutionContext
from ..content import ToolCallBlock
from ..requests import ToolSpec
from ._documentation import resolve_documentation
from ._encoding import encode_tool_result, error_tool_result
from ._schema import ArgumentAdapter
from .errors import ToolDefinitionError, UnknownToolError
from .values import SyncToolPolicy, ToolContext, ToolErrorPolicy, ToolResult

ToolCallable: TypeAlias = Callable[..., object]
"""Callable shape accepted for model-tool registration.

Definition-time signature inspection imposes the precise supported parameter
contract; this broad alias also accommodates callable objects and both
synchronous and asynchronous return paths.
"""


@dataclass(frozen=True, slots=True, init=False)
class FunctionTool:
    """Immutable validated model-facing view of one Python callable.

    Applications create tools directly or with :func:`tool`, then register them
    in a ``ToolSet``. Construction resolves all schema and documentation
    metadata so invocation performs only argument validation, optional context
    injection, callable execution, and result encoding. The callable remains
    caller-owned and may be shared across conversations; it owns mutable state,
    resource lifetime, and thread safety.
    """

    function: ToolCallable = field(repr=False)
    name: str
    description: str | None
    error_policy: ToolErrorPolicy
    sync_policy: SyncToolPolicy
    spec: ToolSpec
    _adapter: ArgumentAdapter = field(repr=False, compare=False)

    def __init__(
        self,
        function: ToolCallable,
        *,
        name: str | None = None,
        description: str | None = None,
        error_policy: ToolErrorPolicy | str = ToolErrorPolicy.PROPAGATE,
        sync_policy: SyncToolPolicy | str = SyncToolPolicy.INLINE,
    ) -> None:
        if not callable(cast(object, function)):
            raise TypeError("function must be callable.")
        resolved_error_policy = _resolve_error_policy(error_policy)
        resolved_sync_policy = _resolve_sync_policy(sync_policy)

        resolved_name = _resolve_name(function, name)
        resolved_description, parameter_descriptions = resolve_documentation(
            function,
            tool_name=resolved_name,
            explicit_description=description,
        )
        try:
            adapter = ArgumentAdapter(
                function,
                resolved_name,
                parameter_descriptions,
            )
            spec = ToolSpec(
                resolved_name,
                description=resolved_description,
                parameters=adapter.schema,
            )
        except ToolDefinitionError:
            raise
        except (TypeError, ValueError) as error:
            raise ToolDefinitionError(
                f"Cannot define tool {resolved_name!r}: {error}",
                tool_name=resolved_name,
            ) from None

        object.__setattr__(self, "function", function)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "description", resolved_description)
        object.__setattr__(self, "error_policy", resolved_error_policy)
        object.__setattr__(self, "sync_policy", resolved_sync_policy)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "_adapter", adapter)

    async def invoke(
        self,
        call: ToolCallBlock,
        execution: ExecutionContext[Any],
    ) -> ToolResult:
        """Validate one model call, invoke the callable, and encode its result.

        A call-name mismatch raises ``UnknownToolError``. Adapter validation
        errors propagate for ``InvokeTool`` to return to the model. Callable
        exceptions follow ``error_policy``; cancellation-like base exceptions
        are never converted.
        """

        if not isinstance(cast(object, call), ToolCallBlock):
            raise TypeError("call must be a ToolCallBlock.")
        if not isinstance(cast(object, execution), ExecutionContext):
            raise TypeError("execution must be an ExecutionContext.")
        if call.name != self.name:
            raise UnknownToolError(call.name)

        context = ToolContext(execution, call)
        arguments = self._adapter.validate(
            call.arguments,
            context,
            tool_name=self.name,
        )
        try:
            value = await self._call(arguments)
        except Exception as error:
            if self.error_policy is ToolErrorPolicy.PROPAGATE:
                raise
            return error_tool_result(error)
        return encode_tool_result(value, tool_name=self.name)

    async def _call(self, arguments: dict[str, object]) -> object:
        """Execute according to sync policy and await any returned awaitable."""

        if self.sync_policy is SyncToolPolicy.THREAD and not iscoroutinefunction(self.function):
            value = await asyncio.to_thread(partial(self.function, **arguments))
        else:
            value = self.function(**arguments)
        if isawaitable(value):
            return await value
        return value


ToolLike: TypeAlias = FunctionTool | ToolCallable
"""Already adapted tool or plain callable accepted by ``ToolSet``."""


@overload
def tool(function: ToolCallable, /) -> FunctionTool:
    """Wrap one callable immediately with default tool options."""

    ...


@overload
def tool(
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    error_policy: ToolErrorPolicy | str = ToolErrorPolicy.PROPAGATE,
    sync_policy: SyncToolPolicy | str = SyncToolPolicy.INLINE,
) -> Callable[[ToolCallable], FunctionTool]:
    """Return a decorator configured with explicit tool options."""

    ...


def tool(
    function: ToolCallable | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    error_policy: ToolErrorPolicy | str = ToolErrorPolicy.PROPAGATE,
    sync_policy: SyncToolPolicy | str = SyncToolPolicy.INLINE,
) -> FunctionTool | Callable[[ToolCallable], FunctionTool]:
    """Create a ``FunctionTool`` directly or configure a callable decorator."""

    factory = partial(
        FunctionTool,
        name=name,
        description=description,
        error_policy=error_policy,
        sync_policy=sync_policy,
    )
    if function is None:
        return factory
    return factory(function)


def _resolve_name(function: ToolCallable, explicit_name: str | None) -> str:
    """Prefer an explicit stable name, otherwise require callable ``__name__``."""

    if explicit_name is not None:
        if not isinstance(cast(object, explicit_name), str):
            raise TypeError("name must be a string.")
        if not explicit_name:
            raise ValueError("name must not be empty.")
        return explicit_name

    inferred_name = getattr(function, "__name__", None)
    if not isinstance(inferred_name, str) or not inferred_name:
        raise ToolDefinitionError(
            "Callable tools without a non-empty __name__ require an explicit name."
        )
    return inferred_name


def _resolve_error_policy(value: ToolErrorPolicy | str) -> ToolErrorPolicy:
    if isinstance(value, ToolErrorPolicy):
        return value
    if not isinstance(cast(object, value), str):
        raise TypeError("error_policy must be a ToolErrorPolicy or string value.")
    try:
        return ToolErrorPolicy(value)
    except ValueError:
        raise ValueError(f"Unsupported tool error policy {value!r}.") from None


def _resolve_sync_policy(value: SyncToolPolicy | str) -> SyncToolPolicy:
    if isinstance(value, SyncToolPolicy):
        return value
    if not isinstance(cast(object, value), str):
        raise TypeError("sync_policy must be a SyncToolPolicy or string value.")
    try:
        return SyncToolPolicy(value)
    except ValueError:
        raise ValueError(f"Unsupported sync tool policy {value!r}.") from None


__all__ = ["FunctionTool", "ToolLike", "tool"]
