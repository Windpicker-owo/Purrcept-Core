"""Bridge model-requested tool calls into the core effect runtime.

The model loop resolves a call name against ``ToolSet`` and yields an
``InvokeTool`` effect. Its inline executor invokes the selected ``FunctionTool``
and converts expected model mistakes—unknown tools and invalid arguments—into
error results that the next model round can inspect. Tool implementation
exceptions follow the function tool's explicit error policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ...context import ExecutionContext
from ...effect import Effect
from ..content import ToolCallBlock
from ._encoding import error_tool_result
from .errors import ToolInputValidationError, UnknownToolError
from .function import FunctionTool
from .values import ToolResult


@dataclass(frozen=True, slots=True)
class InvokeTool(Effect[ToolResult]):
    """Immutable effect pairing a model tool call with its local resolution.

    ``tool`` may be ``None`` to preserve an unknown call as an executable effect
    and model-readable error. A non-``None`` tool must still match the call name
    at execution, preventing stale or incorrect registry resolution.
    """

    tool: FunctionTool | None
    call: ToolCallBlock

    def __post_init__(self) -> None:
        if self.tool is not None and not isinstance(
            cast(object, self.tool),
            FunctionTool,
        ):
            raise TypeError("tool must be a FunctionTool or None.")
        if not isinstance(cast(object, self.call), ToolCallBlock):
            raise TypeError("call must be a ToolCallBlock.")

    async def execute(self, context: ExecutionContext[object]) -> ToolResult:
        """Return expected call errors while preserving configured tool failures."""

        tool = self.tool
        if tool is None or tool.name != self.call.name:
            return error_tool_result(UnknownToolError(self.call.name))
        try:
            return await tool.invoke(self.call, context)
        except ToolInputValidationError as error:
            return error_tool_result(error)


__all__ = ["InvokeTool"]
