"""Refresh request-local reminders after a tool call without storing projections.

The reminder source reads mutable workspace state before every model request.
Its dynamic value changes after tool execution, while the conversation stores
only the explicitly registered persistent safety reminder.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor
from purrcept_core.models import (
    Conversation,
    FinishReason,
    FunctionTool,
    Message,
    MessageRole,
    Model,
    ModelEventSink,
    ModelRequest,
    ModelResponse,
    ModelTurnResult,
    PromptTraceEvent,
    ReminderPlacement,
    ReminderResolutionContext,
    ReminderScope,
    SystemReminder,
    TextBlock,
    ToolCallBlock,
    tool,
)


@dataclass(slots=True)
class WorkspaceState:
    """Mutable application state observed by a reminder source and tool."""

    dirty: bool = False
    changed_file: str | None = None


class WorkspaceReminderSource:
    """Project current workspace state before every model request."""

    source_id = "workspace"

    def __init__(self, workspace: WorkspaceState) -> None:
        self._workspace = workspace

    def resolve(
        self,
        context: ReminderResolutionContext,
    ) -> tuple[SystemReminder, ...]:
        """Project the latest workspace fields into one request-local reminder."""

        del context
        changed = self._workspace.changed_file or "none"
        return (
            SystemReminder(
                (TextBlock(f"workspace dirty={self._workspace.dirty}; changed_file={changed}"),),
                key="workspace:state",
                placement=ReminderPlacement.TAIL,
            ),
        )


class OfflineBackend:
    """Ask for one state-changing tool call, then return a final answer."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Request one workspace mutation, then acknowledge refreshed state."""

        del emit
        self.requests.append(request)
        if not any(message.role is MessageRole.TOOL for message in request.messages):
            return ModelResponse(
                Message(
                    MessageRole.ASSISTANT,
                    (
                        ToolCallBlock(
                            "mark-dirty-1",
                            "mark_dirty",
                            arguments={"path": "src/auth.py"},
                        ),
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALL,
                model=model,
            )
        return ModelResponse(
            Message.assistant("The refreshed workspace state is visible."),
            model=model,
        )


def make_mark_dirty(workspace: WorkspaceState) -> FunctionTool:
    """Create a tool closure that mutates the supplied workspace state."""

    @tool
    def mark_dirty(path: str) -> str:
        """Record one modified workspace file.

        Args:
            path: Modified file path.
        """

        workspace.dirty = True
        workspace.changed_file = path
        return path

    return mark_dirty


def flow(conversation: Conversation) -> AgentFlow[ModelTurnResult]:
    """Ask one question that requires the state-changing tool."""

    return (yield from conversation.ask("Modify auth.py and report the current workspace state."))


def _workspace_text(request: ModelRequest) -> str:
    reminder = next(item for item in request.reminders if item.key == "workspace:state")
    block = reminder.content[0]
    assert isinstance(block, TextBlock)
    return block.text


async def main() -> None:
    """Run the dynamic-reminder scenario and verify stored-state isolation."""

    workspace = WorkspaceState()
    backend = OfflineBackend()
    trace: list[PromptTraceEvent] = []
    conversation = Model(backend, "offline-demo").conversation(
        tools=(make_mark_dirty(workspace),),
        reminder_sources=(WorkspaceReminderSource(workspace),),
        prompt_trace_sink=trace.append,
    )
    persistent = conversation.remind(
        "Apply repository safety rules.",
        key="runtime:safety",
        scope=ReminderScope.CONVERSATION,
    )
    before = conversation.snapshot()

    result = await AgentDriver(InlineExecutor()).run(flow(conversation), host=None)

    assert _workspace_text(backend.requests[0]) == "workspace dirty=False; changed_file=none"
    assert _workspace_text(backend.requests[1]) == (
        "workspace dirty=True; changed_file=src/auth.py"
    )
    assert before.reminders.reminders == (persistent,)
    assert conversation.snapshot().reminders.reminders == (persistent,)
    assert all(
        reminder.key != "workspace:state"
        for reminder in conversation.snapshot().reminders.reminders
    )
    assert result.text == "The refreshed workspace state is visible."
    assert trace
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
