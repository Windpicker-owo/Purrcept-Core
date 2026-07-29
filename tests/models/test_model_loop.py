"""Verify transactional model turns across context, prompts, tools, and failures.

The scenarios assert provisional history ordering, atomic commit, continuation
policy, repeated tool rounds, loop limits, busy-state cleanup, and validation
at every orchestration hand-off.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.driver import AgentDriver
from purrcept_core.events import EffectFailed, EffectStarted, EffectSucceeded
from purrcept_core.executor import InlineExecutor
from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.content import TextBlock, ToolCallBlock, ToolResultBlock
from purrcept_core.models.context_policy import (
    ContextBudgetExceededError,
    PreparedContext,
    RequestTokenBudget,
)
from purrcept_core.models.continuation import ContinuationPolicy, ModelContinuation
from purrcept_core.models.conversation import Conversation, ConversationState
from purrcept_core.models.effects import Generate
from purrcept_core.models.loop import (
    ConversationBusyError,
    ModelTurnResult,
    ToolLoopLimitError,
)
from purrcept_core.models.messages import Message, MessageRole
from purrcept_core.models.model import Model
from purrcept_core.models.reminder_sources import ReminderResolutionContext
from purrcept_core.models.reminders import ReminderScope, SystemReminder
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse
from purrcept_core.models.tools import InvokeTool
from purrcept_core.testing import RecordingEventSink


class SequenceBackend:
    def __init__(self, outcomes: list[ModelResponse | BaseException]) -> None:
        self.outcomes = outcomes
        self.requests: list[ModelRequest] = []
        self.models: list[str] = []

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del emit
        self.requests.append(request)
        self.models.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _tool_response(
    *calls: ToolCallBlock,
    continuation: ModelContinuation | None = None,
) -> ModelResponse:
    return ModelResponse(
        Message(MessageRole.ASSISTANT, calls),
        continuation=continuation,
    )


def _final_response(
    text: str = "done",
    *,
    continuation: ModelContinuation | None = None,
) -> ModelResponse:
    return ModelResponse(
        Message.assistant(text),
        continuation=continuation,
    )


async def _drive(
    conversation: Conversation,
    message: str = "hello",
    *,
    events: RecordingEventSink | None = None,
    max_model_rounds: int | None = None,
) -> ModelTurnResult:
    """Run one conversation turn through the same inline effect path as applications."""

    flow = conversation.ask(message, max_model_rounds=max_model_rounds)
    return await AgentDriver(
        InlineExecutor(),
        event_sink=events,
    ).run(flow, host=object())


async def test_tool_loop_runs_each_generate_and_tool_as_an_independent_effect() -> None:
    def lookup(query: str) -> str:
        """Look up a value."""

        return f"found:{query}"

    call = ToolCallBlock("call-1", "lookup", arguments={"query": "cat"})
    first_continuation = ModelContinuation("provider", {"cursor": "first"})
    final_continuation = ModelContinuation("provider", {"cursor": "final"})
    backend = SequenceBackend(
        [
            _tool_response(call, continuation=first_continuation),
            _final_response("The answer is cat.", continuation=final_continuation),
        ]
    )
    conversation = Conversation(
        Model(backend, "smart-model"),
        tools=(lookup,),
        continuation_policy=ContinuationPolicy.PROVIDER_MANAGED,
    )
    events = RecordingEventSink()

    result = await _drive(conversation, "cat", events=events)

    assert result.text == "The answer is cat."
    assert result.rounds == result.model_rounds == 2
    assert result.continuation == final_continuation
    assert len(result.tool_results) == 1
    assert result.tool_results[0].content == (TextBlock("found:cat"),)
    assert result.messages == conversation.messages
    assert conversation.turn_index == 1
    assert conversation.state.turn_boundaries == (4,)
    assert [message.role for message in conversation.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    tool_message = conversation.messages[2]
    assert tool_message.name == "lookup"
    assert tool_message.content == (
        ToolResultBlock(
            "call-1",
            content=(TextBlock("found:cat"),),
        ),
    )

    assert len(backend.requests) == 2
    first_request, second_request = backend.requests
    assert first_request.messages == (Message.user("cat"),)
    assert second_request.messages == conversation.messages[:-1]
    assert first_request.continuation is None
    assert second_request.continuation == first_continuation
    assert first_request.tools == conversation.tools.specs
    assert backend.models == ["smart-model", "smart-model"]

    started = events.of_type(EffectStarted)
    succeeded = events.of_type(EffectSucceeded)
    assert [type(event.effect) for event in started] == [
        Generate,
        InvokeTool,
        Generate,
    ]
    assert [type(event.effect) for event in succeeded] == [
        Generate,
        InvokeTool,
        Generate,
    ]
    assert [event.step_index for event in started] == [0, 1, 2]
    assert [event.effect_id for event in started] == [event.effect_id for event in succeeded]
    assert not events.of_type(EffectFailed)
    with pytest.raises(FrozenInstanceError):
        result.model_rounds = 3  # type: ignore[misc]


async def test_tool_calls_are_invoked_sequentially_in_response_order() -> None:
    invoked: list[str] = []

    def first() -> str:
        invoked.append("first")
        return "one"

    def second() -> str:
        invoked.append("second")
        return "two"

    backend = SequenceBackend(
        [
            _tool_response(
                ToolCallBlock("call-1", "first"),
                ToolCallBlock("call-2", "second"),
            ),
            _final_response(),
        ]
    )
    conversation = Conversation(Model(backend, "model"), tools=(first, second))
    events = RecordingEventSink()

    result = await _drive(conversation, events=events)

    assert invoked == ["first", "second"]
    assert [tool_result.content for tool_result in result.tool_results] == [
        (TextBlock("one"),),
        (TextBlock("two"),),
    ]
    assert [type(event.effect) for event in events.of_type(EffectStarted)] == [
        Generate,
        InvokeTool,
        InvokeTool,
        Generate,
    ]
    assert [message.name for message in backend.requests[1].messages[-2:]] == [
        "first",
        "second",
    ]


async def test_unknown_tools_return_an_error_result_to_the_model() -> None:
    backend = SequenceBackend(
        [
            _tool_response(ToolCallBlock("missing-1", "missing")),
            _final_response("recovered"),
        ]
    )
    conversation = Conversation(Model(backend, "model"))

    result = await _drive(conversation)

    assert result.text == "recovered"
    assert result.tool_results[0].is_error is True
    tool_message = backend.requests[1].messages[-1]
    block = cast(ToolResultBlock, tool_message.content[0])
    assert tool_message.role is MessageRole.TOOL
    assert block.tool_call_id == "missing-1"
    assert block.is_error is True
    assert "Unknown tool" in tool_message.text


@pytest.mark.parametrize(
    ("policy", "expected_first", "expected_second"),
    [
        (ContinuationPolicy.CLIENT_MANAGED, None, None),
        (
            ContinuationPolicy.PROVIDER_MANAGED,
            ModelContinuation("provider", {"cursor": "base"}),
            ModelContinuation("provider", {"cursor": "round-1"}),
        ),
        (
            ContinuationPolicy.AUTO,
            ModelContinuation("provider", {"cursor": "base"}),
            ModelContinuation("provider", {"cursor": "round-1"}),
        ),
    ],
)
async def test_continuation_policy_controls_what_requests_send(
    policy: ContinuationPolicy,
    expected_first: ModelContinuation | None,
    expected_second: ModelContinuation | None,
) -> None:
    base_continuation = ModelContinuation("provider", {"cursor": "base"})
    round_continuation = ModelContinuation("provider", {"cursor": "round-1"})
    final_continuation = ModelContinuation("provider", {"cursor": "round-2"})
    state = ConversationState(
        (Message.user("old"), Message.assistant("old")),
        continuation=base_continuation,
        turn_index=1,
        turn_boundaries=(2,),
    )
    backend = SequenceBackend(
        [
            _tool_response(
                ToolCallBlock("missing", "unknown"),
                continuation=round_continuation,
            ),
            _final_response(continuation=final_continuation),
        ]
    )
    conversation = Conversation(
        Model(backend, "model"),
        continuation_policy=policy,
        state=state,
    )

    result = await _drive(conversation, "new")

    assert [request.continuation for request in backend.requests] == [
        expected_first,
        expected_second,
    ]
    assert result.continuation is final_continuation
    assert conversation.state.turn_boundaries == (2, 6)
    assert conversation.turn_index == 2
    assert result.messages == conversation.messages[2:]


async def test_reminders_have_stable_priority_and_scope_lifecycles() -> None:
    next_request = SystemReminder(
        (TextBlock("next"),),
        key="next",
        scope=ReminderScope.NEXT_REQUEST,
        priority=0,
    )
    turn = SystemReminder(
        (TextBlock("turn"),),
        key="turn",
        scope=ReminderScope.TURN,
        priority=5,
    )
    conversation_a = SystemReminder(
        (TextBlock("conversation-a"),),
        key="conversation-a",
        scope=ReminderScope.CONVERSATION,
        priority=5,
    )
    conversation_b = SystemReminder(
        (TextBlock("conversation-b"),),
        key="conversation-b",
        scope=ReminderScope.CONVERSATION,
        priority=1,
    )
    backend = SequenceBackend(
        [
            _tool_response(ToolCallBlock("missing", "unknown")),
            _final_response(),
        ]
    )
    conversation = Conversation(
        Model(backend, "model"),
        reminders=(next_request, turn, conversation_a, conversation_b),
    )

    await _drive(conversation)

    assert backend.requests[0].reminders == (
        turn,
        conversation_a,
        conversation_b,
        next_request,
    )
    assert backend.requests[1].reminders == (
        turn,
        conversation_a,
        conversation_b,
    )
    assert conversation.reminders == (conversation_a, conversation_b)
    assert all(
        reminder not in conversation.messages
        for reminder in (next_request, turn, conversation_a, conversation_b)
    )


async def test_model_failure_rolls_back_turn_and_consumes_attempted_reminders() -> None:
    expected = LookupError("provider failed")
    state = ConversationState(
        (Message.user("old"), Message.assistant("old")),
        turn_index=1,
        turn_boundaries=(2,),
    )
    next_request = SystemReminder(
        (TextBlock("next"),),
        scope=ReminderScope.NEXT_REQUEST,
    )
    turn = SystemReminder((TextBlock("turn"),), scope=ReminderScope.TURN)
    persistent = SystemReminder(
        (TextBlock("persistent"),),
        key="persistent",
        scope=ReminderScope.CONVERSATION,
    )
    conversation = Conversation(
        Model(SequenceBackend([expected]), "model"),
        reminders=(next_request, turn, persistent),
        state=state,
    )

    with pytest.raises(LookupError) as caught:
        await _drive(conversation)

    assert caught.value is expected
    assert conversation.state is state
    assert conversation.reminders == (persistent,)
    assert conversation.busy is False


async def test_failure_before_generate_keeps_next_request_but_consumes_turn() -> None:
    class InvalidPolicy:
        def prepare(self, state: ConversationState, model: Model) -> object:
            del state, model
            return object()

    next_request = SystemReminder(
        (TextBlock("next"),),
        scope=ReminderScope.NEXT_REQUEST,
    )
    turn = SystemReminder((TextBlock("turn"),), scope=ReminderScope.TURN)
    conversation = Conversation(
        Model(SequenceBackend([]), "model"),
        reminders=(next_request, turn),
        context_policy=InvalidPolicy(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="must return PreparedContext"):
        await _drive(conversation)

    assert conversation.state == ConversationState()
    assert conversation.reminders == (next_request,)
    assert conversation.busy is False


def test_flow_that_is_never_advanced_does_not_acquire_or_consume_reminders() -> None:
    next_request = SystemReminder((TextBlock("next"),), key="next")
    turn = SystemReminder(
        (TextBlock("turn"),),
        key="turn",
        scope=ReminderScope.TURN,
    )
    conversation = Conversation(
        Model(SequenceBackend([]), "model"),
        reminders=(next_request, turn),
    )

    flow = conversation.ask("never advanced")
    flow.close()

    assert conversation.busy is False
    assert conversation.reminders == (next_request, turn)


async def test_async_context_policy_is_rejected_without_becoming_an_effect() -> None:
    class AsyncPolicy:
        async def prepare(
            self,
            state: ConversationState,
            model: Model,
        ) -> PreparedContext:
            del model
            return PreparedContext(
                state.messages,
                continuation=state.continuation,
                turn_boundaries=state.turn_boundaries,
            )

    conversation = Conversation(
        Model(SequenceBackend([]), "model"),
        context_policy=AsyncPolicy(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="must be synchronous"):
        await _drive(conversation)

    assert conversation.state == ConversationState()
    assert conversation.busy is False


async def test_tool_exception_rolls_back_without_committing_partial_messages() -> None:
    expected = LookupError("tool failed")

    def explode() -> str:
        raise expected

    backend = SequenceBackend([_tool_response(ToolCallBlock("explode-1", "explode"))])
    conversation = Conversation(Model(backend, "model"), tools=(explode,))
    events = RecordingEventSink()

    with pytest.raises(LookupError) as caught:
        await _drive(conversation, events=events)

    assert caught.value is expected
    assert conversation.state == ConversationState()
    assert conversation.busy is False
    assert [type(event.effect) for event in events.of_type(EffectStarted)] == [
        Generate,
        InvokeTool,
    ]
    assert len(events.of_type(EffectFailed)) == 1
    assert isinstance(events.of_type(EffectFailed)[0].effect, InvokeTool)


async def test_dynamic_reminder_source_refreshes_after_a_tool_changes_runtime_state() -> None:
    workspace = {"dirty": False}

    class WorkspaceSource:
        source_id = "workspace"

        def __init__(self) -> None:
            self.contexts: list[ReminderResolutionContext] = []

        def resolve(self, context: ReminderResolutionContext):
            self.contexts.append(context)
            return (
                SystemReminder(
                    (TextBlock(f"dirty={workspace['dirty']}"),),
                    key="workspace:state",
                ),
            )

    source = WorkspaceSource()

    def modify_workspace() -> str:
        workspace["dirty"] = True
        return "modified"

    backend = SequenceBackend(
        [
            _tool_response(ToolCallBlock("modify-1", "modify_workspace")),
            _final_response(),
        ]
    )
    conversation = Conversation(
        Model(backend, "model"),
        tools=(modify_workspace,),
        reminder_sources=(source,),
    )

    await _drive(conversation)

    assert backend.requests[0].reminders[0].content == (TextBlock("dirty=False"),)
    assert backend.requests[1].reminders[0].content == (TextBlock("dirty=True"),)
    assert [context.model_round for context in source.contexts] == [1, 2]
    assert all(
        TextBlock("dirty=False") not in message.content
        and TextBlock("dirty=True") not in message.content
        for message in conversation.messages
    )
    assert conversation.reminders == ()


async def test_source_failure_rolls_back_history_and_preserves_next_request() -> None:
    expected = LookupError("source failed")
    next_request = SystemReminder((TextBlock("next"),), key="next")

    class FailingSource:
        source_id = "failing"

        def resolve(self, context: ReminderResolutionContext):
            del context
            raise expected

    conversation = Conversation(
        Model(SequenceBackend([]), "model"),
        reminders=(next_request,),
        reminder_sources=(FailingSource(),),
    )

    with pytest.raises(LookupError) as caught:
        await _drive(conversation)

    assert caught.value is expected
    assert conversation.state == ConversationState()
    assert conversation.reminders == (next_request,)


async def test_full_request_budget_counts_control_and_tools_before_generate() -> None:
    seen: list[ModelRequest] = []

    def count(request: ModelRequest, model: Model) -> int:
        del model
        seen.append(request)
        return (
            len(request.messages)
            + len(request.instructions)
            + len(request.reminders)
            + len(request.tools)
        )

    def lookup(query: str) -> str:
        return query

    reminder = SystemReminder((TextBlock("control"),), key="control")
    backend = SequenceBackend([_final_response()])
    conversation = Conversation(
        Model(backend, "model"),
        instructions="policy",
        reminders=(reminder,),
        tools=(lookup,),
        request_budget=RequestTokenBudget(4, count),
    )

    await _drive(conversation)

    assert seen == backend.requests
    assert len(seen[0].instructions) == 1
    assert len(seen[0].reminders) == 1
    assert len(seen[0].tools) == 1


async def test_rejected_full_request_budget_does_not_consume_next_request() -> None:
    next_request = SystemReminder((TextBlock("large-control"),), key="next")
    conversation = Conversation(
        Model(SequenceBackend([]), "model"),
        reminders=(next_request,),
        request_budget=RequestTokenBudget(1, lambda request, model: 2),
    )

    with pytest.raises(ContextBudgetExceededError, match="exceeding max_tokens=1"):
        await _drive(conversation)

    assert conversation.reminders == (next_request,)


async def test_round_limit_fails_before_unusable_tool_side_effects() -> None:
    invoked = False

    def must_not_run() -> str:
        nonlocal invoked
        invoked = True
        return "unexpected"

    response = _tool_response(ToolCallBlock("call-1", "must_not_run"))
    backend = SequenceBackend([response])
    conversation = Conversation(
        Model(backend, "model"),
        tools=(must_not_run,),
        max_model_rounds=1,
    )
    events = RecordingEventSink()

    with pytest.raises(ToolLoopLimitError) as caught:
        await _drive(conversation, events=events)

    assert caught.value.max_model_rounds == 1
    assert caught.value.model_rounds == 1
    assert caught.value.last_response is response
    assert invoked is False
    assert conversation.state == ConversationState()
    assert [type(event.effect) for event in events.of_type(EffectStarted)] == [Generate]


def test_busy_guard_blocks_concurrent_turns_and_mutations() -> None:
    next_request = SystemReminder(
        (TextBlock("next"),),
        scope=ReminderScope.NEXT_REQUEST,
    )
    turn = SystemReminder((TextBlock("turn"),), scope=ReminderScope.TURN)
    conversation = Conversation(
        Model(SequenceBackend([_final_response()]), "model"),
        reminders=(next_request, turn),
    )
    checkpoint = conversation.snapshot()
    first_flow = conversation.ask("first")

    first_effect = next(first_flow)

    assert isinstance(first_effect, Generate)
    assert conversation.busy is True
    assert conversation.history_snapshot() == ConversationState()
    second_flow = conversation.ask("second")
    with pytest.raises(ConversationBusyError):
        next(second_flow)
    with pytest.raises(ConversationBusyError):
        conversation.remind("blocked")
    with pytest.raises(ConversationBusyError):
        conversation.remove_reminder("missing")
    with pytest.raises(ConversationBusyError):
        conversation.snapshot()
    with pytest.raises(ConversationBusyError):
        conversation.fork()
    with pytest.raises(ConversationBusyError):
        conversation.restore(checkpoint)
    with pytest.raises(ConversationBusyError):
        conversation.restore_history(ConversationState())
    with pytest.raises(ConversationBusyError):
        conversation.clear_history()
    with pytest.raises(ConversationBusyError):
        conversation.clear_reminders()
    with pytest.raises(ConversationBusyError):
        conversation.reset()
    with pytest.raises(ConversationBusyError):
        _ = conversation.reminders
    with pytest.raises(ConversationBusyError):
        _ = conversation.reminder_state
    with pytest.warns(DeprecationWarning, match="clear_history"):
        with pytest.raises(ConversationBusyError):
            conversation.clear()

    first_flow.close()

    assert conversation.busy is False
    assert conversation.reminders == ()
    assert conversation.state == ConversationState()


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ModelTurnResult(
                cast(ModelResponse, object()),
                ConversationState(),
                1,
            ),
            TypeError,
            "response must be",
        ),
        (
            lambda: ModelTurnResult(
                _final_response(),
                cast(ConversationState, object()),
                1,
            ),
            TypeError,
            "state must be",
        ),
        (
            lambda: ModelTurnResult(
                _final_response(),
                ConversationState(),
                True,
            ),
            TypeError,
            "max_model_rounds must be an int",
        ),
        (
            lambda: ModelTurnResult(
                _final_response(),
                ConversationState(),
                0,
            ),
            ValueError,
            "greater than zero",
        ),
        (
            lambda: ModelTurnResult(
                _final_response(),
                ConversationState(),
                1,
                tool_results=1,  # type: ignore[arg-type]
            ),
            TypeError,
            "tool_results must be an iterable",
        ),
        (
            lambda: ModelTurnResult(
                _final_response(),
                ConversationState(),
                1,
                tool_results=(object(),),  # type: ignore[arg-type]
            ),
            TypeError,
            "only ToolResult",
        ),
        (
            lambda: ToolLoopLimitError(
                0,
                last_response=_final_response(),
            ),
            ValueError,
            "greater than zero",
        ),
        (
            lambda: ToolLoopLimitError(
                1,
                last_response=cast(ModelResponse, object()),
            ),
            TypeError,
            "last_response must be",
        ),
    ],
)
def test_model_turn_values_validate_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_empty_state_turn_result_has_no_turn_messages() -> None:
    result = ModelTurnResult(_final_response(), ConversationState(), 1)

    assert result.messages == ()
    assert result.continuation is None
    assert str(ConversationBusyError()) == "The conversation already has an active turn."
