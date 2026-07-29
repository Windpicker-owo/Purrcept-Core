"""Verify conversation configuration, atomic state, checkpoints, and reminder APIs.

The scenarios distinguish committed history from active-turn state, exercise
restore and fork isolation, and assert that prompt-control lifecycles retain
their documented idle-state boundaries.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.caching import CacheMode, PromptCachePolicy
from purrcept_core.models.content import TextBlock
from purrcept_core.models.context_policy import AppendOnlyContext, RequestTokenBudget
from purrcept_core.models.continuation import ContinuationPolicy, ModelContinuation
from purrcept_core.models.conversation import (
    Conversation,
    ConversationCheckpoint,
    ConversationState,
)
from purrcept_core.models.instructions import SystemInstruction
from purrcept_core.models.messages import Message
from purrcept_core.models.model import Model
from purrcept_core.models.prompt import PromptCompiler, PromptRenderError, PromptTemplate
from purrcept_core.models.reminder_sources import ReminderResolutionContext
from purrcept_core.models.reminders import (
    ReminderPlacement,
    ReminderScope,
    ReminderState,
    SystemReminder,
)
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse
from purrcept_core.models.settings import ModelSettings, ToolChoice
from purrcept_core.models.tools import ToolSet


class NeverBackend:
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model, emit
        raise AssertionError("backend must not run in value-object tests")


def _model() -> Model:
    return Model(NeverBackend(), "test-model")


def _complete_state() -> ConversationState:
    return ConversationState(
        (Message.user("hello"), Message.assistant("hi")),
        continuation=ModelContinuation("provider", {"cursor": 1}),
        turn_index=1,
        turn_boundaries=(2,),
    )


def test_conversation_state_is_a_frozen_snapshot() -> None:
    messages = [Message.user("hello"), Message.assistant("hi")]
    boundaries = [2]
    continuation = ModelContinuation("provider", {"cursor": 1})

    state = ConversationState(
        messages,  # type: ignore[arg-type]
        continuation=continuation,
        turn_index=1,
        turn_boundaries=boundaries,  # type: ignore[arg-type]
    )
    messages.clear()
    boundaries.clear()

    assert state.messages == (Message.user("hello"), Message.assistant("hi"))
    assert state.continuation is continuation
    assert state.turn_index == 1
    assert state.turn_boundaries == (2,)
    assert not hasattr(state, "__dict__")
    with pytest.raises(FrozenInstanceError):
        state.turn_index = 2  # type: ignore[misc]


def test_conversation_checkpoint_is_a_frozen_complete_state_value() -> None:
    history = _complete_state()
    reminders = ReminderState((SystemReminder((TextBlock("next"),), key="next"),))
    checkpoint = ConversationCheckpoint(history, reminders)

    assert checkpoint.history is history
    assert checkpoint.reminders is reminders
    assert not hasattr(checkpoint, "__dict__")
    with pytest.raises(FrozenInstanceError):
        checkpoint.history = ConversationState()  # type: ignore[misc]
    with pytest.raises(TypeError, match="history must be a ConversationState"):
        ConversationCheckpoint(object(), reminders)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reminders must be a ReminderState"):
        ConversationCheckpoint(history, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ConversationState(1),  # type: ignore[arg-type]
            TypeError,
            "messages must be an iterable",
        ),
        (
            lambda: ConversationState((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only Message",
        ),
        (
            lambda: ConversationState(
                (),
                continuation=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "ModelContinuation or None",
        ),
        (
            lambda: ConversationState((), turn_index=True),
            TypeError,
            "turn_index must be an int",
        ),
        (
            lambda: ConversationState((), turn_index=-1),
            ValueError,
            "greater than or equal to zero",
        ),
        (
            lambda: ConversationState((), turn_boundaries=1),  # type: ignore[arg-type]
            TypeError,
            "turn_boundaries must be an iterable",
        ),
        (
            lambda: ConversationState(
                (Message.user("x"),),
                turn_index=1,
                turn_boundaries=(True,),
            ),
            TypeError,
            "only ints",
        ),
        (
            lambda: ConversationState(
                (Message.user("x"),),
                turn_index=1,
                turn_boundaries=(0,),
            ),
            ValueError,
            "strictly increasing and positive",
        ),
        (
            lambda: ConversationState(
                (Message.user("x"), Message.assistant("x")),
                turn_index=2,
                turn_boundaries=(1, 1),
            ),
            ValueError,
            "strictly increasing and positive",
        ),
        (
            lambda: ConversationState(
                (Message.user("x"),),
                turn_index=1,
                turn_boundaries=(2,),
            ),
            ValueError,
            "must not exceed",
        ),
        (
            lambda: ConversationState(
                (Message.user("x"),),
                turn_index=1,
                turn_boundaries=(),
            ),
            ValueError,
            "turn_index must equal",
        ),
    ],
)
def test_conversation_state_validates_every_boundary(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_conversation_normalizes_configuration_and_takes_json_snapshots() -> None:
    def echo(value: str) -> str:
        return value

    metadata: dict[str, object] = {"tags": ["one"]}
    options: dict[str, object] = {"reasoning": {"effort": "low"}}
    state = _complete_state()

    conversation = Conversation(
        _model(),
        instructions="Be concise.",
        reminders="Use metric units.",
        tools=(echo,),
        cache="prefer",
        continuation_policy="auto",
        max_model_rounds=3,
        metadata=metadata,  # type: ignore[arg-type]
        provider_options=options,  # type: ignore[arg-type]
        state=state,
    )
    metadata.clear()
    options.clear()

    assert conversation.instructions == (SystemInstruction.from_text("Be concise."),)
    assert conversation.reminders == (SystemReminder((TextBlock("Use metric units."),)),)
    assert isinstance(conversation.tools, ToolSet)
    assert tuple(conversation.tools.tools) == ("echo",)
    assert conversation.settings == ModelSettings()
    assert conversation.cache == PromptCachePolicy(mode=CacheMode.PREFER)
    assert conversation.continuation_policy is ContinuationPolicy.AUTO
    assert isinstance(conversation.context_policy, AppendOnlyContext)
    assert conversation.max_model_rounds == 3
    assert conversation.metadata == {"tags": ("one",)}
    assert conversation.provider_options == {
        "reasoning": {"effort": "low"},
    }
    assert conversation.state is state
    assert conversation.messages == state.messages
    assert conversation.continuation is state.continuation
    assert conversation.turn_index == 1
    assert conversation.busy is False


def test_conversation_exposes_prompt_control_configuration() -> None:
    class Source:
        source_id = "source"

        def resolve(self, context: ReminderResolutionContext):
            del context
            return ()

    source = Source()
    compiler = PromptCompiler()
    budget = RequestTokenBudget(10, lambda request, model: 1)
    events: list[object] = []
    sink = events.append
    conversation = Conversation(
        _model(),
        reminder_sources=(source,),
        prompt_compiler=compiler,
        request_budget=budget,
        prompt_trace_sink=sink,
    )

    assert conversation.reminder_sources == (source,)
    assert conversation.prompt_compiler is compiler
    assert conversation.request_budget is budget
    assert conversation.prompt_trace_sink is sink
    assert conversation.reminder_state == ReminderState()


def test_conversation_accepts_single_objects_and_existing_toolset() -> None:
    instruction = SystemInstruction.from_text("safe")
    reminder = SystemReminder(
        (TextBlock("remember"),),
        key="same",
        scope=ReminderScope.CONVERSATION,
    )
    tools = ToolSet()
    settings = ModelSettings()
    cache = PromptCachePolicy()
    policy = AppendOnlyContext()

    conversation = Conversation(
        _model(),
        instructions=instruction,
        reminders=(reminder, reminder),
        tools=tools,
        settings=settings,
        cache=cache,
        continuation_policy=ContinuationPolicy.PROVIDER_MANAGED,
        context_policy=policy,
    )

    assert conversation.instructions == (instruction,)
    assert conversation.reminders == (reminder,)
    assert conversation.tools is tools
    assert conversation.settings is settings
    assert conversation.cache is cache
    assert conversation.continuation_policy is ContinuationPolicy.PROVIDER_MANAGED
    assert conversation.context_policy is policy


def test_conversation_compiles_templates_at_its_model_boundary() -> None:
    instruction_template = PromptTemplate(
        "instruction",
        "Act as {role}.",
        values={"role": "a researcher"},
    )
    reminder_template = PromptTemplate(
        "reminder",
        "Use {style}.",
        values={"style": "citations"},
    )
    explicit_instruction = SystemInstruction.from_text("Be concise.")

    conversation = _model().conversation(
        instructions=(instruction_template, explicit_instruction),
        reminders=reminder_template,
    )

    assert conversation.instructions == (
        SystemInstruction.from_text("Act as a researcher."),
        explicit_instruction,
    )
    assert conversation.reminders == (SystemReminder((TextBlock("Use citations."),)),)


def test_conversation_rejects_unbound_templates_before_starting() -> None:
    template = PromptTemplate("unbound", "Use {style}.")

    with pytest.raises(PromptRenderError, match="style"):
        Conversation(_model(), instructions=template)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"instructions": object()}, TypeError, "instructions must be"),
        (
            {"instructions": (object(),)},
            TypeError,
            "only SystemInstruction",
        ),
        (
            {
                "instructions": (
                    SystemInstruction.from_text("one", key="same"),
                    SystemInstruction.from_text("two", key="same"),
                )
            },
            ValueError,
            "unique non-empty keys",
        ),
        ({"reminders": object()}, TypeError, "reminders must be"),
        (
            {"reminders": (object(),)},
            TypeError,
            "only SystemReminder",
        ),
        ({"tools": object()}, TypeError, "tools must be"),
        ({"settings": object()}, TypeError, "settings must be"),
        (
            {"settings": ModelSettings(tool_choice=ToolChoice.REQUIRED)},
            ValueError,
            "tools must not be empty",
        ),
        ({"cache": "forever"}, ValueError, "Unsupported cache mode"),
        (
            {"continuation_policy": "sometimes"},
            ValueError,
            "Unsupported continuation policy",
        ),
        ({"context_policy": object()}, TypeError, "implement ContextPolicy"),
        (
            {"prompt_compiler": object()},
            TypeError,
            "prompt_compiler must be a PromptCompiler",
        ),
        (
            {"request_budget": object()},
            TypeError,
            "request_budget must be a RequestTokenBudget",
        ),
        (
            {"prompt_trace_sink": object()},
            TypeError,
            "prompt_trace_sink must be callable",
        ),
        ({"max_model_rounds": True}, TypeError, "must be an int"),
        ({"max_model_rounds": 0}, ValueError, "greater than zero"),
        ({"state": object()}, TypeError, "ConversationState or None"),
        ({"metadata": []}, TypeError, "metadata must be a mapping"),
        (
            {"provider_options": {"bad": object()}},
            TypeError,
            "unsupported JSON value",
        ),
    ],
)
def test_conversation_validates_configuration(
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        Conversation(_model(), **kwargs)  # type: ignore[arg-type]


def test_conversation_requires_a_model() -> None:
    with pytest.raises(TypeError, match="model must be a Model"):
        Conversation(object())  # type: ignore[arg-type]


def test_snapshot_restore_fork_and_clear_use_committed_state() -> None:
    reminder = SystemReminder(
        (TextBlock("persistent"),),
        key="persistent",
        scope=ReminderScope.CONVERSATION,
    )
    conversation = Conversation(_model(), reminders=reminder)
    state = _complete_state()

    conversation.restore_history(state)
    checkpoint = conversation.snapshot()
    fork = conversation.fork()
    empty_checkpoint = ConversationCheckpoint(ConversationState(), ReminderState())
    empty_fork = conversation.fork(empty_checkpoint)
    conversation.clear_history()

    assert checkpoint == ConversationCheckpoint(state, ReminderState((reminder,)))
    assert conversation.history_snapshot() == ConversationState()
    assert conversation.state == ConversationState()
    assert conversation.reminders == (reminder,)
    assert fork is not conversation
    assert fork.state is state
    assert fork.reminders == (reminder,)
    assert fork.model is conversation.model
    assert empty_fork.state == ConversationState()
    assert empty_fork.reminders == ()
    fork.reset()
    assert conversation.state == ConversationState()
    assert fork.reminders == ()

    conversation.restore(checkpoint)
    assert conversation.state is state
    assert conversation.reminders == (reminder,)


def test_restore_and_fork_validate_state() -> None:
    conversation = Conversation(_model())

    with pytest.raises(TypeError, match="checkpoint must be a ConversationCheckpoint"):
        conversation.restore(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="checkpoint must be a ConversationCheckpoint or None"):
        conversation.fork(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state must be a ConversationState"):
        conversation.restore_history(object())  # type: ignore[arg-type]


def test_clear_alias_warns_and_only_clears_history() -> None:
    reminder = SystemReminder((TextBlock("next"),))
    conversation = Conversation(_model(), reminders=reminder, state=_complete_state())

    with pytest.warns(DeprecationWarning, match="clear_history"):
        conversation.clear()

    assert conversation.state == ConversationState()
    assert conversation.reminders == (reminder,)


def test_clear_reminders_by_scope_and_reset_have_explicit_boundaries() -> None:
    next_request = SystemReminder((TextBlock("next"),), key="next")
    turn = SystemReminder(
        (TextBlock("turn"),),
        key="turn",
        scope=ReminderScope.TURN,
    )
    persistent = SystemReminder(
        (TextBlock("persistent"),),
        key="persistent",
        scope=ReminderScope.CONVERSATION,
    )
    conversation = Conversation(
        _model(),
        reminders=(next_request, turn, persistent),
        state=_complete_state(),
    )

    conversation.clear_reminders("turn")
    assert conversation.reminders == (next_request, persistent)
    assert conversation.state == _complete_state()

    conversation.clear_reminders()
    assert conversation.reminders == ()
    assert conversation.state == _complete_state()

    conversation.remind(next_request)
    conversation.reset()
    assert conversation.state == ConversationState()
    assert conversation.reminders == ()


def test_remind_replaces_keyed_values_and_remove_has_an_alias() -> None:
    conversation = Conversation(_model())
    first = conversation.remind(
        "first",
        key="policy",
        scope="turn",
        placement="tail",
        priority=2,
    )
    unkeyed = conversation.remind("unkeyed")
    replacement = conversation.remind(
        "replacement",
        key="policy",
        scope=ReminderScope.CONVERSATION,
    )

    assert first.scope is ReminderScope.TURN
    assert first.placement is ReminderPlacement.TAIL
    assert first.priority == 2
    assert conversation.reminders == (replacement, unkeyed)
    assert conversation.remove("missing") is False
    assert conversation.remove_reminder("policy") is True
    assert conversation.reminders == (unkeyed,)


def test_remind_accepts_an_existing_value_without_options() -> None:
    conversation = Conversation(_model())
    reminder = SystemReminder((TextBlock("existing"),), key="existing")

    assert conversation.remind(reminder) is reminder

    with pytest.raises(TypeError, match="options cannot be supplied"):
        conversation.remind(reminder, priority=1)
    with pytest.raises(TypeError, match="string or SystemReminder"):
        conversation.remind(object())  # type: ignore[arg-type]


def test_remind_accepts_a_bound_template_with_standard_lifecycle_options() -> None:
    conversation = Conversation(_model())
    template = PromptTemplate(
        "continue",
        "Continue {task}.",
        values={"task": "research"},
    )

    reminder = conversation.remind(
        template,
        key="continue",
        scope=ReminderScope.CONVERSATION,
        placement=ReminderPlacement.TAIL,
        priority=10,
    )

    assert reminder == SystemReminder(
        (TextBlock("Continue research."),),
        key="continue",
        scope=ReminderScope.CONVERSATION,
        placement=ReminderPlacement.TAIL,
        priority=10,
    )
    assert conversation.reminders == (reminder,)


@pytest.mark.parametrize(
    ("key", "error_type", "match"),
    [
        (1, TypeError, "key must be a string"),
        ("", ValueError, "key must not be empty"),
    ],
)
def test_remove_reminder_validates_key(
    key: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        Conversation(_model()).remove_reminder(cast(str, key))


@pytest.mark.parametrize(
    ("message", "error_type", "match"),
    [
        (object(), TypeError, "string or Message"),
        (Message.assistant("no"), ValueError, "role must be user"),
    ],
)
def test_ask_validates_user_messages(
    message: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        Conversation(_model()).ask(message)  # type: ignore[arg-type]


def test_ask_accepts_an_explicit_user_message() -> None:
    flow = Conversation(_model()).ask(Message.user("hello"))

    flow.close()


def test_ask_compiles_a_bound_template_as_the_user_message() -> None:
    template = PromptTemplate(
        "question",
        "Research {topic}.",
        values={"topic": "prompt caching"},
    )
    flow = Conversation(_model()).ask(template)
    effect = next(flow)

    assert effect.request.messages == (Message.user("Research prompt caching."),)
    flow.close()


@pytest.mark.parametrize(
    ("value", "error_type"),
    [(True, TypeError), (0, ValueError)],
)
def test_ask_validates_round_override(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="max_model_rounds"):
        Conversation(_model()).ask(
            "hello",
            max_model_rounds=value,  # type: ignore[arg-type]
        )
