"""Verify deterministic prompt compilation, turn lifecycles, budgets, and diagnostics.

The scenarios cover immutable draft snapshots, reminder ordering and
consumption, dynamic-source resolution, complete-turn trimming, trace metadata,
and rejection of ambiguous or invalid public inputs.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.models import (
    CompiledPrompt,
    ContextBudgetExceededError,
    ConversationState,
    Message,
    Model,
    ModelContinuation,
    ModelRequest,
    ModelResponse,
    PreparedContext,
    PromptCachePolicy,
    PromptCompileError,
    PromptCompiler,
    PromptDiagnostics,
    PromptDraft,
    ReminderPlacement,
    ReminderResolutionContext,
    ReminderScope,
    ReminderState,
    RequestTokenBudget,
    SystemInstruction,
    SystemReminder,
    TextBlock,
    ToolSpec,
)
from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.prompt.compiler import TurnPromptSession
from purrcept_core.models.settings import ModelSettings


class NeverBackend:
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model, emit
        raise AssertionError("backend must not run in compiler tests")


def _model() -> Model:
    return Model(NeverBackend(), "compiler-model")


def _draft(history: PreparedContext | None = None) -> PromptDraft:
    return PromptDraft(
        PreparedContext((Message.user("current"),)) if history is None else history,
        instructions=(SystemInstruction.from_text("policy"),),
        tools=(ToolSpec("lookup"),),
        metadata={"trace": ["one"]},
        provider_options={"fixture": {"enabled": True}},
    )


def _resolution_context(history: PreparedContext | None = None) -> ReminderResolutionContext:
    prepared = PreparedContext((Message.user("current"),)) if history is None else history
    return ReminderResolutionContext(
        _model(),
        ConversationState(),
        prepared,
        1,
        1,
    )


def _reminder(
    text: str,
    *,
    key: str | None = None,
    scope: ReminderScope = ReminderScope.NEXT_REQUEST,
    priority: int = 0,
    placement: ReminderPlacement = ReminderPlacement.AUTO,
) -> SystemReminder:
    """Build concise reminder fixtures while leaving every lifecycle knob explicit."""

    return SystemReminder(
        (TextBlock(text),),
        key=key,
        scope=scope,
        priority=priority,
        placement=placement,
    )


def test_prompt_draft_snapshots_transport_inputs_and_uses_standard_defaults() -> None:
    metadata: dict[str, object] = {"trace": ["one"]}
    options: dict[str, object] = {"fixture": {"enabled": True}}
    instructions = [SystemInstruction.from_text("policy")]
    tools = [ToolSpec("lookup")]
    settings = ModelSettings()
    cache = PromptCachePolicy()
    continuation = ModelContinuation("provider", {"cursor": 1})

    draft = PromptDraft(
        PreparedContext((Message.user("current"),)),
        instructions=instructions,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
        settings=settings,
        cache=cache,
        continuation=continuation,
        metadata=metadata,  # type: ignore[arg-type]
        provider_options=options,  # type: ignore[arg-type]
    )
    instructions.clear()
    tools.clear()
    metadata.clear()
    options.clear()

    assert draft.instructions == (SystemInstruction.from_text("policy"),)
    assert draft.tools == (ToolSpec("lookup"),)
    assert draft.settings is settings
    assert draft.cache is cache
    assert draft.continuation is continuation
    assert draft.metadata == {"trace": ("one",)}
    assert draft.provider_options == {"fixture": {"enabled": True}}
    assert PromptDraft(PreparedContext((Message.user("x"),))).settings == ModelSettings()
    assert not hasattr(draft, "__dict__")
    with pytest.raises(FrozenInstanceError):
        draft.history = PreparedContext((Message.user("other"),))  # type: ignore[misc]


def test_prompt_compiler_sorts_stably_and_reports_metadata_without_raw_text() -> None:
    low = _reminder("low", key="low", priority=-1)
    high_a = _reminder(
        "high-a",
        key="high-a",
        scope=ReminderScope.TURN,
        priority=10,
        placement=ReminderPlacement.TAIL,
    )
    high_b = _reminder(
        "high-b",
        key="high-b",
        scope=ReminderScope.CONVERSATION,
        priority=10,
    )
    dynamic = _reminder("dynamic", key="dynamic", priority=5)
    compiler = PromptCompiler()

    first = compiler.compile(
        _draft(),
        stored_reminders=(low, high_a),
        turn_reminders=(high_b,),
        dynamic_reminders=(dynamic,),
        dynamic_source_ids=("workspace",),
        model_round=1,
    )
    second = compiler.compile(
        _draft(),
        stored_reminders=(low, high_a),
        turn_reminders=(high_b,),
        dynamic_reminders=(dynamic,),
        dynamic_source_ids=("workspace",),
        model_round=1,
    )

    assert first == second
    assert first.request.reminders == (high_a, high_b, dynamic, low)
    assert first.request.messages == (Message.user("current"),)
    assert first.diagnostics.reminder_keys == (
        "high-a",
        "high-b",
        "dynamic",
        "low",
    )
    assert first.diagnostics.reminder_scopes == (
        ReminderScope.TURN,
        ReminderScope.CONVERSATION,
        None,
        ReminderScope.NEXT_REQUEST,
    )
    assert first.diagnostics.reminder_placements[0] is ReminderPlacement.TAIL
    assert first.diagnostics.reminder_priorities == (10, 10, 5, -1)
    assert first.diagnostics.dynamic_source_ids == ("workspace",)
    assert len(first.diagnostics.content_hashes) == 4
    assert all(len(value) == 64 for value in first.diagnostics.content_hashes)
    assert len(first.diagnostics.prompt_fingerprint) == 64
    assert first.consumed_next_request is True


def test_turn_prompt_session_owns_fixed_scope_lifecycles_across_rounds() -> None:
    next_request = _reminder("next", key="next", priority=-1)
    turn = _reminder(
        "turn",
        key="turn",
        scope=ReminderScope.TURN,
        priority=5,
    )
    persistent = _reminder(
        "persistent",
        key="persistent",
        scope=ReminderScope.CONVERSATION,
    )
    session = TurnPromptSession.begin(ReminderState((next_request, turn, persistent)))

    assert session.remaining_state.reminders == (next_request, persistent)

    first = session.compile(
        _draft(),
        context=_resolution_context(),
        model_round=1,
    )
    second = session.compile(
        _draft(),
        context=_resolution_context(),
        model_round=2,
    )

    assert first.request.reminders == (turn, persistent, next_request)
    assert first.consumed_next_request is True
    assert session.remaining_state.reminders == (persistent,)
    assert second.request.reminders == (turn, persistent)
    assert second.consumed_next_request is False


def test_request_budget_recompiles_after_removing_old_complete_turns() -> None:
    history = PreparedContext(
        (
            Message.user("old-1"),
            Message.assistant("old-1"),
            Message.user("old-2"),
            Message.assistant("old-2"),
            Message.user("current"),
        ),
        turn_boundaries=(2, 4),
    )
    counted_requests: list[ModelRequest] = []

    def count(request: ModelRequest, model: Model) -> int:
        del model
        counted_requests.append(request)
        return len(request.messages)

    session = TurnPromptSession.begin(ReminderState())
    compiled = session.compile(
        _draft(history),
        context=_resolution_context(history),
        model_round=1,
        request_budget=RequestTokenBudget(3, count),
    )

    assert [len(request.messages) for request in counted_requests] == [5, 3]
    assert compiled.request.messages == history.messages[2:]
    assert compiled.diagnostics.estimated_tokens == 3
    assert compiled.diagnostics.trimmed_turns == 1


def test_request_budget_rejects_an_oversized_current_turn() -> None:
    budget = RequestTokenBudget(1, lambda request, model: len(request.messages) + 1)
    session = TurnPromptSession.begin(ReminderState())

    with pytest.raises(ContextBudgetExceededError) as caught:
        session.compile(
            _draft(),
            context=_resolution_context(),
            model_round=1,
            request_budget=budget,
        )

    assert caught.value.max_tokens == 1
    assert caught.value.required_tokens == 2


def test_compiler_rejects_duplicate_keys_across_stored_and_dynamic_state() -> None:
    stored = _reminder("stored", key="same")
    dynamic = _reminder("dynamic", key="same")

    with pytest.raises(PromptCompileError, match="appears more than once") as caught:
        PromptCompiler().compile(
            _draft(),
            stored_reminders=(stored,),
            dynamic_reminders=(dynamic,),
            dynamic_source_ids=("dynamic",),
            model_round=1,
        )

    assert caught.value.reminder_key == "same"


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: PromptDraft(cast(PreparedContext, object())),
            TypeError,
            "history must be a PreparedContext",
        ),
        (
            lambda: PromptDraft(
                PreparedContext((Message.user("x"),)),
                instructions=cast(tuple[SystemInstruction, ...], (object(),)),
            ),
            TypeError,
            "only SystemInstruction",
        ),
        (
            lambda: PromptDraft(
                PreparedContext((Message.user("x"),)),
                tools=cast(tuple[ToolSpec, ...], (object(),)),
            ),
            TypeError,
            "only ToolSpec",
        ),
        (
            lambda: PromptDraft(
                PreparedContext((Message.user("x"),)),
                settings=cast(ModelSettings, object()),
            ),
            TypeError,
            "settings must be a ModelSettings",
        ),
        (
            lambda: PromptDraft(
                PreparedContext((Message.user("x"),)),
                cache=cast(PromptCachePolicy, object()),
            ),
            TypeError,
            "cache must be a PromptCachePolicy",
        ),
        (
            lambda: PromptDraft(
                PreparedContext((Message.user("x"),)),
                continuation=cast(ModelContinuation, object()),
            ),
            TypeError,
            "continuation must be a ModelContinuation",
        ),
        (
            lambda: CompiledPrompt(
                cast(ModelRequest, object()),
                cast(PromptDiagnostics, object()),
            ),
            TypeError,
            "request must be a ModelRequest",
        ),
        (
            lambda: CompiledPrompt(
                ModelRequest((Message.user("x"),)),
                cast(PromptDiagnostics, object()),
            ),
            TypeError,
            "diagnostics must be PromptDiagnostics",
        ),
        (
            lambda: CompiledPrompt(
                ModelRequest((Message.user("x"),)),
                PromptDiagnostics(1, (), (), (), (), (), "fingerprint"),
                consumed_next_request=cast(bool, 1),
            ),
            TypeError,
            "consumed_next_request must be a bool",
        ),
        (
            lambda: PromptCompiler().compile(cast(PromptDraft, object()), model_round=1),
            TypeError,
            "draft must be a PromptDraft",
        ),
        (
            lambda: PromptCompiler().compile(_draft(), model_round=True),
            TypeError,
            "model_round must be an int",
        ),
        (
            lambda: PromptCompiler().compile(_draft(), model_round=0),
            ValueError,
            "model_round must be greater than zero",
        ),
        (
            lambda: PromptCompiler().compile(
                _draft(),
                stored_reminders=cast(tuple[SystemReminder, ...], (object(),)),
                model_round=1,
            ),
            TypeError,
            "stored_reminders must contain only SystemReminder",
        ),
        (
            lambda: PromptCompiler().compile(
                _draft(),
                dynamic_source_ids=cast(tuple[str, ...], (1,)),
                model_round=1,
            ),
            TypeError,
            "dynamic_source_ids must contain only strings",
        ),
        (
            lambda: PromptCompiler().compile(
                _draft(),
                dynamic_source_ids=("",),
                model_round=1,
            ),
            ValueError,
            "must not contain empty strings",
        ),
        (
            lambda: PromptCompiler().compile(
                _draft(),
                dynamic_source_ids=("same", "same"),
                model_round=1,
            ),
            ValueError,
            "must be unique",
        ),
        (
            lambda: TurnPromptSession(
                cast(ReminderState, object()),
                compiler=PromptCompiler(),
            ),
            TypeError,
            "state must be a ReminderState",
        ),
        (
            lambda: TurnPromptSession(
                ReminderState(),
                compiler=cast(PromptCompiler, object()),
            ),
            TypeError,
            "compiler must be a PromptCompiler",
        ),
        (
            lambda: TurnPromptSession.begin(ReminderState()).compile(
                _draft(),
                context=cast(ReminderResolutionContext, object()),
                model_round=1,
            ),
            TypeError,
            "context must be a ReminderResolutionContext",
        ),
        (
            lambda: TurnPromptSession.begin(ReminderState()).compile(
                _draft(),
                context=_resolution_context(),
                model_round=1,
                request_budget=cast(RequestTokenBudget, object()),
            ),
            TypeError,
            "request_budget must be a RequestTokenBudget",
        ),
    ],
)
def test_prompt_compiler_values_validate_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
