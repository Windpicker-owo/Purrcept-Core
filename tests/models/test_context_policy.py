"""Verify context policies retain whole turns and enforce synchronous token budgets."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.backend import ModelEventSink
from purrcept_core.models.context_policy import (
    AppendOnlyContext,
    ContextBudgetExceededError,
    ContextPolicy,
    PreparedContext,
    RequestTokenBudget,
    SlidingWindowContext,
    TokenBudgetContext,
)
from purrcept_core.models.continuation import ModelContinuation
from purrcept_core.models.conversation import ConversationState
from purrcept_core.models.messages import Message
from purrcept_core.models.model import Model
from purrcept_core.models.requests import ModelRequest
from purrcept_core.models.responses import ModelResponse


class NeverBackend:
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        del request, model, emit
        raise AssertionError("context policy tests do not generate")


def _model() -> Model:
    return Model(NeverBackend(), "test-model")


def _state_with_pending_message() -> ConversationState:
    """Build two complete turns followed by an indivisible in-progress suffix."""

    return ConversationState(
        (
            Message.user("one"),
            Message.assistant("one"),
            Message.user("two"),
            Message.assistant("two"),
            Message.user("pending"),
        ),
        continuation=ModelContinuation("provider", {"cursor": "next"}),
        turn_index=2,
        turn_boundaries=(2, 4),
    )


def test_prepared_context_is_an_immutable_snapshot() -> None:
    messages = [Message.user("hello")]
    boundaries = [1]
    continuation = ModelContinuation("provider", {"cursor": 1})

    prepared = PreparedContext(
        messages,  # type: ignore[arg-type]
        continuation=continuation,
        turn_boundaries=boundaries,  # type: ignore[arg-type]
    )
    messages.append(Message.user("changed"))
    boundaries.clear()

    assert prepared.messages == (Message.user("hello"),)
    assert prepared.continuation is continuation
    assert prepared.turn_boundaries == (1,)
    assert not hasattr(prepared, "__dict__")
    with pytest.raises(FrozenInstanceError):
        prepared.messages = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: PreparedContext(1),  # type: ignore[arg-type]
            TypeError,
            "messages must be an iterable",
        ),
        (
            lambda: PreparedContext((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only Message",
        ),
        (
            lambda: PreparedContext(
                (),
                continuation=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "ModelContinuation or None",
        ),
        (
            lambda: PreparedContext((), turn_boundaries=1),  # type: ignore[arg-type]
            TypeError,
            "turn_boundaries must be an iterable",
        ),
        (
            lambda: PreparedContext(
                (Message.user("x"),),
                turn_boundaries=(True,),
            ),
            TypeError,
            "only ints",
        ),
        (
            lambda: PreparedContext(
                (Message.user("x"),),
                turn_boundaries=(0,),
            ),
            ValueError,
            "strictly increasing and positive",
        ),
        (
            lambda: PreparedContext(
                (Message.user("x"), Message.assistant("x")),
                turn_boundaries=(1, 1),
            ),
            ValueError,
            "strictly increasing and positive",
        ),
        (
            lambda: PreparedContext(
                (Message.user("x"),),
                turn_boundaries=(2,),
            ),
            ValueError,
            "must not exceed",
        ),
    ],
)
def test_prepared_context_validates_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_append_only_context_preserves_the_full_state() -> None:
    state = _state_with_pending_message()
    policy = AppendOnlyContext()

    model = _model()
    prepared = policy.prepare(state, model)

    assert isinstance(policy, ContextPolicy)
    assert prepared == PreparedContext(
        state.messages,
        continuation=state.continuation,
        turn_boundaries=state.turn_boundaries,
    )
    with pytest.raises(TypeError, match="state must be a ConversationState"):
        policy.prepare(object(), model)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model must be a Model"):
        policy.prepare(state, object())  # type: ignore[arg-type]


def test_sliding_window_keeps_complete_recent_turns_and_pending_suffix() -> None:
    state = _state_with_pending_message()

    prepared = SlidingWindowContext(1).prepare(state, _model())

    assert prepared.messages == (
        Message.user("two"),
        Message.assistant("two"),
        Message.user("pending"),
    )
    assert prepared.turn_boundaries == (2,)
    assert prepared.continuation is state.continuation


def test_sliding_window_supports_zero_and_oversized_windows() -> None:
    state = _state_with_pending_message()

    model = _model()
    no_history = SlidingWindowContext(0).prepare(state, model)
    all_history = SlidingWindowContext(10).prepare(state, model)

    assert no_history.messages == (Message.user("pending"),)
    assert no_history.turn_boundaries == ()
    assert all_history.messages == state.messages
    assert all_history.turn_boundaries == state.turn_boundaries
    with pytest.raises(TypeError, match="state must be a ConversationState"):
        SlidingWindowContext(1).prepare(object(), model)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model must be a Model"):
        SlidingWindowContext(1).prepare(state, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "error_type", "match"),
    [
        (True, TypeError, "must be an int"),
        (1.5, TypeError, "must be an int"),
        (-1, ValueError, "greater than or equal to zero"),
    ],
)
def test_sliding_window_validates_max_turns(
    value: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        SlidingWindowContext(value)  # type: ignore[arg-type]


def test_token_budget_keeps_newest_whole_turns_and_is_synchronous() -> None:
    state = _state_with_pending_message()
    model = _model()
    calls: list[tuple[tuple[Message, ...], Model]] = []

    def count(messages: tuple[Message, ...], model: Model) -> int:
        calls.append((messages, model))
        return len(messages)

    prepared = TokenBudgetContext(3, count).prepare(state, model)

    assert prepared.messages == (
        Message.user("two"),
        Message.assistant("two"),
        Message.user("pending"),
    )
    assert prepared.turn_boundaries == (2,)
    assert [len(messages) for messages, _ in calls] == [1, 3, 5]
    assert all(selected_model is model for _, selected_model in calls)


def test_token_budget_rejects_an_oversized_current_turn() -> None:
    state = _state_with_pending_message()

    with pytest.raises(ContextBudgetExceededError) as caught:
        TokenBudgetContext(2, lambda messages, model: len(messages) * 3).prepare(
            state,
            _model(),
        )

    assert caught.value.max_tokens == 2
    assert caught.value.required_tokens == 3


def test_token_budget_handles_a_state_without_completed_turns() -> None:
    state = ConversationState((Message.user("pending"),))

    prepared = TokenBudgetContext(
        1,
        lambda messages, model: len(messages),
    ).prepare(state, _model())

    assert prepared.messages == state.messages
    assert prepared.turn_boundaries == ()


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: TokenBudgetContext(True, len),  # type: ignore[arg-type]
            TypeError,
            "max_tokens must be an int",
        ),
        (
            lambda: TokenBudgetContext(0, len),  # type: ignore[arg-type]
            ValueError,
            "max_tokens must be greater than zero",
        ),
        (
            lambda: TokenBudgetContext(1, object()),  # type: ignore[arg-type]
            TypeError,
            "counter must be",
        ),
        (
            lambda: ContextBudgetExceededError(1, True),
            TypeError,
            "TokenCounter must return an int",
        ),
        (
            lambda: ContextBudgetExceededError(1, -1),
            ValueError,
            "non-negative",
        ),
    ],
)
def test_token_budget_validates_construction(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("count", "error_type", "match"),
    [
        (True, TypeError, "must return an int"),
        (-1, ValueError, "non-negative"),
    ],
)
def test_token_budget_validates_counter_results(
    count: object,
    error_type: type[Exception],
    match: str,
) -> None:
    policy = TokenBudgetContext(
        10,
        lambda messages, model: count,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(error_type, match=match):
        policy.prepare(_state_with_pending_message(), _model())


def test_token_budget_validates_prepare_boundaries() -> None:
    policy = TokenBudgetContext(10, lambda messages, model: len(messages))
    state = _state_with_pending_message()

    with pytest.raises(TypeError, match="state must be a ConversationState"):
        policy.prepare(object(), _model())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model must be a Model"):
        policy.prepare(state, object())  # type: ignore[arg-type]


def test_token_budget_rejects_an_async_counter() -> None:
    async def count(messages: tuple[Message, ...], model: Model) -> int:
        del model
        return len(messages)

    policy = TokenBudgetContext(10, count)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be synchronous"):
        policy.prepare(_state_with_pending_message(), _model())


def test_request_token_budget_counts_the_complete_request() -> None:
    seen: list[tuple[ModelRequest, Model]] = []

    def count(request: ModelRequest, model: Model) -> int:
        seen.append((request, model))
        return len(request.messages) + len(request.reminders) + len(request.tools)

    request = ModelRequest((Message.user("current"),))
    model = _model()
    budget = RequestTokenBudget(10, count)

    assert budget.count(request, model) == 1
    assert seen == [(request, model)]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: RequestTokenBudget(True, len),  # type: ignore[arg-type]
            TypeError,
            "max_tokens must be an int",
        ),
        (
            lambda: RequestTokenBudget(0, len),  # type: ignore[arg-type]
            ValueError,
            "max_tokens must be greater than zero",
        ),
        (
            lambda: RequestTokenBudget(1, object()),  # type: ignore[arg-type]
            TypeError,
            "counter must be",
        ),
    ],
)
def test_request_token_budget_validates_construction(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("value", "error_type", "match"),
    [
        (True, TypeError, "must return an int"),
        (-1, ValueError, "non-negative"),
    ],
)
def test_request_token_budget_validates_counter_results(
    value: object,
    error_type: type[Exception],
    match: str,
) -> None:
    budget = RequestTokenBudget(
        10,
        lambda request, model: value,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(error_type, match=match):
        budget.count(ModelRequest((Message.user("x"),)), _model())


def test_request_token_budget_rejects_invalid_inputs_and_async_counter() -> None:
    async def count(request: ModelRequest, model: Model) -> int:
        del model
        return len(request.messages)

    budget = RequestTokenBudget(10, count)  # type: ignore[arg-type]
    request = ModelRequest((Message.user("x"),))

    with pytest.raises(TypeError, match="request must be a ModelRequest"):
        budget.count(object(), _model())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model must be a Model"):
        budget.count(request, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be synchronous"):
        budget.count(request, _model())


def test_request_token_budget_rejects_a_non_coroutine_awaitable() -> None:
    class AwaitableCount:
        def __await__(self):
            yield
            return 1

    budget = RequestTokenBudget(
        10,
        lambda request, model: AwaitableCount(),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TypeError, match="must be synchronous"):
        budget.count(ModelRequest((Message.user("x"),)), _model())
