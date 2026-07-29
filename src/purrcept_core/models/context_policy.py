"""Select immutable conversation history for the next model request.

Conversation orchestration calls a :class:`ContextPolicy` before prompt
compilation in every model round. This module owns transcript-only selection
and token-budget helpers; it never mutates committed history, resolves prompt
reminders, or performs provider I/O.

Policies must preserve complete turns and the entire in-progress turn.
Token-counting hooks are synchronous so request construction remains a
deterministic phase between effect executions.
"""

from __future__ import annotations

from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from ..errors import PurrceptError
from .continuation import ModelContinuation
from .messages import Message
from .model import Model

if TYPE_CHECKING:
    from .conversation import ConversationState
    from .requests import ModelRequest


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """The immutable transcript snapshot selected by a context policy.

    ``turn_boundaries`` are strictly increasing end offsets into ``messages``.
    They describe only complete turns; messages after the final boundary belong
    to the current in-progress turn.
    """

    messages: tuple[Message, ...]
    continuation: ModelContinuation | None = field(default=None, kw_only=True)
    turn_boundaries: tuple[int, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        messages = _normalize_messages(self.messages)
        _validate_continuation(self.continuation)
        boundaries = _normalize_boundaries(
            self.turn_boundaries,
            message_count=len(messages),
        )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "turn_boundaries", boundaries)


@runtime_checkable
class ContextPolicy(Protocol):
    """Extension contract for synchronous transcript selection.

    Applications construct a policy and register it on a conversation. Before
    each model request, the model loop calls :meth:`prepare` with the current
    provisional state and selected model. Implementations return a new
    :class:`PreparedContext` without mutating either input, yielding effects, or
    returning an awaitable.

    Built-ins are :class:`AppendOnlyContext`,
    :class:`SlidingWindowContext`, and :class:`TokenBudgetContext`; selection is
    explicit in conversation configuration. Exceptions and invalid return
    values abort the uncommitted turn.
    """

    def prepare(self, state: ConversationState, model: Model) -> PreparedContext:
        """Return a prepared immutable view without yielding another Effect."""

        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Extension contract for counting one candidate transcript synchronously.

    A :class:`TokenBudgetContext` calls the counter repeatedly while extending
    a candidate backward by complete turns. Implementations must be side-effect
    free for repeated inputs, return a non-negative ``int``, and own any
    tokenizer caching or thread-safety concerns.
    """

    def __call__(self, messages: tuple[Message, ...], model: Model) -> int:
        """Return a non-negative token count."""

        ...


@runtime_checkable
class RequestTokenCounter(Protocol):
    """Extension contract for counting one complete request synchronously.

    :class:`RequestTokenBudget` invokes the counter after prompt compilation,
    which includes instructions, reminders, tools, and transport options.
    Implementations must return a non-negative ``int`` and must not return an
    awaitable.
    """

    def __call__(self, request: ModelRequest, model: Model) -> int:
        """Return a non-negative token count."""

        ...


class ContextBudgetExceededError(PurrceptError):
    """Raised when no complete old turn can be removed to satisfy a budget."""

    def __init__(self, max_tokens: int, required_tokens: int) -> None:
        _validate_max_tokens(max_tokens)
        _validate_token_count(required_tokens)
        self.max_tokens = max_tokens
        self.required_tokens = required_tokens
        super().__init__(
            f"The current turn requires {required_tokens} tokens, "
            f"exceeding max_tokens={max_tokens}."
        )


@dataclass(frozen=True, slots=True)
class AppendOnlyContext:
    """Send the complete append-only transcript on every request."""

    def prepare(self, state: ConversationState, model: Model) -> PreparedContext:
        """Snapshot the entire committed and provisional transcript unchanged."""

        state = _require_conversation_state(state)
        _require_model(model)
        return PreparedContext(
            state.messages,
            continuation=state.continuation,
            turn_boundaries=state.turn_boundaries,
        )


@dataclass(frozen=True, slots=True)
class SlidingWindowContext:
    """Keep recent complete turns plus the entire in-progress turn.

    ``max_turns`` counts completed turns only. A value of zero still retains all
    messages belonging to the current turn.
    """

    max_turns: int

    def __post_init__(self) -> None:
        _validate_max_turns(self.max_turns)

    def prepare(self, state: ConversationState, model: Model) -> PreparedContext:
        """Return the newest complete turns and the unsplittable current suffix."""

        state = _require_conversation_state(state)
        _require_model(model)
        boundaries = state.turn_boundaries
        retained_start = max(0, len(boundaries) - self.max_turns)
        start_index = 0 if retained_start == 0 else boundaries[retained_start - 1]
        messages = state.messages[start_index:]
        retained_boundaries = tuple(
            boundary - start_index for boundary in boundaries[retained_start:]
        )
        return PreparedContext(
            messages,
            continuation=state.continuation,
            turn_boundaries=retained_boundaries,
        )


@dataclass(frozen=True, slots=True)
class TokenBudgetContext:
    """Apply a transcript-only budget without splitting any turn.

    The current in-progress turn is counted first and is never trimmed. Older
    complete turns are considered from newest to oldest, producing the longest
    suffix that fits.
    """

    max_tokens: int
    counter: TokenCounter = field(repr=False)

    def __post_init__(self) -> None:
        _validate_max_tokens(self.max_tokens)
        if not callable(cast(object, self.counter)):
            raise TypeError("counter must be a synchronous TokenCounter.")

    def prepare(self, state: ConversationState, model: Model) -> PreparedContext:
        """Select the longest whole-turn transcript suffix within the budget."""

        state = _require_conversation_state(state)
        model = _require_model(model)

        # Invariant: the provisional current turn is indivisible. Reject it
        # before considering old history so callers can distinguish an
        # intrinsically oversized request from ordinary history trimming.
        boundaries = state.turn_boundaries
        current_start = boundaries[-1] if boundaries else 0
        current_messages = state.messages[current_start:]
        current_tokens = _count_tokens(self.counter, current_messages, model)
        if current_tokens > self.max_tokens:
            raise ContextBudgetExceededError(self.max_tokens, current_tokens)

        # Extend the candidate backward one complete turn at a time. Counting
        # the complete suffix on each iteration lets provider tokenizers include
        # cross-message framing costs instead of assuming additive counts.
        start_index = current_start
        retained_start = len(boundaries)
        for turn_index in range(len(boundaries) - 1, -1, -1):
            candidate_start = 0 if turn_index == 0 else boundaries[turn_index - 1]
            candidate = state.messages[candidate_start:]
            if _count_tokens(self.counter, candidate, model) > self.max_tokens:
                break
            start_index = candidate_start
            retained_start = turn_index

        retained_boundaries = tuple(
            boundary - start_index for boundary in boundaries[retained_start:]
        )
        return PreparedContext(
            state.messages[start_index:],
            continuation=state.continuation,
            turn_boundaries=retained_boundaries,
        )


@dataclass(frozen=True, slots=True)
class RequestTokenBudget:
    """Validate complete compiled requests against a synchronous token counter.

    ``TurnPromptSession`` owns the trimming loop: it recompiles after dropping
    one old complete turn and calls :meth:`count` again. This value owns only
    validation of the counter boundary and the configured limit.
    """

    max_tokens: int
    counter: RequestTokenCounter = field(repr=False)

    def __post_init__(self) -> None:
        _validate_max_tokens(self.max_tokens)
        if not callable(cast(object, self.counter)):
            raise TypeError("counter must be a synchronous RequestTokenCounter.")

    def count(self, request: ModelRequest, model: Model) -> int:
        """Count and validate one complete request."""

        from .requests import ModelRequest

        if not isinstance(cast(object, request), ModelRequest):
            raise TypeError("request must be a ModelRequest.")
        model = _require_model(model)
        count: object = self.counter(request, model)
        if isawaitable(count):
            if isinstance(count, Coroutine):
                count.close()
            raise TypeError("RequestTokenCounter must be synchronous and return an int.")
        _validate_request_token_count(count)
        return count


def _require_conversation_state(value: object) -> ConversationState:
    from .conversation import ConversationState

    if not isinstance(value, ConversationState):
        raise TypeError("state must be a ConversationState.")
    return value


def _require_model(value: object) -> Model:
    if not isinstance(value, Model):
        raise TypeError("model must be a Model.")
    return value


def _normalize_messages(value: object) -> tuple[Message, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("messages must be an iterable.")
    messages = tuple(cast(Iterable[object], value))
    if not all(isinstance(message, Message) for message in messages):
        raise TypeError("messages must contain only Message instances.")
    return cast(tuple[Message, ...], messages)


def _validate_continuation(value: object) -> None:
    if value is not None and not isinstance(value, ModelContinuation):
        raise TypeError("continuation must be a ModelContinuation or None.")


def _normalize_boundaries(
    value: object,
    *,
    message_count: int,
) -> tuple[int, ...]:
    """Validate complete-turn end offsets against the selected message tuple."""

    if not isinstance(value, Iterable):
        raise TypeError("turn_boundaries must be an iterable.")
    raw_boundaries = tuple(cast(Iterable[object], value))
    if not all(
        isinstance(boundary, int) and not isinstance(boundary, bool) for boundary in raw_boundaries
    ):
        raise TypeError("turn_boundaries must contain only ints.")
    boundaries = cast(tuple[int, ...], raw_boundaries)
    previous = 0
    for boundary in boundaries:
        if boundary <= previous:
            raise ValueError("turn_boundaries must be strictly increasing and positive.")
        if boundary > message_count:
            raise ValueError("turn_boundaries must not exceed the number of messages.")
        previous = boundary
    return boundaries


def _validate_max_turns(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_turns must be an int.")
    if value < 0:
        raise ValueError("max_turns must be greater than or equal to zero.")


def _validate_max_tokens(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_tokens must be an int.")
    if value <= 0:
        raise ValueError("max_tokens must be greater than zero.")


def _validate_token_count(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("TokenCounter must return an int.")
    if value < 0:
        raise ValueError("TokenCounter must return a non-negative count.")


def _validate_request_token_count(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("RequestTokenCounter must return an int.")
    if value < 0:
        raise ValueError("RequestTokenCounter must return a non-negative count.")


def _count_tokens(
    counter: TokenCounter,
    messages: tuple[Message, ...],
    model: Model,
) -> int:
    """Call a transcript counter while rejecting awaitables and invalid counts."""

    count: object = counter(messages, model)
    if isawaitable(count):
        if isinstance(count, Coroutine):
            count.close()
        raise TypeError("TokenCounter must be synchronous and return an int.")
    _validate_token_count(count)
    return count


__all__ = [
    "AppendOnlyContext",
    "ContextBudgetExceededError",
    "ContextPolicy",
    "PreparedContext",
    "RequestTokenBudget",
    "RequestTokenCounter",
    "SlidingWindowContext",
    "TokenBudgetContext",
    "TokenCounter",
]
