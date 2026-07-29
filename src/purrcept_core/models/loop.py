"""Orchestrate one transactional conversation turn across model and tool effects.

The public facade supplies committed immutable state and validated reusable
configuration to :func:`run_model_turn`. The generator builds provisional
history, prepares and compiles each request, yields model generation, and
executes requested tools in provider order. It returns a new committed state
only after a final assistant response.

This module never mutates the input state or the owning ``Conversation``.
Failures, cancellation, and tool-loop limits unwind the generator without a
partial transcript commit; the facade owns busy-state and reminder-session
cleanup around this loop.
"""

from __future__ import annotations

from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TYPE_CHECKING, cast

from ..errors import PurrceptError
from ..flow import AgentFlow, perform
from ._utils import JsonValue
from .caching import PromptCachePolicy
from .context_policy import ContextPolicy, PreparedContext, RequestTokenBudget
from .continuation import ContinuationPolicy, ModelContinuation
from .instructions import SystemInstruction
from .messages import Message, MessageRole
from .model import Model
from .prompt.compiler import PromptDraft, TurnPromptSession
from .reminder_sources import ReminderResolutionContext
from .responses import ModelResponse
from .settings import ModelSettings
from .tools import InvokeTool, ToolResult, ToolSet

if TYPE_CHECKING:
    from .conversation import ConversationState


class ConversationBusyError(PurrceptError):
    """Raised when mutable conversation state already has an active turn."""

    def __init__(self) -> None:
        super().__init__("The conversation already has an active turn.")


class ToolLoopLimitError(PurrceptError):
    """Raised before tools run when no model round remains for their results.

    The last response is retained for diagnosis, but neither that response nor
    its requested tools are committed to conversation history.
    """

    def __init__(
        self,
        max_model_rounds: int,
        *,
        last_response: ModelResponse,
    ) -> None:
        validate_max_model_rounds(max_model_rounds)
        if not isinstance(cast(object, last_response), ModelResponse):
            raise TypeError("last_response must be a ModelResponse.")
        self.max_model_rounds = max_model_rounds
        self.model_rounds = max_model_rounds
        self.last_response = last_response
        super().__init__(f"Model tool loop exceeded max_model_rounds={max_model_rounds}.")


@dataclass(frozen=True, slots=True)
class ModelTurnResult:
    """The immutable committed result of one complete user/model/tool turn."""

    response: ModelResponse
    state: ConversationState
    model_rounds: int
    tool_results: tuple[ToolResult, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.response), ModelResponse):
            raise TypeError("response must be a ModelResponse.")
        _require_conversation_state(self.state)
        validate_max_model_rounds(self.model_rounds)
        tool_results = _normalize_tool_results(self.tool_results)
        object.__setattr__(self, "tool_results", tool_results)

    @property
    def text(self) -> str:
        """Return the final response text."""

        return self.response.text

    @property
    def rounds(self) -> int:
        """Alias for the number of model generations in this turn."""

        return self.model_rounds

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return the messages committed by this turn."""

        boundaries = self.state.turn_boundaries
        if not boundaries:
            return ()
        start = boundaries[-2] if len(boundaries) > 1 else 0
        end = boundaries[len(boundaries) - 1]
        return self.state.messages[start:end]

    @property
    def continuation(self) -> ModelContinuation | None:
        """Return the latest continuation committed with the turn."""

        return self.state.continuation


def run_model_turn(
    state: ConversationState,
    user_message: Message,
    *,
    model: Model,
    instructions: tuple[SystemInstruction, ...],
    prompt_session: TurnPromptSession,
    tools: ToolSet,
    settings: ModelSettings,
    cache: PromptCachePolicy,
    continuation_policy: ContinuationPolicy,
    context_policy: ContextPolicy,
    max_model_rounds: int,
    metadata: Mapping[str, JsonValue],
    provider_options: Mapping[str, JsonValue],
    request_budget: RequestTokenBudget | None,
) -> AgentFlow[ModelTurnResult]:
    """Run one model/tool transaction and return a new immutable state.

    The caller must provide committed ``state`` and a user-role message. Model
    responses and tool results remain provisional until a response contains no
    tool calls. Ordinary effect failures enter this generator through
    ``perform`` and propagate unless application-level wrapping handles them.
    """

    # Validate the transaction boundary before allocating provisional state or
    # yielding any effect. This guarantees input errors cannot consume prompt
    # state or produce backend calls.
    state = _require_conversation_state(state)
    if not isinstance(cast(object, user_message), Message):
        raise TypeError("user_message must be a Message.")
    if user_message.role is not MessageRole.USER:
        raise ValueError("user_message role must be user.")
    validate_max_model_rounds(max_model_rounds)

    # The mutable lists are transaction-local working copies. Only the frozen
    # ConversationState constructed at the successful return point can escape.
    working_messages = list(state.messages)
    working_messages.append(user_message)
    working_continuation = state.continuation
    tool_results: list[ToolResult] = []

    for model_round in range(1, max_model_rounds + 1):
        # Context selection and prompt compilation are repeated after each tool
        # round so dynamic reminders and token budgets observe the latest
        # provisional transcript.
        provisional_state = _new_conversation_state(
            messages=tuple(working_messages),
            continuation=working_continuation,
            turn_index=state.turn_index,
            turn_boundaries=state.turn_boundaries,
        )
        prepared_value = cast(
            object,
            context_policy.prepare(provisional_state, model),
        )
        if isawaitable(prepared_value):
            if isinstance(prepared_value, Coroutine):
                prepared_value.close()
            raise TypeError("ContextPolicy.prepare() must be synchronous.")
        if not isinstance(prepared_value, PreparedContext):
            raise TypeError("ContextPolicy.prepare() must return PreparedContext.")
        prepared = prepared_value

        draft = PromptDraft(
            prepared,
            instructions=instructions,
            tools=tools.specs,
            settings=settings,
            cache=cache,
            continuation=_request_continuation(
                continuation_policy,
                prepared.continuation,
            ),
            metadata=metadata,
            provider_options=provider_options,
        )
        compiled = prompt_session.compile(
            draft,
            context=ReminderResolutionContext(
                model,
                provisional_state,
                prepared,
                state.turn_index + 1,
                model_round,
            ),
            model_round=model_round,
            request_budget=request_budget,
        )

        # The model call is the async runtime boundary. Its assistant message
        # and continuation are added only to transaction-local state.
        response = yield from perform(model.generate(compiled.request))

        working_messages.append(response.message)
        working_continuation = response.continuation
        calls = response.tool_calls
        if not calls:
            # Commit point: one final assistant response closes the turn. The
            # new boundary includes the user message, every intermediate
            # assistant/tool pair, and this final response.
            committed_state = _new_conversation_state(
                messages=tuple(working_messages),
                continuation=working_continuation,
                turn_index=state.turn_index + 1,
                turn_boundaries=(*state.turn_boundaries, len(working_messages)),
            )
            return ModelTurnResult(
                response,
                committed_state,
                model_round,
                tool_results=tuple(tool_results),
            )

        if model_round == max_model_rounds:
            # Ordering: reject before invoking tools because their results could
            # never be delivered to another model round.
            raise ToolLoopLimitError(
                max_model_rounds,
                last_response=response,
            )

        # Tool calls execute sequentially in provider order. This preserves
        # deterministic side effects and appends a matching tool message after
        # each successful invocation.
        for call in calls:
            result = yield from perform(InvokeTool(tools.get(call.name), call))
            tool_results.append(result)
            working_messages.append(
                Message(
                    MessageRole.TOOL,
                    (result.to_block(call.id),),
                    name=call.name,
                )
            )

    raise AssertionError("unreachable model loop state")


def _new_conversation_state(
    *,
    messages: tuple[Message, ...],
    continuation: ModelContinuation | None,
    turn_index: int,
    turn_boundaries: tuple[int, ...],
) -> ConversationState:
    """Construct a validated state without creating an import cycle at startup."""

    from .conversation import ConversationState

    return ConversationState(
        messages,
        continuation=continuation,
        turn_index=turn_index,
        turn_boundaries=turn_boundaries,
    )


def _require_conversation_state(value: object) -> ConversationState:
    from .conversation import ConversationState

    if not isinstance(value, ConversationState):
        raise TypeError("state must be a ConversationState.")
    return value


def _request_continuation(
    policy: ContinuationPolicy,
    continuation: ModelContinuation | None,
) -> ModelContinuation | None:
    """Suppress provider state only when transcript history is authoritative."""

    if policy is ContinuationPolicy.CLIENT_MANAGED:
        return None
    return continuation


def _normalize_tool_results(value: object) -> tuple[ToolResult, ...]:
    """Snapshot tool results stored on the immutable turn result."""

    if not isinstance(value, Iterable):
        raise TypeError("tool_results must be an iterable.")
    results = tuple(cast(Iterable[object], value))
    if not all(isinstance(result, ToolResult) for result in results):
        raise TypeError("tool_results must contain only ToolResult instances.")
    return cast(tuple[ToolResult, ...], results)


def validate_max_model_rounds(value: object) -> None:
    """Validate the positive model-round limit shared by facade and loop."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_model_rounds must be an int.")
    if value <= 0:
        raise ValueError("max_model_rounds must be greater than zero.")


__all__ = [
    "ConversationBusyError",
    "ModelTurnResult",
    "ToolLoopLimitError",
]
