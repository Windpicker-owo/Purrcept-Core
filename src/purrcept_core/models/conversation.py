"""Own committed conversation state around transactional model-turn flows.

:class:`Conversation` is the mutable application facade. It snapshots reusable
model, prompt, tool, and policy configuration, while each :meth:`ask` returns a
synchronous agent flow that performs model and tool effects. The lower-level
model loop builds a provisional transcript and returns a new immutable
:class:`ConversationState`; the facade commits that state only after the whole
turn succeeds.

Only one turn may be active on a facade. History and reminder checkpoints can
be restored or forked while idle. Prompt-reminder lifecycles are finalized even
when a turn fails, but an uncommitted transcript is never partially installed.
Backend resources, effect execution, and event delivery remain runtime-owned.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from ..flow import AgentFlow
from ._utils import JsonValue, freeze_json_object
from .caching import CacheMode, PromptCachePolicy
from .content import TextBlock
from .context_policy import AppendOnlyContext, ContextPolicy, RequestTokenBudget
from .continuation import ContinuationPolicy, ModelContinuation
from .instructions import SystemInstruction
from .loop import (
    ConversationBusyError,
    ModelTurnResult,
    run_model_turn,
    validate_max_model_rounds,
)
from .messages import Message, MessageRole
from .model import Model
from .prompt import PromptCompiler, PromptTemplate, PromptTraceSink
from .prompt.compiler import TurnPromptSession
from .prompt.tracing import validate_prompt_trace_sink
from .reminder_sources import ReminderSource, normalize_reminder_sources
from .reminders import (
    ReminderPlacement,
    ReminderScope,
    ReminderState,
    SystemReminder,
)
from .settings import ModelSettings, ToolChoice
from .tools import ToolLike, ToolSet


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Immutable committed transcript, continuation, and turn bookkeeping.

    ``turn_boundaries`` stores the exclusive message offset after each complete
    turn, so its length must equal ``turn_index``. Any provisional current turn
    is created by the model loop in a separate state and is never installed here
    until completion.
    """

    messages: tuple[Message, ...] = ()
    continuation: ModelContinuation | None = field(default=None, kw_only=True)
    turn_index: int = field(default=0, kw_only=True)
    turn_boundaries: tuple[int, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        messages = _normalize_messages(self.messages)
        _validate_continuation(self.continuation)
        _validate_turn_index(self.turn_index)
        boundaries = _normalize_turn_boundaries(
            self.turn_boundaries,
            message_count=len(messages),
        )
        if len(boundaries) != self.turn_index:
            raise ValueError("turn_index must equal the number of turn_boundaries.")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "turn_boundaries", boundaries)


@dataclass(frozen=True, slots=True)
class ConversationCheckpoint:
    """Complete mutable facade state, excluding reusable configuration.

    A checkpoint contains committed history and stored prompt reminders. Models,
    tools, policies, callbacks, and other constructor configuration remain owned
    by the receiving conversation when the checkpoint is restored.
    """

    history: ConversationState
    reminders: ReminderState

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.history), ConversationState):
            raise TypeError("history must be a ConversationState.")
        if not isinstance(cast(object, self.reminders), ReminderState):
            raise TypeError("reminders must be a ReminderState.")


class Conversation:
    """Mutable facade that owns committed history and idle reminder state.

    Applications create a facade through :meth:`Model.conversation` or directly,
    then pass flows returned by :meth:`ask` to an agent driver. Configuration is
    normalized and retained for the facade's lifetime. Each active turn owns a
    prompt session and provisional transcript; successful completion commits a
    new state atomically, while failure leaves committed history unchanged.

    A facade is intentionally single-turn: overlapping advancement raises
    :class:`ConversationBusyError`. It is not a synchronization primitive;
    callers that need concurrent branches should create them with :meth:`fork`.
    Reminder acquisition/consumption is finalized in ``finally`` even on
    failure, because those lifecycles belong to attempted requests rather than
    transcript commit.
    """

    __slots__ = (
        "_busy",
        "_cache",
        "_context_policy",
        "_continuation_policy",
        "_instructions",
        "_max_model_rounds",
        "_metadata",
        "_model",
        "_prompt_compiler",
        "_prompt_trace_sink",
        "_provider_options",
        "_reminder_sources",
        "_reminder_state",
        "_request_budget",
        "_settings",
        "_state",
        "_tools",
    )

    def __init__(
        self,
        model: Model,
        *,
        instructions: (
            str | PromptTemplate | SystemInstruction | Iterable[PromptTemplate | SystemInstruction]
        ) = (),
        reminders: (
            str | PromptTemplate | SystemReminder | Iterable[PromptTemplate | SystemReminder]
        ) = (),
        tools: ToolSet | Iterable[ToolLike] = (),
        settings: ModelSettings | None = None,
        cache: PromptCachePolicy | CacheMode | str | None = None,
        continuation_policy: ContinuationPolicy | str = (ContinuationPolicy.CLIENT_MANAGED),
        context_policy: ContextPolicy | None = None,
        reminder_sources: Iterable[ReminderSource] = (),
        prompt_compiler: PromptCompiler | None = None,
        request_budget: RequestTokenBudget | None = None,
        prompt_trace_sink: PromptTraceSink | None = None,
        max_model_rounds: int = 8,
        metadata: Mapping[str, JsonValue] | None = None,
        provider_options: Mapping[str, JsonValue] | None = None,
        state: ConversationState | None = None,
    ) -> None:
        # Normalize every reusable dependency before assigning instance state.
        # This keeps construction all-or-nothing and ensures later turn code can
        # treat configuration as already validated and snapshotted.
        if not isinstance(cast(object, model), Model):
            raise TypeError("model must be a Model.")
        normalized_instructions = _normalize_instructions(instructions)
        normalized_reminder_state = _normalize_reminders(reminders)
        normalized_tools = _normalize_tools(tools)
        normalized_settings = ModelSettings() if settings is None else settings
        if not isinstance(cast(object, normalized_settings), ModelSettings):
            raise TypeError("settings must be a ModelSettings.")
        if normalized_settings.tool_choice is ToolChoice.REQUIRED and not normalized_tools.specs:
            raise ValueError("tools must not be empty when tool_choice is required.")
        normalized_cache = _normalize_cache(cache)
        normalized_continuation_policy = _normalize_continuation_policy(continuation_policy)
        normalized_context_policy = (
            AppendOnlyContext() if context_policy is None else context_policy
        )
        if not isinstance(cast(object, normalized_context_policy), ContextPolicy):
            raise TypeError("context_policy must implement ContextPolicy.")
        normalized_sources = normalize_reminder_sources(reminder_sources)
        normalized_compiler = PromptCompiler() if prompt_compiler is None else prompt_compiler
        if not isinstance(cast(object, normalized_compiler), PromptCompiler):
            raise TypeError("prompt_compiler must be a PromptCompiler or None.")
        if request_budget is not None and not isinstance(
            cast(object, request_budget),
            RequestTokenBudget,
        ):
            raise TypeError("request_budget must be a RequestTokenBudget or None.")
        validate_prompt_trace_sink(prompt_trace_sink)
        validate_max_model_rounds(max_model_rounds)
        normalized_state = ConversationState() if state is None else state
        if not isinstance(cast(object, normalized_state), ConversationState):
            raise TypeError("state must be a ConversationState or None.")

        # Lifecycle: these configuration objects are shared across turns; only
        # `_state`, `_reminder_state`, and `_busy` change after construction.
        self._model = model
        self._instructions = normalized_instructions
        self._reminder_state = normalized_reminder_state
        self._tools = normalized_tools
        self._settings = normalized_settings
        self._cache = normalized_cache
        self._continuation_policy = normalized_continuation_policy
        self._context_policy = normalized_context_policy
        self._reminder_sources = normalized_sources
        self._prompt_compiler = normalized_compiler
        self._request_budget = request_budget
        self._prompt_trace_sink = prompt_trace_sink
        self._max_model_rounds = max_model_rounds
        self._metadata = freeze_json_object(
            {} if metadata is None else metadata,
            field_name="metadata",
        )
        self._provider_options = freeze_json_object(
            {} if provider_options is None else provider_options,
            field_name="provider_options",
        )
        self._state = normalized_state
        self._busy = False

    @property
    def model(self) -> Model:
        """Return the immutable model and backend binding used for every turn."""

        return self._model

    @property
    def instructions(self) -> tuple[SystemInstruction, ...]:
        """Return the fixed system instructions in configured order."""

        return self._instructions

    @property
    def reminders(self) -> tuple[SystemReminder, ...]:
        """Return stored reminders while no prompt session owns their lifecycle."""

        self._ensure_idle()
        return self._reminder_state.reminders

    @property
    def reminder_state(self) -> ReminderState:
        """Return the complete idle prompt-control state."""

        self._ensure_idle()
        return self._reminder_state

    @property
    def tools(self) -> ToolSet:
        """Return the immutable local tool registry used by automatic tool loops."""

        return self._tools

    @property
    def settings(self) -> ModelSettings:
        """Return the immutable generation settings shared by model rounds."""

        return self._settings

    @property
    def cache(self) -> PromptCachePolicy:
        """Return the provider-neutral prompt-cache policy."""

        return self._cache

    @property
    def continuation_policy(self) -> ContinuationPolicy:
        """Return who is responsible for sending provider continuation state."""

        return self._continuation_policy

    @property
    def context_policy(self) -> ContextPolicy:
        """Return the policy that selects history before each prompt compilation."""

        return self._context_policy

    @property
    def reminder_sources(self) -> tuple[ReminderSource, ...]:
        """Return dynamic reminder sources in request-resolution order."""

        return self._reminder_sources

    @property
    def prompt_compiler(self) -> PromptCompiler:
        """Return the pure compiler used by each turn-local prompt session."""

        return self._prompt_compiler

    @property
    def request_budget(self) -> RequestTokenBudget | None:
        """Return the optional complete-request token budget."""

        return self._request_budget

    @property
    def prompt_trace_sink(self) -> PromptTraceSink | None:
        """Return the optional synchronous metadata-only prompt trace callback."""

        return self._prompt_trace_sink

    @property
    def max_model_rounds(self) -> int:
        """Return the default per-turn limit including the initial model call."""

        return self._max_model_rounds

    @property
    def metadata(self) -> Mapping[str, JsonValue]:
        """Return the read-only request metadata snapshot."""

        return self._metadata

    @property
    def provider_options(self) -> Mapping[str, JsonValue]:
        """Return the read-only provider-specific request option snapshot."""

        return self._provider_options

    @property
    def state(self) -> ConversationState:
        """Return the latest committed history, even during an active turn."""

        return self._state

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return messages from the latest committed history."""

        return self._state.messages

    @property
    def continuation(self) -> ModelContinuation | None:
        """Return continuation state from the latest committed turn."""

        return self._state.continuation

    @property
    def turn_index(self) -> int:
        """Return the number of committed complete turns."""

        return self._state.turn_index

    @property
    def busy(self) -> bool:
        """Return whether a turn flow has acquired this facade."""

        return self._busy

    def ask(
        self,
        message: str | PromptTemplate | Message,
        *,
        max_model_rounds: int | None = None,
    ) -> AgentFlow[ModelTurnResult]:
        """Validate one user input and return its transactional agent flow.

        Calling this method does not acquire the facade. The busy lifecycle
        begins when the returned generator is first advanced and ends when it
        returns, raises, or is closed.
        """

        user_message = _normalize_user_message(message)
        rounds = self._max_model_rounds if max_model_rounds is None else max_model_rounds
        validate_max_model_rounds(rounds)
        return self._ask_flow(user_message, rounds)

    def _ask_flow(
        self,
        user_message: Message,
        max_model_rounds: int,
    ) -> AgentFlow[ModelTurnResult]:
        """Own the busy flag, prompt session, and atomic commit for one turn."""

        if self._busy:
            raise ConversationBusyError()
        self._busy = True

        # Ordering: acquire TURN reminders only after the busy transition. No
        # second flow may observe or consume prompt-control state while this
        # session owns it.
        prompt_session = TurnPromptSession.begin(
            self._reminder_state,
            compiler=self._prompt_compiler,
            reminder_sources=self._reminder_sources,
            trace_sink=self._prompt_trace_sink,
        )

        try:
            result = yield from run_model_turn(
                self._state,
                user_message,
                model=self._model,
                instructions=self._instructions,
                prompt_session=prompt_session,
                tools=self._tools,
                settings=self._settings,
                cache=self._cache,
                continuation_policy=self._continuation_policy,
                context_policy=self._context_policy,
                max_model_rounds=max_model_rounds,
                metadata=self._metadata,
                provider_options=self._provider_options,
                request_budget=self._request_budget,
            )

            # Commit only the state returned after the final model response and
            # all requested tools have completed successfully.
            self._state = result.state
            return result
        finally:
            # Lifecycle: request-scoped reminder consumption survives a failed
            # transcript transaction, while the prompt session never installs
            # dynamic reminders into stored state.
            self._reminder_state = prompt_session.remaining_state
            self._busy = False

    def snapshot(self) -> ConversationCheckpoint:
        """Return complete idle history and prompt-control state."""

        self._ensure_idle()
        return ConversationCheckpoint(self._state, self._reminder_state)

    def history_snapshot(self) -> ConversationState:
        """Return committed history, including while an uncommitted turn is active."""

        return self._state

    def restore(self, checkpoint: ConversationCheckpoint) -> None:
        """Restore complete mutable state while retaining Conversation configuration."""

        self._ensure_idle()
        if not isinstance(cast(object, checkpoint), ConversationCheckpoint):
            raise TypeError("checkpoint must be a ConversationCheckpoint.")
        self._state = checkpoint.history
        self._reminder_state = checkpoint.reminders

    def restore_history(self, state: ConversationState) -> None:
        """Replace only committed history, leaving prompt-control state unchanged."""

        self._ensure_idle()
        if not isinstance(cast(object, state), ConversationState):
            raise TypeError("state must be a ConversationState.")
        self._state = state

    def fork(
        self,
        checkpoint: ConversationCheckpoint | None = None,
    ) -> Conversation:
        """Create an independent facade from one complete idle checkpoint."""

        self._ensure_idle()
        fork_checkpoint = (
            ConversationCheckpoint(self._state, self._reminder_state)
            if checkpoint is None
            else checkpoint
        )
        if not isinstance(cast(object, fork_checkpoint), ConversationCheckpoint):
            raise TypeError("checkpoint must be a ConversationCheckpoint or None.")
        return Conversation(
            self._model,
            instructions=self._instructions,
            reminders=fork_checkpoint.reminders.reminders,
            tools=self._tools,
            settings=self._settings,
            cache=self._cache,
            continuation_policy=self._continuation_policy,
            context_policy=self._context_policy,
            reminder_sources=self._reminder_sources,
            prompt_compiler=self._prompt_compiler,
            request_budget=self._request_budget,
            prompt_trace_sink=self._prompt_trace_sink,
            max_model_rounds=self._max_model_rounds,
            metadata=self._metadata,
            provider_options=self._provider_options,
            state=fork_checkpoint.history,
        )

    def clear_history(self) -> None:
        """Clear committed history and continuation, retaining prompt-control state."""

        self._ensure_idle()
        self._state = ConversationState()

    def clear_reminders(self, scope: ReminderScope | str | None = None) -> None:
        """Clear all stored reminders or only one scope."""

        self._ensure_idle()
        self._reminder_state = self._reminder_state.clear(scope)

    def reset(self) -> None:
        """Clear both history and stored prompt-control state."""

        self._ensure_idle()
        self._state = ConversationState()
        self._reminder_state = ReminderState()

    def clear(self) -> None:
        """Deprecated alias for :meth:`clear_history`."""

        warnings.warn(
            "Conversation.clear() is deprecated; use clear_history().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.clear_history()

    def remind(
        self,
        reminder: str | PromptTemplate | SystemReminder,
        *,
        key: str | None = None,
        scope: ReminderScope | str = ReminderScope.NEXT_REQUEST,
        placement: ReminderPlacement | str = ReminderPlacement.AUTO,
        priority: int = 0,
    ) -> SystemReminder:
        """Register a reminder, replacing the existing reminder with the same key."""

        self._ensure_idle()
        if isinstance(reminder, str):
            normalized = SystemReminder(
                (TextBlock(reminder),),
                key=key,
                scope=cast(ReminderScope, scope),
                placement=cast(ReminderPlacement, placement),
                priority=priority,
            )
        elif isinstance(reminder, PromptTemplate):
            normalized = reminder.reminder(
                key=key,
                scope=cast(ReminderScope, scope),
                placement=cast(ReminderPlacement, placement),
                priority=priority,
            )
        elif isinstance(cast(object, reminder), SystemReminder):
            if (
                key is not None
                or scope != ReminderScope.NEXT_REQUEST
                or placement != ReminderPlacement.AUTO
                or priority != 0
            ):
                raise TypeError("reminder options cannot be supplied with a SystemReminder.")
            normalized = reminder
        else:
            raise TypeError("reminder must be a string or SystemReminder, or a PromptTemplate.")
        self._reminder_state = self._reminder_state.upsert(normalized)
        return normalized

    def remove_reminder(self, key: str) -> bool:
        """Remove the keyed reminder and report whether it existed."""

        self._ensure_idle()
        state, removed = self._reminder_state.remove(key)
        self._reminder_state = state
        return removed

    def remove(self, key: str) -> bool:
        """Alias for remove_reminder()."""

        return self.remove_reminder(key)

    def _ensure_idle(self) -> None:
        """Protect mutable checkpoint and reminder state from an active session."""

        if self._busy:
            raise ConversationBusyError()


# ---------------------------------------------------------------------------
# State and configuration normalization
# ---------------------------------------------------------------------------


def _normalize_messages(value: object) -> tuple[Message, ...]:
    """Snapshot a homogeneous conversation transcript."""

    if not isinstance(value, Iterable):
        raise TypeError("messages must be an iterable.")
    messages = tuple(cast(Iterable[object], value))
    if not all(isinstance(message, Message) for message in messages):
        raise TypeError("messages must contain only Message instances.")
    return cast(tuple[Message, ...], messages)


def _validate_continuation(value: object) -> None:
    if value is not None and not isinstance(value, ModelContinuation):
        raise TypeError("continuation must be a ModelContinuation or None.")


def _validate_turn_index(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("turn_index must be an int.")
    if value < 0:
        raise ValueError("turn_index must be greater than or equal to zero.")


def _normalize_turn_boundaries(
    value: object,
    *,
    message_count: int,
) -> tuple[int, ...]:
    """Validate exclusive complete-turn offsets against the message snapshot."""

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


def _normalize_instructions(
    value: object,
) -> tuple[SystemInstruction, ...]:
    """Normalize authoring conveniences into unique keyed instructions."""

    if isinstance(value, str):
        instructions = (SystemInstruction.from_text(value),)
    elif isinstance(value, PromptTemplate):
        instructions = (value.instruction(),)
    elif isinstance(value, SystemInstruction):
        instructions = (value,)
    elif isinstance(value, Iterable):
        raw_instructions = tuple(cast(Iterable[object], value))
        if not all(
            isinstance(instruction, (PromptTemplate, SystemInstruction))
            for instruction in raw_instructions
        ):
            raise TypeError(
                "instructions must contain only SystemInstruction or PromptTemplate instances."
            )
        instructions = tuple(
            instruction.instruction()
            if isinstance(instruction, PromptTemplate)
            else cast(SystemInstruction, instruction)
            for instruction in raw_instructions
        )
    else:
        raise TypeError(
            "instructions must be a string, PromptTemplate, SystemInstruction, or iterable."
        )
    keys = [instruction.key for instruction in instructions if instruction.key is not None]
    if len(keys) != len(set(keys)):
        raise ValueError("instructions must have unique non-empty keys.")
    return instructions


def _normalize_reminders(
    value: object,
) -> ReminderState:
    """Normalize authoring conveniences into an immutable reminder state."""

    if isinstance(value, str):
        reminders = (SystemReminder((TextBlock(value),)),)
    elif isinstance(value, PromptTemplate):
        reminders = (value.reminder(),)
    elif isinstance(value, SystemReminder):
        reminders = (value,)
    elif isinstance(value, Iterable):
        raw_reminders = tuple(cast(Iterable[object], value))
        if not all(
            isinstance(reminder, (PromptTemplate, SystemReminder)) for reminder in raw_reminders
        ):
            raise TypeError(
                "reminders must contain only SystemReminder or PromptTemplate instances."
            )
        reminders = tuple(
            reminder.reminder()
            if isinstance(reminder, PromptTemplate)
            else cast(SystemReminder, reminder)
            for reminder in raw_reminders
        )
    else:
        raise TypeError("reminders must be a string, PromptTemplate, SystemReminder, or iterable.")
    return ReminderState(reminders)


def _normalize_tools(value: object) -> ToolSet:
    if isinstance(value, ToolSet):
        return value
    if not isinstance(value, Iterable):
        raise TypeError("tools must be a ToolSet or iterable.")
    return ToolSet(cast(Iterable[ToolLike], value))


def _normalize_cache(
    value: object,
) -> PromptCachePolicy:
    """Normalize shorthand cache modes into a complete policy value."""

    if value is None:
        return PromptCachePolicy()
    if isinstance(value, PromptCachePolicy):
        return value
    try:
        mode = CacheMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported cache mode: {value!r}.") from error
    return PromptCachePolicy(mode=mode)


def _normalize_continuation_policy(
    value: object,
) -> ContinuationPolicy:
    """Normalize a policy enum or its serialized string value."""

    try:
        return ContinuationPolicy(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported continuation policy: {value!r}.") from error


def _normalize_user_message(value: object) -> Message:
    if isinstance(value, str):
        return Message.user(value)
    if isinstance(value, PromptTemplate):
        return value.message(MessageRole.USER)
    if not isinstance(value, Message):
        raise TypeError("message must be a string or Message, or a PromptTemplate.")
    if value.role is not MessageRole.USER:
        raise ValueError("message role must be user.")
    return value


__all__ = ["Conversation", "ConversationCheckpoint", "ConversationState"]
