"""Compile history and prompt-control inputs into deterministic model requests.

``PromptCompiler`` is a pure boundary: it merges stored, turn, and dynamic
reminders, validates stable-key uniqueness, sorts by priority without
disturbing equal-priority order, and builds diagnostics without retaining raw
prompt text. ``TurnPromptSession`` owns the stateful lifecycle around that pure
step, including dynamic-source resolution, full-request budget trimming,
request-scoped reminder consumption, and trace emission.

Compilation performs no effects or asynchronous work. A request is considered
accepted only after it fits the configured budget and its compiled trace event
has been delivered; only then may ``NEXT_REQUEST`` reminders be consumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import TypeVar, cast

from .._utils import (
    JsonValue,
    empty_json_object,
    freeze_json_object,
)
from ..caching import PromptCachePolicy
from ..context_policy import (
    ContextBudgetExceededError,
    PreparedContext,
    RequestTokenBudget,
)
from ..continuation import ModelContinuation
from ..instructions import SystemInstruction
from ..reminder_sources import (
    ReminderResolutionContext,
    ReminderSource,
    normalize_reminder_sources,
    resolve_reminder_source,
)
from ..reminders import (
    ReminderPlacement,
    ReminderScope,
    ReminderState,
    SystemReminder,
)
from ..requests import ModelRequest, ToolSpec
from ..settings import ModelSettings
from .errors import PromptCompileError
from .tracing import (
    PromptCompiled,
    PromptTraceSink,
    ReminderConsumed,
    ReminderSourceResolved,
    emit_prompt_trace,
    source_failure_event,
)

ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class PromptDraft:
    """Immutable non-reminder inputs required to compile one model request."""

    history: PreparedContext
    instructions: tuple[SystemInstruction, ...] = field(default=(), kw_only=True)
    tools: tuple[ToolSpec, ...] = field(default=(), kw_only=True)
    settings: ModelSettings = field(default_factory=ModelSettings, kw_only=True)
    cache: PromptCachePolicy = field(default_factory=PromptCachePolicy, kw_only=True)
    continuation: ModelContinuation | None = field(default=None, kw_only=True)
    metadata: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )
    provider_options: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.history), PreparedContext):
            raise TypeError("history must be a PreparedContext.")
        instructions = _normalize_instances(
            self.instructions,
            SystemInstruction,
            field_name="instructions",
        )
        tools = _normalize_instances(self.tools, ToolSpec, field_name="tools")
        if not isinstance(cast(object, self.settings), ModelSettings):
            raise TypeError("settings must be a ModelSettings.")
        if not isinstance(cast(object, self.cache), PromptCachePolicy):
            raise TypeError("cache must be a PromptCachePolicy.")
        if self.continuation is not None and not isinstance(
            cast(object, self.continuation),
            ModelContinuation,
        ):
            raise TypeError("continuation must be a ModelContinuation or None.")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(self.metadata, field_name="metadata"),
        )
        object.__setattr__(
            self,
            "provider_options",
            freeze_json_object(self.provider_options, field_name="provider_options"),
        )


@dataclass(frozen=True, slots=True)
class PromptDiagnostics:
    """Metadata describing the effective prompt without retaining raw text.

    Fingerprints and content hashes support correlation and change detection;
    they are diagnostic values, not canonical serialization or security
    boundaries.
    """

    model_round: int
    reminder_keys: tuple[str | None, ...]
    reminder_scopes: tuple[ReminderScope | None, ...]
    reminder_placements: tuple[ReminderPlacement, ...]
    reminder_priorities: tuple[int, ...]
    content_hashes: tuple[str, ...]
    prompt_fingerprint: str
    dynamic_source_ids: tuple[str, ...] = field(default=(), kw_only=True)
    estimated_tokens: int | None = field(default=None, kw_only=True)
    trimmed_turns: int = field(default=0, kw_only=True)


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """One immutable request plus diagnostics and pending consumption metadata."""

    request: ModelRequest
    diagnostics: PromptDiagnostics
    consumed_next_request: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.request), ModelRequest):
            raise TypeError("request must be a ModelRequest.")
        if not isinstance(cast(object, self.diagnostics), PromptDiagnostics):
            raise TypeError("diagnostics must be PromptDiagnostics.")
        if not isinstance(cast(object, self.consumed_next_request), bool):
            raise TypeError("consumed_next_request must be a bool.")


@dataclass(frozen=True, slots=True)
class PromptCompiler:
    """Pure compiler from an explicit draft and effective reminders.

    Prompt sessions or advanced callers create a compiler and invoke
    :meth:`compile` synchronously. The compiler retains no state between calls.
    Inputs are already immutable snapshots; conflicts raise
    :class:`PromptCompileError` before a request escapes.
    """

    def compile(
        self,
        draft: PromptDraft,
        *,
        stored_reminders: tuple[SystemReminder, ...] = (),
        turn_reminders: tuple[SystemReminder, ...] = (),
        dynamic_reminders: tuple[SystemReminder, ...] = (),
        dynamic_source_ids: tuple[str, ...] = (),
        model_round: int,
    ) -> CompiledPrompt:
        """Compile, sort, and validate one provider-neutral request."""

        if not isinstance(cast(object, draft), PromptDraft):
            raise TypeError("draft must be a PromptDraft.")
        _validate_positive_int(model_round, field_name="model_round")
        stored = _normalize_reminders(stored_reminders, field_name="stored_reminders")
        turn = _normalize_reminders(turn_reminders, field_name="turn_reminders")
        dynamic = _normalize_reminders(dynamic_reminders, field_name="dynamic_reminders")
        source_ids = _normalize_source_ids(dynamic_source_ids)

        # Stored and turn reminders retain their declared scope. Dynamic
        # reminders are request-local, so diagnostics record no stored scope for
        # them even if the returned value carries a default scope.
        entries = [
            *((reminder, reminder.scope) for reminder in stored),
            *((reminder, reminder.scope) for reminder in turn),
            *((reminder, None) for reminder in dynamic),
        ]
        _validate_unique_reminder_keys(tuple(reminder for reminder, _ in entries))
        # Ordering: Python's stable sort preserves source order for equal
        # priorities, making stored/turn/dynamic registration order observable
        # and deterministic without a second tie-break key.
        entries.sort(key=lambda entry: -entry[0].priority)
        reminders = tuple(reminder for reminder, _ in entries)
        scopes = tuple(scope for _, scope in entries)

        request = ModelRequest(
            draft.history.messages,
            instructions=draft.instructions,
            reminders=reminders,
            tools=draft.tools,
            settings=draft.settings,
            cache=draft.cache,
            continuation=draft.continuation,
            metadata=draft.metadata,
            provider_options=draft.provider_options,
        )
        diagnostics = PromptDiagnostics(
            model_round,
            tuple(reminder.key for reminder in reminders),
            scopes,
            tuple(reminder.placement for reminder in reminders),
            tuple(reminder.priority for reminder in reminders),
            tuple(_reminder_hash(reminder) for reminder in reminders),
            _request_hash(request),
            dynamic_source_ids=source_ids,
        )
        return CompiledPrompt(
            request,
            diagnostics,
            consumed_next_request=any(
                reminder.scope is ReminderScope.NEXT_REQUEST for reminder in (*stored, *turn)
            ),
        )


class TurnPromptSession:
    """Own prompt-control state acquired by exactly one active conversation turn.

    ``Conversation`` creates a session after acquiring its busy flag and reads
    :attr:`remaining_state` during ``finally`` cleanup. TURN reminders are
    removed from that remaining state immediately but remain active throughout
    this session. NEXT_REQUEST reminders remain active until the first request
    is accepted. Conversation-scoped reminders survive every compilation.

    Dynamic sources are resolved for every candidate request and are never
    stored. A request-budget retry may therefore resolve a source more than once
    with progressively trimmed prepared context. Source, compiler, counter, and
    trace failures abort compilation and propagate to the owning turn.
    """

    __slots__ = (
        "_base_reminders",
        "_compiler",
        "_remaining_state",
        "_sources",
        "_trace_sink",
    )

    def __init__(
        self,
        state: ReminderState,
        *,
        compiler: PromptCompiler,
        reminder_sources: Iterable[ReminderSource] = (),
        trace_sink: PromptTraceSink | None = None,
    ) -> None:
        if not isinstance(cast(object, state), ReminderState):
            raise TypeError("state must be a ReminderState.")
        if not isinstance(cast(object, compiler), PromptCompiler):
            raise TypeError("compiler must be a PromptCompiler.")
        # Lifecycle: TURN reminders are acquired for this session at creation.
        # They stay in `_base_reminders` for all model rounds but are omitted
        # from the state returned to the facade, even if the turn later fails.
        self._base_reminders = state.reminders
        self._remaining_state = ReminderState(
            tuple(reminder for reminder in state if reminder.scope is not ReminderScope.TURN)
        )
        self._compiler = compiler
        self._sources = normalize_reminder_sources(reminder_sources)
        self._trace_sink = trace_sink

    @classmethod
    def begin(
        cls,
        state: ReminderState,
        *,
        compiler: PromptCompiler | None = None,
        reminder_sources: Iterable[ReminderSource] = (),
        trace_sink: PromptTraceSink | None = None,
    ) -> TurnPromptSession:
        """Acquire TURN reminders and start one isolated prompt-control session."""

        return cls(
            state,
            compiler=PromptCompiler() if compiler is None else compiler,
            reminder_sources=reminder_sources,
            trace_sink=trace_sink,
        )

    @property
    def remaining_state(self) -> ReminderState:
        """Return state that survives when the active turn ends now."""

        return self._remaining_state

    def compile(
        self,
        draft: PromptDraft,
        *,
        context: ReminderResolutionContext,
        model_round: int,
        request_budget: RequestTokenBudget | None = None,
    ) -> CompiledPrompt:
        """Compile one accepted request, trimming only old complete turns.

        Budget retries rebuild the full request because instructions, dynamic
        reminders, tools, and provider framing may make token counts
        non-additive. The current incomplete turn is never removed.
        """

        if not isinstance(cast(object, context), ReminderResolutionContext):
            raise TypeError("context must be a ReminderResolutionContext.")
        if request_budget is not None and not isinstance(
            cast(object, request_budget),
            RequestTokenBudget,
        ):
            raise TypeError("request_budget must be a RequestTokenBudget or None.")

        candidate = draft
        trimmed_turns = 0
        while True:
            # Dynamic sources must see the exact prepared history being tested.
            # Re-resolve them after every trim rather than carrying potentially
            # stale projections into the next candidate.
            resolution_context = replace(
                context,
                prepared_context=candidate.history,
                model_round=model_round,
            )
            dynamic, source_ids = self._resolve_sources(
                resolution_context,
                model_round=model_round,
            )
            compiled = self._compiler.compile(
                candidate,
                stored_reminders=self._active_reminders(),
                dynamic_reminders=dynamic,
                dynamic_source_ids=source_ids,
                model_round=model_round,
            )
            estimated_tokens: int | None = None
            if request_budget is not None:
                estimated_tokens = request_budget.count(compiled.request, context.model)
                if estimated_tokens > request_budget.max_tokens:
                    trimmed_history = _drop_oldest_complete_turn(candidate.history)
                    if trimmed_history is None:
                        raise ContextBudgetExceededError(
                            request_budget.max_tokens,
                            estimated_tokens,
                        )
                    candidate = replace(candidate, history=trimmed_history)
                    trimmed_turns += 1
                    continue

            # Ordering: trace the accepted request before consuming reminders.
            # If the synchronous trace sink fails, no request is returned to the
            # model and NEXT_REQUEST state remains available for a later attempt.
            diagnostics = replace(
                compiled.diagnostics,
                estimated_tokens=estimated_tokens,
                trimmed_turns=trimmed_turns,
            )
            accepted = replace(compiled, diagnostics=diagnostics)
            self._emit_compiled(accepted)
            if accepted.consumed_next_request:
                consumed = tuple(
                    reminder
                    for reminder in self._remaining_state
                    if reminder.scope is ReminderScope.NEXT_REQUEST
                )
                emit_prompt_trace(
                    self._trace_sink,
                    ReminderConsumed(
                        model_round,
                        tuple(reminder.key for reminder in consumed),
                    ),
                )
                self._remaining_state = self._remaining_state.clear(ReminderScope.NEXT_REQUEST)
            return accepted

    def _active_reminders(self) -> tuple[SystemReminder, ...]:
        """Return stored reminders still active for the next candidate request."""

        next_request_active = any(
            reminder.scope is ReminderScope.NEXT_REQUEST for reminder in self._remaining_state
        )
        return tuple(
            reminder
            for reminder in self._base_reminders
            if reminder.scope is not ReminderScope.NEXT_REQUEST or next_request_active
        )

    def _resolve_sources(
        self,
        context: ReminderResolutionContext,
        *,
        model_round: int,
    ) -> tuple[tuple[SystemReminder, ...], tuple[str, ...]]:
        """Resolve dynamic sources in registration order with failure tracing."""

        reminders: list[SystemReminder] = []
        source_ids: list[str] = []
        for source in self._sources:
            source_id = source.source_id
            try:
                resolved = resolve_reminder_source(source, context)
            except Exception as error:
                emit_prompt_trace(
                    self._trace_sink,
                    source_failure_event(
                        model_round=model_round,
                        source_id=source_id,
                        error=error,
                    ),
                )
                raise
            reminders.extend(resolved)
            source_ids.append(source_id)
            emit_prompt_trace(
                self._trace_sink,
                ReminderSourceResolved(
                    model_round,
                    source_id,
                    tuple(reminder.key for reminder in resolved),
                ),
            )
        return tuple(reminders), tuple(source_ids)

    def _emit_compiled(self, compiled: CompiledPrompt) -> None:
        """Project compiled diagnostics into the public trace event shape."""

        diagnostics = compiled.diagnostics
        emit_prompt_trace(
            self._trace_sink,
            PromptCompiled(
                diagnostics.model_round,
                diagnostics.prompt_fingerprint,
                diagnostics.reminder_keys,
                diagnostics.reminder_scopes,
                diagnostics.reminder_placements,
                diagnostics.reminder_priorities,
                diagnostics.content_hashes,
                dynamic_source_ids=diagnostics.dynamic_source_ids,
                estimated_tokens=diagnostics.estimated_tokens,
                trimmed_turns=diagnostics.trimmed_turns,
            ),
        )


def _normalize_instances(
    value: object,
    expected_type: type[ValueT],
    *,
    field_name: str,
) -> tuple[ValueT, ...]:
    """Snapshot a homogeneous iterable used by a prompt draft."""

    if not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an iterable.")
    items = tuple(cast(Iterable[object], value))
    if not all(isinstance(item, expected_type) for item in items):
        raise TypeError(f"{field_name} must contain only {expected_type.__name__} instances.")
    return cast(tuple[ValueT, ...], items)


def _normalize_reminders(
    value: object,
    *,
    field_name: str,
) -> tuple[SystemReminder, ...]:
    return _normalize_instances(value, SystemReminder, field_name=field_name)


def _normalize_source_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable):
        raise TypeError("dynamic_source_ids must be an iterable.")
    source_ids = tuple(cast(Iterable[object], value))
    if not all(isinstance(source_id, str) for source_id in source_ids):
        raise TypeError("dynamic_source_ids must contain only strings.")
    if any(not source_id for source_id in cast(tuple[str, ...], source_ids)):
        raise ValueError("dynamic_source_ids must not contain empty strings.")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("dynamic_source_ids must be unique.")
    return cast(tuple[str, ...], source_ids)


def _validate_unique_reminder_keys(reminders: tuple[SystemReminder, ...]) -> None:
    """Reject ambiguous stable reminder identities across all input sources."""

    seen: set[str] = set()
    for reminder in reminders:
        key = reminder.key
        if key is None:
            continue
        if key in seen:
            raise PromptCompileError(
                f"Reminder key {key!r} appears more than once in the compiled prompt.",
                reminder_key=key,
            )
        seen.add(key)


def _drop_oldest_complete_turn(history: PreparedContext) -> PreparedContext | None:
    """Return history without its oldest complete turn, preserving the suffix."""

    if not history.turn_boundaries:
        return None
    removed_messages = history.turn_boundaries[0]
    return PreparedContext(
        history.messages[removed_messages:],
        continuation=history.continuation,
        turn_boundaries=tuple(
            boundary - removed_messages for boundary in history.turn_boundaries[1:]
        ),
    )


def _reminder_hash(reminder: SystemReminder) -> str:
    return sha256(repr(reminder.content).encode("utf-8")).hexdigest()


def _request_hash(request: ModelRequest) -> str:
    return sha256(repr(request).encode("utf-8")).hexdigest()


def _validate_positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")


__all__ = [
    "CompiledPrompt",
    "PromptCompiler",
    "PromptDiagnostics",
    "PromptDraft",
]
