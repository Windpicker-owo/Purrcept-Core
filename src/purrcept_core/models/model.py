"""Bind a provider backend and model name into an explicit immutable handle.

:class:`Model` is the application-selected bridge from provider-neutral core
objects to a concrete backend. It creates low-level ``Generate`` effects and
configured ``Conversation`` facades, but it does not own backend resources or
mutable conversation history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._utils import JsonValue, require_non_empty_string
from .backend import ModelBackend
from .requests import ModelRequest

if TYPE_CHECKING:
    from .caching import CacheMode, PromptCachePolicy
    from .context_policy import ContextPolicy, RequestTokenBudget
    from .continuation import ContinuationPolicy
    from .conversation import Conversation, ConversationState
    from .effects import Generate
    from .instructions import SystemInstruction
    from .prompt import PromptCompiler, PromptTemplate, PromptTraceSink
    from .reminder_sources import ReminderSource
    from .reminders import SystemReminder
    from .settings import ModelSettings
    from .tools import ToolLike, ToolSet


@dataclass(frozen=True, slots=True)
class Model:
    """An immutable model name bound to a runtime-owned backend.

    Applications or dependency-injection code construct this value. The same
    backend may be shared by many models and conversations, so the backend owns
    resource lifetime and concurrency policy.
    """

    backend: ModelBackend = field(repr=False)
    name: str

    def __post_init__(self) -> None:
        _require_model_backend(self.backend)
        require_non_empty_string(self.name, field_name="name")

    def generate(self, request: ModelRequest) -> Generate:
        """Create one low-level generation effect without starting I/O."""

        from .effects import Generate

        return Generate(self, request)

    def conversation(
        self,
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
        continuation_policy: ContinuationPolicy | str = "client_managed",
        context_policy: ContextPolicy | None = None,
        reminder_sources: Iterable[ReminderSource] = (),
        prompt_compiler: PromptCompiler | None = None,
        request_budget: RequestTokenBudget | None = None,
        prompt_trace_sink: PromptTraceSink | None = None,
        max_model_rounds: int = 8,
        metadata: Mapping[str, JsonValue] | None = None,
        provider_options: Mapping[str, JsonValue] | None = None,
        state: ConversationState | None = None,
    ) -> Conversation:
        """Create an independent mutable conversation facade for this model.

        Configuration is validated and snapshotted by ``Conversation``. The
        returned facade owns only its committed history and reminder state; it
        continues to share this model and its backend.
        """

        from .conversation import Conversation

        return Conversation(
            self,
            instructions=instructions,
            reminders=reminders,
            tools=tools,
            settings=settings,
            cache=cache,
            continuation_policy=continuation_policy,
            context_policy=context_policy,
            reminder_sources=reminder_sources,
            prompt_compiler=prompt_compiler,
            request_budget=request_budget,
            prompt_trace_sink=prompt_trace_sink,
            max_model_rounds=max_model_rounds,
            metadata=metadata,
            provider_options=provider_options,
            state=state,
        )


def _require_model_backend(value: object) -> ModelBackend:
    """Validate the structural provider boundary at model construction."""

    if not isinstance(value, ModelBackend):
        raise TypeError("backend must implement ModelBackend.")
    return value


__all__ = ["Model"]
