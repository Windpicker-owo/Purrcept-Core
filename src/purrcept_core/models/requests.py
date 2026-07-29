"""Assemble complete immutable inputs for one model-backend call.

``ToolSpec`` exposes a model-facing JSON Schema, while ``ModelRequest`` combines
history, prompt-control state, generation settings, caching, continuation, and
transport metadata. Construction snapshots every iterable and JSON mapping and
enforces cross-field invariants before a backend receives the request. Provider
serialization remains outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from ._utils import (
    JsonValue,
    empty_json_object,
    freeze_json_object,
    require_non_empty_string,
    validate_optional_non_empty_string,
)
from .caching import PromptCachePolicy
from .continuation import ModelContinuation
from .instructions import SystemInstruction
from .messages import Message
from .reminders import SystemReminder
from .settings import ModelSettings, ToolChoice

ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool exposed to a model, described by a JSON Schema object."""

    name: str
    description: str | None = field(default=None, kw_only=True)
    parameters: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        require_non_empty_string(self.name, field_name="name")
        validate_optional_non_empty_string(self.description, field_name="description")
        object.__setattr__(
            self,
            "parameters",
            freeze_json_object(self.parameters, field_name="parameters"),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A validated provider-neutral prompt and its transport hints.

    At least one message, instruction, or reminder is required. Instruction and
    reminder keys and tool names are unique, and a required tool choice cannot
    be paired with an empty tool set. All nested configuration values are
    immutable snapshots.
    """

    messages: tuple[Message, ...]
    instructions: tuple[SystemInstruction, ...] = field(default=(), kw_only=True)
    reminders: tuple[SystemReminder, ...] = field(default=(), kw_only=True)
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
        messages = _normalize_messages(self.messages)
        instructions = _normalize_instructions(self.instructions)
        reminders = _normalize_reminders(self.reminders)
        tools = _normalize_tools(self.tools)
        if not (messages or instructions or reminders):
            raise ValueError(
                "at least one message, system instruction, or system reminder is required."
            )
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tools must have unique names.")
        _validate_instance(self.settings, ModelSettings, field_name="settings")
        _validate_unique_keys(instructions, field_name="instructions")
        _validate_unique_keys(reminders, field_name="reminders")
        if self.settings.tool_choice is ToolChoice.REQUIRED and not tools:
            raise ValueError("tools must not be empty when tool_choice is required.")
        _validate_instance(self.cache, PromptCachePolicy, field_name="cache")
        if self.continuation is not None:
            _validate_instance(
                self.continuation,
                ModelContinuation,
                field_name="continuation",
            )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "reminders", reminders)
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


def _normalize_messages(value: object) -> tuple[Message, ...]:
    return _normalize_instances(
        value,
        Message,
        field_name="messages",
        item_name="Message",
    )


def _normalize_instructions(value: object) -> tuple[SystemInstruction, ...]:
    return _normalize_instances(
        value,
        SystemInstruction,
        field_name="instructions",
        item_name="SystemInstruction",
    )


def _normalize_reminders(value: object) -> tuple[SystemReminder, ...]:
    return _normalize_instances(
        value,
        SystemReminder,
        field_name="reminders",
        item_name="SystemReminder",
    )


def _normalize_tools(value: object) -> tuple[ToolSpec, ...]:
    return _normalize_instances(
        value,
        ToolSpec,
        field_name="tools",
        item_name="ToolSpec",
    )


def _normalize_instances(
    value: object,
    expected_type: type[ValueT],
    *,
    field_name: str,
    item_name: str,
) -> tuple[ValueT, ...]:
    """Snapshot a homogeneous iterable for use in an immutable request."""

    if not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an iterable.")
    items = tuple(cast(Iterable[object], value))
    if not all(isinstance(item, expected_type) for item in items):
        raise TypeError(f"{field_name} must contain only {item_name} instances.")
    return cast(tuple[ValueT, ...], items)


def _validate_instance(value: object, expected_type: type[object], *, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}.")


def _validate_unique_keys(
    values: tuple[SystemInstruction, ...] | tuple[SystemReminder, ...],
    *,
    field_name: str,
) -> None:
    keys = [value.key for value in values if value.key is not None]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must have unique non-empty keys.")


__all__ = ["ModelRequest", "ToolSpec"]
