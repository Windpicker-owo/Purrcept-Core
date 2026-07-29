"""Define reusable prompt templates with exact fields and explicit render policy.

``PromptTemplate`` parses a restricted subset of Python format syntax once,
snapshots field policies and default bindings, and can render text or convert it
to model instructions, reminders, and messages. Placeholder names are exact
strings, including dotted names; attribute traversal, conversions, and format
specifiers are deliberately excluded in favor of explicit ``RenderPolicy``
objects.

Template operations return new values rather than mutating definitions. Render
failures are translated to structured prompt errors with template and field
context, while the original policy exception remains chained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Formatter
from types import MappingProxyType
from typing import cast

from .._utils import JsonValue, require_non_empty_string
from ..caching import PromptStability
from ..content import TextBlock
from ..instructions import SystemInstruction
from ..messages import Message, MessageRole
from ..reminders import ReminderPlacement, ReminderScope, SystemReminder
from .errors import PromptDefinitionError, PromptRenderError
from .policies import RenderPolicy, optional


@dataclass(frozen=True, slots=True)
class _TemplatePart:
    literal: str
    field_name: str | None = None


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A reusable prompt definition with snapshotted bindings and field policies.

    Applications construct templates directly or store them in a
    ``PromptLibrary``. Mapping structure is copied and exposed read-only;
    application objects bound as values remain caller-owned and are converted
    only when a policy renders them. Instances are safe to reuse as long as
    those bound objects are treated consistently by the caller.
    """

    name: str
    template: str
    policies: Mapping[str, RenderPolicy] = field(
        default_factory=lambda: cast(Mapping[str, RenderPolicy], {}),
        kw_only=True,
    )
    values: Mapping[str, object] = field(
        default_factory=lambda: cast(Mapping[str, object], {}),
        kw_only=True,
    )
    _parts: tuple[_TemplatePart, ...] = field(init=False, repr=False, compare=False)
    _fields: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_non_empty_string(self.name, field_name="name")
        require_non_empty_string(self.template, field_name="template")
        parts, field_names = _parse_template(self.name, self.template)
        policies = _freeze_policies(self.policies)
        values = _freeze_values(self.values, field_name="values")
        _reject_unknown_keys(
            policies,
            known_fields=field_names,
            template_name=self.name,
            field_name="policies",
        )
        _reject_unknown_keys(
            values,
            known_fields=field_names,
            template_name=self.name,
            field_name="values",
        )
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_parts", parts)
        object.__setattr__(self, "_fields", field_names)

    @property
    def fields(self) -> tuple[str, ...]:
        """Return unique placeholder names in first-appearance order."""

        return self._fields

    def get(self, key: str, default: object = None) -> object:
        """Return one bound default value."""

        _require_key(key)
        return self.values.get(key, default)

    def has(self, key: str) -> bool:
        """Return whether a default value is bound for one field."""

        _require_key(key)
        return key in self.values

    def with_values(
        self,
        values: Mapping[str, object] | None = None,
        /,
        **overrides: object,
    ) -> PromptTemplate:
        """Return a new template with merged default values."""

        merged = self._merge_values(values, overrides, error_type=PromptDefinitionError)
        return PromptTemplate(
            self.name,
            self.template,
            policies=self.policies,
            values=merged,
        )

    def without_values(self, *keys: str) -> PromptTemplate:
        """Return a new template without the selected default values."""

        retained = dict(self.values)
        for key in keys:
            _require_key(key)
            retained.pop(key, None)
        return PromptTemplate(
            self.name,
            self.template,
            policies=self.policies,
            values=retained,
        )

    def clear_values(self) -> PromptTemplate:
        """Return the same template definition without any default values."""

        if not self.values:
            return self
        return PromptTemplate(self.name, self.template, policies=self.policies)

    def render(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        strict: bool = True,
        **overrides: object,
    ) -> str:
        """Render the complete template.

        Strict mode reports every missing field. Non-strict mode sends missing
        values through their policy with ``None`` as the input.
        """

        _validate_strict(strict)
        effective = self._merge_values(values, overrides, error_type=PromptRenderError)
        missing = tuple(field_name for field_name in self._fields if field_name not in effective)
        if strict and missing:
            names = ", ".join(repr(item) for item in missing)
            raise PromptRenderError(
                f"Prompt template {self.name!r} is missing values for: {names}.",
                template_name=self.name,
                missing_fields=missing,
            )
        return self._render_parts(effective, preserve_missing=False)

    def build(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        strict: bool = True,
        **overrides: object,
    ) -> str:
        """Alias for :meth:`render` for users migrating from builder APIs."""

        return self.render(values, strict=strict, **overrides)

    def render_partial(
        self,
        values: Mapping[str, object] | None = None,
        /,
        **overrides: object,
    ) -> str:
        """Render supplied fields while preserving unresolved placeholders."""

        effective = self._merge_values(values, overrides, error_type=PromptRenderError)
        return self._render_parts(effective, preserve_missing=True)

    def build_partial(
        self,
        values: Mapping[str, object] | None = None,
        /,
        **overrides: object,
    ) -> str:
        """Alias for :meth:`render_partial`."""

        return self.render_partial(values, **overrides)

    def instruction(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        strict: bool = True,
        key: str | None = None,
        stability: PromptStability = PromptStability.STABLE,
    ) -> SystemInstruction:
        """Render this template as a system instruction."""

        return SystemInstruction.from_text(
            self.render(values, strict=strict),
            key=key,
            stability=stability,
        )

    def reminder(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        strict: bool = True,
        key: str | None = None,
        scope: ReminderScope = ReminderScope.NEXT_REQUEST,
        placement: ReminderPlacement = ReminderPlacement.AUTO,
        priority: int = 0,
    ) -> SystemReminder:
        """Render this template as a system reminder."""

        return SystemReminder(
            (TextBlock(self.render(values, strict=strict)),),
            key=key,
            scope=scope,
            placement=placement,
            priority=priority,
        )

    def message(
        self,
        role: MessageRole | str,
        values: Mapping[str, object] | None = None,
        /,
        *,
        strict: bool = True,
        name: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Message:
        """Render this template as a conversational message."""

        return Message.from_text(
            role,
            self.render(values, strict=strict),
            name=name,
            metadata=metadata,
        )

    def _merge_values(
        self,
        values: Mapping[str, object] | None,
        overrides: Mapping[str, object],
        *,
        error_type: type[PromptDefinitionError] | type[PromptRenderError],
    ) -> dict[str, object]:
        """Merge defaults, an explicit mapping, and keyword overrides in order.

        Later sources intentionally win. Unknown fields are rejected only after
        the merge so the raised error describes the effective call boundary.
        """

        merged = dict(self.values)
        if values is not None:
            merged.update(_freeze_values(values, field_name="values"))
        merged.update(overrides)
        _reject_unknown_keys(
            merged,
            known_fields=self._fields,
            template_name=self.name,
            field_name="values",
            error_type=error_type,
        )
        return merged

    def _render_parts(
        self,
        values: Mapping[str, object],
        *,
        preserve_missing: bool,
    ) -> str:
        """Render parsed parts while optionally retaining unresolved fields."""

        output: list[str] = []
        default_policy = optional()
        for part in self._parts:
            output.append(part.literal)
            if part.field_name is None:
                continue
            if part.field_name not in values and preserve_missing:
                output.append(f"{{{part.field_name}}}")
                continue
            value = values.get(part.field_name)
            policy = self.policies.get(part.field_name, default_policy)
            try:
                output.append(policy(value))
            except Exception as error:
                # Exception translation adds stable authoring context without
                # hiding the policy's original failure from debuggers.
                raise PromptRenderError(
                    (
                        f"Policy for field {part.field_name!r} failed while rendering "
                        f"prompt template {self.name!r}."
                    ),
                    template_name=self.name,
                    field_name=part.field_name,
                ) from error
        return "".join(output)


def _parse_template(
    name: str,
    template: str,
) -> tuple[tuple[_TemplatePart, ...], tuple[str, ...]]:
    """Parse exact placeholders and reject implicit Python formatting behavior."""

    parts: list[_TemplatePart] = []
    field_names: list[str] = []
    try:
        parsed = Formatter().parse(template)
        for literal, field_name, format_spec, conversion in parsed:
            if field_name is not None:
                _validate_field_name(name, field_name)
                if conversion is not None or format_spec:
                    raise PromptDefinitionError(
                        (
                            f"Prompt template {name!r} does not support format "
                            "specifiers or conversions; use a RenderPolicy instead."
                        ),
                        template_name=name,
                    )
                if field_name not in field_names:
                    field_names.append(field_name)
            parts.append(_TemplatePart(literal, field_name))
    except ValueError as error:
        raise PromptDefinitionError(
            f"Prompt template {name!r} has invalid braces: {error}.",
            template_name=name,
        ) from error
    return tuple(parts), tuple(field_names)


def _validate_field_name(template_name: str, field_name: str) -> None:
    if (
        not field_name
        or field_name.strip() != field_name
        or any(character.isspace() for character in field_name)
    ):
        raise PromptDefinitionError(
            f"Prompt template {template_name!r} contains invalid field {field_name!r}.",
            template_name=template_name,
        )


def _freeze_policies(value: object) -> Mapping[str, RenderPolicy]:
    if not isinstance(value, Mapping):
        raise TypeError("policies must be a mapping.")
    source = cast(Mapping[object, object], value)
    policies: dict[str, RenderPolicy] = {}
    for key, policy in source.items():
        _require_mapping_key(key, field_name="policies")
        if not isinstance(policy, RenderPolicy):
            raise TypeError("policies must contain only RenderPolicy values.")
        policies[cast(str, key)] = policy
    return MappingProxyType(policies)


def _freeze_values(value: object, *, field_name: str) -> Mapping[str, object]:
    """Snapshot binding keys while deliberately retaining caller value objects.

    Prompt values may be rich application objects consumed by custom policies,
    so deep-copying or JSON-normalizing them here would change their semantics.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    source = cast(Mapping[object, object], value)
    values: dict[str, object] = {}
    for key, item in source.items():
        _require_mapping_key(key, field_name=field_name)
        values[cast(str, key)] = item
    return MappingProxyType(values)


def _require_mapping_key(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must contain only string keys.")
    if not value:
        raise ValueError(f"{field_name} must not contain empty keys.")


def _reject_unknown_keys(
    values: Mapping[str, object],
    *,
    known_fields: tuple[str, ...],
    template_name: str,
    field_name: str,
    error_type: type[PromptDefinitionError] | type[PromptRenderError] = PromptDefinitionError,
) -> None:
    """Reject bindings that cannot be consumed by the parsed template."""

    unknown = tuple(key for key in values if key not in known_fields)
    if not unknown:
        return
    names = ", ".join(repr(item) for item in unknown)
    message = (
        f"{field_name} contains fields not present in prompt template {template_name!r}: {names}."
    )
    if error_type is PromptRenderError:
        raise PromptRenderError(message, template_name=template_name)
    raise PromptDefinitionError(message, template_name=template_name)


def _validate_strict(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("strict must be a bool.")


def _require_key(value: object) -> None:
    require_non_empty_string(value, field_name="key")


__all__ = ["PromptTemplate"]
