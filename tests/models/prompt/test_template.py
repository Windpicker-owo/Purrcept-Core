"""Verify exact-field prompt parsing, immutable binding, rendering, and conversions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core.models import (
    Message,
    MessageRole,
    PromptStability,
    ReminderPlacement,
    ReminderScope,
    SystemInstruction,
    SystemReminder,
    TextBlock,
)
from purrcept_core.models.prompt import (
    PromptDefinitionError,
    PromptRenderError,
    PromptTemplate,
    RenderPolicy,
    header,
    join_blocks,
    optional,
    trim,
)


def _research_template() -> PromptTemplate:
    return PromptTemplate(
        "research",
        "Question: {user.query}\n\n{context.kb}\n\n{context.kb}",
        policies={
            "user.query": trim(),
            "context.kb": join_blocks().then(header("# Context")),
        },
        values={"context.kb": ()},
    )


def test_template_snapshots_definition_and_renders_exact_dotted_fields() -> None:
    policies = {
        "user.query": trim(),
        "context.kb": join_blocks().then(header("# Context")),
    }
    values: dict[str, object] = {"context.kb": ["one", "two"]}
    template = PromptTemplate(
        "research",
        "Literal {{field}} {user.query}\n{context.kb}\n{user.query}",
        policies=policies,
        values=values,
    )
    policies.clear()
    values.clear()

    assert template.fields == ("user.query", "context.kb")
    assert template.render({"user.query": "  why?  "}) == (
        "Literal {field} why?\n# Context\none\n\ntwo\nwhy?"
    )
    assert template.build({"user.query": "why?"}) == (
        "Literal {field} why?\n# Context\none\n\ntwo\nwhy?"
    )
    assert template.policies
    assert template.values == {"context.kb": ["one", "two"]}
    assert not hasattr(template, "__dict__")
    with pytest.raises(FrozenInstanceError):
        template.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        template.values["context.kb"] = "changed"


def test_template_strict_and_non_strict_missing_value_semantics() -> None:
    template = PromptTemplate(
        "missing",
        "{required}|{fallback}|{optional}",
        policies={"fallback": optional("default")},
    )

    with pytest.raises(PromptRenderError, match="required") as caught:
        template.render()

    assert caught.value.template_name == "missing"
    assert caught.value.missing_fields == ("required", "fallback", "optional")
    assert template.render(strict=False) == "|default|"
    with pytest.raises(TypeError, match="strict must be a bool"):
        template.render(strict=cast(bool, 1))


def test_values_merge_immutably_with_explicit_overrides_winning() -> None:
    original = PromptTemplate(
        "greet",
        "{greeting}, {name}!",
        values={"greeting": "Hello", "name": "base"},
    )
    bound = original.with_values({"name": "mapping"}, greeting="Hi")

    assert original.render() == "Hello, base!"
    assert bound.render() == "Hi, mapping!"
    assert bound.render({"name": "value"}, greeting="Welcome") == "Welcome, value!"
    assert bound.get("name") == "mapping"
    assert bound.get("missing", "fallback") == "fallback"
    assert bound.has("greeting") is True
    assert bound.has("missing") is False


def test_partial_rendering_and_value_removal_keep_templates_reusable() -> None:
    template = PromptTemplate(
        "person",
        "{{literal}} {name} is {age}",
        values={"name": "Alice", "age": 30},
    )
    no_age = template.without_values("age", "already-missing")
    empty = template.clear_values()

    assert no_age.render_partial() == "{literal} Alice is {age}"
    assert no_age.build_partial({"age": 31}) == "{literal} Alice is 31"
    assert empty.render_partial() == "{literal} {name} is {age}"
    assert empty.clear_values() is empty


def test_templates_convert_to_all_existing_prompt_channels() -> None:
    template = PromptTemplate("policy", "Use {style}.", values={"style": "citations"})

    instruction = template.instruction(
        key="policy",
        stability=PromptStability.GROWING,
    )
    reminder = template.reminder(
        key="policy",
        scope=ReminderScope.CONVERSATION,
        placement=ReminderPlacement.TAIL,
        priority=5,
    )
    message = template.message(
        MessageRole.USER,
        name="researcher",
        metadata={"source": "template"},
    )

    assert instruction == SystemInstruction.from_text(
        "Use citations.",
        key="policy",
        stability=PromptStability.GROWING,
    )
    assert reminder == SystemReminder(
        (TextBlock("Use citations."),),
        key="policy",
        scope=ReminderScope.CONVERSATION,
        placement=ReminderPlacement.TAIL,
        priority=5,
    )
    assert message == Message.user(
        "Use citations.",
        name="researcher",
        metadata={"source": "template"},
    )
    assert template.instruction(strict=False).content == (TextBlock("Use citations."),)


def test_policy_failures_report_template_and_field_without_being_swallowed() -> None:
    def fail(value: object) -> str:
        del value
        raise RuntimeError("broken")

    template = PromptTemplate(
        "failure",
        "{field}",
        policies={"field": RenderPolicy(fail)},
        values={"field": "value"},
    )
    invalid_result = PromptTemplate(
        "invalid-result",
        "{field}",
        policies={"field": RenderPolicy(cast(object, lambda value: value))},
        values={"field": 1},
    )

    with pytest.raises(PromptRenderError, match="Policy for field") as caught:
        template.render()
    with pytest.raises(PromptRenderError, match="invalid-result") as invalid:
        invalid_result.render()

    assert caught.value.field_name == "field"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert isinstance(invalid.value.__cause__, TypeError)


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: PromptTemplate(cast(str, 1), "text"),
            TypeError,
            "name must be a string",
        ),
        (
            lambda: PromptTemplate("", "text"),
            ValueError,
            "name must not be empty",
        ),
        (
            lambda: PromptTemplate("name", cast(str, 1)),
            TypeError,
            "template must be a string",
        ),
        (
            lambda: PromptTemplate("name", ""),
            ValueError,
            "template must not be empty",
        ),
        (
            lambda: PromptTemplate("name", "{field"),
            PromptDefinitionError,
            "invalid braces",
        ),
        (
            lambda: PromptTemplate("name", "{}"),
            PromptDefinitionError,
            "invalid field",
        ),
        (
            lambda: PromptTemplate("name", "{ field}"),
            PromptDefinitionError,
            "invalid field",
        ),
        (
            lambda: PromptTemplate("name", "{field name}"),
            PromptDefinitionError,
            "invalid field",
        ),
        (
            lambda: PromptTemplate("name", "{field!r}"),
            PromptDefinitionError,
            "RenderPolicy",
        ),
        (
            lambda: PromptTemplate("name", "{field:>10}"),
            PromptDefinitionError,
            "RenderPolicy",
        ),
        (
            lambda: PromptTemplate("name", "{field}", policies=cast(object, [])),
            TypeError,
            "policies must be a mapping",
        ),
        (
            lambda: PromptTemplate(
                "name",
                "{field}",
                policies=cast(dict[str, RenderPolicy], {1: trim()}),
            ),
            TypeError,
            "policies must contain only string keys",
        ),
        (
            lambda: PromptTemplate("name", "{field}", policies={"": trim()}),
            ValueError,
            "policies must not contain empty keys",
        ),
        (
            lambda: PromptTemplate(
                "name",
                "{field}",
                policies=cast(dict[str, RenderPolicy], {"field": object()}),
            ),
            TypeError,
            "only RenderPolicy",
        ),
        (
            lambda: PromptTemplate("name", "{field}", policies={"other": trim()}),
            PromptDefinitionError,
            "not present",
        ),
        (
            lambda: PromptTemplate("name", "{field}", values=cast(object, [])),
            TypeError,
            "values must be a mapping",
        ),
        (
            lambda: PromptTemplate(
                "name",
                "{field}",
                values=cast(dict[str, object], {1: "value"}),
            ),
            TypeError,
            "values must contain only string keys",
        ),
        (
            lambda: PromptTemplate("name", "{field}", values={"": "value"}),
            ValueError,
            "values must not contain empty keys",
        ),
        (
            lambda: PromptTemplate("name", "{field}", values={"other": "value"}),
            PromptDefinitionError,
            "not present",
        ),
    ],
)
def test_template_definitions_validate_authoring_errors(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_rendering_and_binding_reject_unknown_or_invalid_value_maps() -> None:
    template = PromptTemplate("name", "{field}")

    with pytest.raises(TypeError, match="values must be a mapping"):
        template.render(cast(object, []))
    with pytest.raises(PromptRenderError, match="not present"):
        template.render({"other": "value"})
    with pytest.raises(PromptRenderError, match="not present"):
        template.render(other="value")
    with pytest.raises(TypeError, match="values must contain only string keys"):
        template.with_values(cast(dict[str, object], {1: "value"}))
    with pytest.raises(PromptDefinitionError, match="not present"):
        template.with_values(other="value")


@pytest.mark.parametrize("method_name", ["get", "has"])
def test_template_lookup_methods_validate_keys(method_name: str) -> None:
    template = _research_template()
    method = getattr(template, method_name)

    with pytest.raises(TypeError, match="key must be a string"):
        method(1)
    with pytest.raises(ValueError, match="key must not be empty"):
        method("")


def test_without_values_validates_keys() -> None:
    template = _research_template()

    with pytest.raises(TypeError, match="key must be a string"):
        template.without_values(cast(str, 1))
