"""Verify structured prompt errors preserve authoring and compilation context."""

from __future__ import annotations

from typing import cast

import pytest

from purrcept_core.models.prompt import (
    PromptDefinitionError,
    PromptError,
    PromptRenderError,
    UnknownPromptError,
)


def test_prompt_errors_preserve_structured_context() -> None:
    definition = PromptDefinitionError("bad definition", template_name="named")
    render = PromptRenderError(
        "missing",
        template_name="named",
        field_name="query",
        missing_fields=("query", "context"),
    )
    unknown = UnknownPromptError("missing")

    assert isinstance(definition, PromptError)
    assert definition.template_name == "named"
    assert render.template_name == "named"
    assert render.field_name == "query"
    assert render.missing_fields == ("query", "context")
    assert unknown.name == "missing"
    assert "missing" in str(unknown)


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: PromptDefinitionError(""),
            ValueError,
            "message must not be empty",
        ),
        (
            lambda: PromptDefinitionError("bad", template_name=cast(str, 1)),
            TypeError,
            "template_name must be a string",
        ),
        (
            lambda: PromptDefinitionError("bad", template_name=""),
            ValueError,
            "template_name must not be empty",
        ),
        (
            lambda: PromptRenderError("", template_name="named"),
            ValueError,
            "message must not be empty",
        ),
        (
            lambda: PromptRenderError("bad", template_name=""),
            ValueError,
            "template_name must not be empty",
        ),
        (
            lambda: PromptRenderError(
                "bad",
                template_name="named",
                field_name=cast(str, 1),
            ),
            TypeError,
            "field_name must be a string",
        ),
        (
            lambda: PromptRenderError(
                "bad",
                template_name="named",
                field_name="",
            ),
            ValueError,
            "field_name must not be empty",
        ),
        (
            lambda: PromptRenderError(
                "bad",
                template_name="named",
                missing_fields="field",
            ),
            TypeError,
            "missing_fields must be an iterable",
        ),
        (
            lambda: PromptRenderError(
                "bad",
                template_name="named",
                missing_fields=cast(tuple[str, ...], (1,)),
            ),
            TypeError,
            "missing_fields must contain only strings",
        ),
        (
            lambda: PromptRenderError(
                "bad",
                template_name="named",
                missing_fields=("",),
            ),
            ValueError,
            "must not contain empty strings",
        ),
        (
            lambda: UnknownPromptError(""),
            ValueError,
            "name must not be empty",
        ),
    ],
)
def test_prompt_errors_validate_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
