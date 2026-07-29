"""Verify prompt libraries are immutable, ordered, local, and duplicate-free."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import purrcept_core
import purrcept_core.models as models
import purrcept_core.models.prompt as prompt_api
from purrcept_core.models.prompt import (
    PromptDefinitionError,
    PromptLibrary,
    PromptTemplate,
    UnknownPromptError,
)


def _template(name: str = "greet") -> PromptTemplate:
    return PromptTemplate(name, "Hello {name}")


def test_prompt_library_is_a_local_immutable_ordered_registry() -> None:
    greet = _template()
    farewell = _template("farewell")
    source = [greet, farewell]
    library = PromptLibrary(source)
    source.clear()

    assert len(library) == 2
    assert tuple(library) == (greet, farewell)
    assert tuple(library.templates) == ("greet", "farewell")
    assert library.get("greet") is greet
    assert library.get("missing") is None
    assert library.resolve("farewell") is farewell
    assert not hasattr(library, "__dict__")
    with pytest.raises(TypeError):
        library.templates["new"] = greet
    with pytest.raises(FrozenInstanceError):
        library._templates = {}  # type: ignore[misc]


@pytest.mark.parametrize("value", [1, "template", b"template"])
def test_library_rejects_invalid_construction_inputs(value: object) -> None:
    with pytest.raises(TypeError, match="templates must be an iterable"):
        PromptLibrary(cast(object, value))


def test_library_rejects_invalid_items_and_duplicate_names() -> None:
    with pytest.raises(TypeError, match="only PromptTemplate"):
        PromptLibrary((cast(PromptTemplate, object()),))
    with pytest.raises(PromptDefinitionError, match="registered more than once") as caught:
        PromptLibrary((_template(), _template()))

    assert caught.value.template_name == "greet"


def test_library_resolution_validates_and_reports_names() -> None:
    library = PromptLibrary()

    with pytest.raises(TypeError, match="name must be a string"):
        library.get(cast(str, 1))
    with pytest.raises(UnknownPromptError, match="missing"):
        library.resolve("missing")
    with pytest.raises(ValueError, match="name must not be empty"):
        library.resolve("")


def test_prompt_subpackage_is_complete_and_aggregated_only_by_models() -> None:
    expected = {
        "CompiledPrompt",
        "PromptCompileError",
        "PromptCompiled",
        "PromptCompiler",
        "PromptDefinitionError",
        "PromptDiagnostics",
        "PromptDraft",
        "PromptError",
        "PromptLibrary",
        "PromptRenderError",
        "PromptTemplate",
        "PromptTraceEvent",
        "PromptTraceSink",
        "PromptTransform",
        "RenderPolicy",
        "ReminderConsumed",
        "ReminderSourceFailed",
        "ReminderSourceResolved",
        "UnknownPromptError",
        "header",
        "is_effectively_empty",
        "join_blocks",
        "min_len",
        "optional",
        "trim",
        "wrap",
    }

    assert set(prompt_api.__all__) == expected
    assert expected <= set(models.__all__)
    assert all(hasattr(prompt_api, name) for name in prompt_api.__all__)
    assert models.PromptTemplate is PromptTemplate
    assert not hasattr(purrcept_core, "PromptTemplate")
