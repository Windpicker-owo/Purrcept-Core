"""Verify generation settings normalize provider-neutral values and reject ambiguity."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.settings import ModelSettings, ToolChoice


def test_settings_normalize_and_snapshot_generation_controls() -> None:
    stops = ["END"]

    settings = ModelSettings(
        temperature=0,
        max_output_tokens=100,
        stop_sequences=stops,  # type: ignore[arg-type]
        tool_choice="required",  # type: ignore[arg-type]
        parallel_tool_calls=True,
    )
    stops.append("changed")

    assert settings.temperature == 0.0
    assert settings.max_output_tokens == 100
    assert settings.stop_sequences == ("END",)
    assert settings.tool_choice is ToolChoice.REQUIRED
    assert settings.parallel_tool_calls is True
    with pytest.raises(FrozenInstanceError):
        settings.temperature = 1.0  # type: ignore[misc]


def test_settings_have_provider_neutral_defaults() -> None:
    settings = ModelSettings()

    assert settings.temperature is None
    assert settings.max_output_tokens is None
    assert settings.stop_sequences == ()
    assert settings.tool_choice is ToolChoice.AUTO
    assert settings.parallel_tool_calls is None
    assert {choice.value for choice in ToolChoice} == {"auto", "none", "required"}


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ModelSettings(temperature=True),  # type: ignore[arg-type]
            TypeError,
            "temperature must be a number",
        ),
        (
            lambda: ModelSettings(temperature="warm"),  # type: ignore[arg-type]
            TypeError,
            "temperature must be a number",
        ),
        (
            lambda: ModelSettings(temperature=float("inf")),
            ValueError,
            "temperature must be finite",
        ),
        (
            lambda: ModelSettings(temperature=-0.1),
            ValueError,
            "greater than or equal",
        ),
        (
            lambda: ModelSettings(max_output_tokens=True),  # type: ignore[arg-type]
            TypeError,
            "max_output_tokens must be an int",
        ),
        (
            lambda: ModelSettings(max_output_tokens="many"),  # type: ignore[arg-type]
            TypeError,
            "max_output_tokens must be an int",
        ),
        (
            lambda: ModelSettings(max_output_tokens=0),
            ValueError,
            "greater than zero",
        ),
        (
            lambda: ModelSettings(stop_sequences=1),  # type: ignore[arg-type]
            TypeError,
            "stop_sequences must be an iterable",
        ),
        (
            lambda: ModelSettings(stop_sequences=(1,)),  # type: ignore[arg-type]
            TypeError,
            "only strings",
        ),
        (
            lambda: ModelSettings(stop_sequences=("",)),
            ValueError,
            "empty strings",
        ),
        (
            lambda: ModelSettings(tool_choice="sometimes"),  # type: ignore[arg-type]
            ValueError,
            "Unsupported tool choice",
        ),
        (
            lambda: ModelSettings(parallel_tool_calls=1),  # type: ignore[arg-type]
            TypeError,
            "parallel_tool_calls must be a bool or None",
        ),
    ],
)
def test_settings_validate_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
