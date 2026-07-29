"""Verify prompt render-policy composition, empty semantics, and validation."""

from __future__ import annotations

from typing import cast

import pytest

from purrcept_core.models.prompt import (
    RenderPolicy,
    header,
    is_effectively_empty,
    join_blocks,
    min_len,
    optional,
    trim,
    wrap,
)


class WhitespaceValue:
    def __str__(self) -> str:
        return "   "


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        (" \n", True),
        ("text", False),
        ({}, True),
        ({"key": "value"}, False),
        ([], True),
        ([1], False),
        (set(), True),
        ({"value"}, False),
        (0, False),
    ],
)
def test_effective_empty_semantics(value: object, expected: bool) -> None:
    assert is_effectively_empty(value) is expected


def test_render_policy_is_frozen_composable_and_checks_custom_results() -> None:
    policy = trim().then(header("Title"))

    assert policy("  body  ") == "Title\nbody"
    with pytest.raises(TypeError, match="other must be a RenderPolicy"):
        policy.then(cast(RenderPolicy, object()))
    with pytest.raises(TypeError, match="must return a string"):
        RenderPolicy(cast(object, lambda value: value))(1)
    with pytest.raises(TypeError, match="transform must be callable"):
        RenderPolicy(cast(object, 1))


def test_optional_and_trim_render_empty_and_present_values() -> None:
    fallback = optional("missing")

    assert fallback(None) == "missing"
    assert fallback([]) == "missing"
    assert fallback(0) == "0"
    assert trim()(None) == ""
    assert trim()("  body  ") == "body"
    with pytest.raises(TypeError, match="empty must be a string"):
        optional(cast(str, 1))


def test_header_and_wrap_only_decorate_effective_content() -> None:
    titled = header("# Context", separator=": ")
    fenced = wrap("<context>", "</context>")

    assert titled(None) == ""
    assert titled(WhitespaceValue()) == ""
    assert titled("body") == "# Context: body"
    assert fenced(None) == ""
    assert fenced(WhitespaceValue()) == ""
    assert fenced("body") == "<context>body</context>"


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: header(cast(str, 1)), "title must be a string"),
        (lambda: header("title", separator=cast(str, 1)), "separator must be a string"),
        (lambda: wrap(cast(str, 1)), "prefix must be a string"),
        (lambda: wrap(suffix=cast(str, 1)), "suffix must be a string"),
        (lambda: join_blocks(cast(str, 1)), "separator must be a string"),
    ],
)
def test_text_policy_factories_validate_delimiters(
    factory: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        factory()  # type: ignore[operator]


def test_join_blocks_filters_and_joins_sequences() -> None:
    policy = join_blocks(" | ")

    assert policy(None) == ""
    assert policy([" one ", "", None, WhitespaceValue(), "two"]) == "one | two"
    assert policy(("one", "two")) == "one | two"
    assert policy(42) == "42"


def test_min_len_filters_short_values_and_validates_threshold() -> None:
    policy = min_len(3)

    assert policy(None) == ""
    assert policy(" x ") == ""
    assert policy(" xyz ") == " xyz "
    assert min_len(0)("") == ""
    with pytest.raises(TypeError, match="length must be an int"):
        min_len(cast(int, True))
    with pytest.raises(TypeError, match="length must be an int"):
        min_len(cast(int, 1.5))
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        min_len(-1)
