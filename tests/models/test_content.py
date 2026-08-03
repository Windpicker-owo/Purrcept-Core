"""Verify content blocks and nested JSON arguments are immutable and extensible."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from purrcept_core.models.content import (
    ContentBlock,
    ImageBlock,
    ImageBytes,
    ImageUrl,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


@dataclass(frozen=True, slots=True)
class CustomBlock(ContentBlock):
    value: str


def test_content_blocks_are_extensible_immutable_value_objects() -> None:
    text = TextBlock("hello")
    reasoning = ReasoningBlock("private model reasoning")
    image_url = ImageUrl("https://example.test/cat.png")
    image_bytes = ImageBytes(b"cat", "image/png")
    remote_image = ImageBlock(image_url, alt_text="a cat")
    inline_image = ImageBlock(image_bytes)
    custom = CustomBlock("provider extension")
    result_source = [text, custom]
    result = ToolResultBlock(
        "call-1",
        content=result_source,  # type: ignore[arg-type]
        is_error=True,
    )

    result_source.append(TextBlock("changed"))

    assert remote_image.source is image_url
    assert inline_image.source is image_bytes
    assert result.content == (text, custom)
    assert result.is_error is True
    with pytest.raises(FrozenInstanceError):
        text.text = "changed"  # type: ignore[misc]
    assert reasoning.text == "private model reasoning"


def test_tool_arguments_are_deeply_copied_and_frozen() -> None:
    items: list[object] = [1, 2.5, True, None, "cat"]
    nested: dict[str, object] = {"items": items}
    source: dict[str, object] = {"nested": nested}

    call = ToolCallBlock(
        "call-1",
        "lookup",
        arguments=source,  # type: ignore[arg-type]
    )
    items.append("changed")
    nested["new"] = "changed"
    source["other"] = "changed"

    assert call.arguments == {
        "nested": {"items": (1, 2.5, True, None, "cat")},
    }
    with pytest.raises(TypeError):
        call.arguments["new"] = "value"
    frozen_nested = call.arguments["nested"]
    assert isinstance(frozen_nested, dict | type(call.arguments))
    with pytest.raises(TypeError):
        frozen_nested["new"] = "value"  # type: ignore[index]


def test_json_freezing_allows_reused_non_cyclic_containers() -> None:
    shared = {"value": [1]}

    call = ToolCallBlock(
        "call-1",
        "lookup",
        arguments={"first": shared, "second": shared},
    )

    assert call.arguments == {
        "first": {"value": (1,)},
        "second": {"value": (1,)},
    }


@pytest.mark.parametrize(
    ("arguments", "error_type", "match"),
    [
        ([], TypeError, "must be a mapping"),
        ({1: "bad"}, TypeError, "string object keys"),
        ({"bad": object()}, TypeError, "unsupported JSON value"),
        ({"bad": b"bytes"}, TypeError, "unsupported JSON value"),
        ({"bad": float("inf")}, ValueError, "finite JSON numbers"),
    ],
)
def test_tool_arguments_reject_non_json_values(
    arguments: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        ToolCallBlock(
            "call-1",
            "lookup",
            arguments=arguments,  # type: ignore[arg-type]
        )


def test_json_freezing_rejects_mapping_and_sequence_cycles() -> None:
    mapping_cycle: dict[str, object] = {}
    mapping_cycle["self"] = mapping_cycle
    sequence_cycle: list[object] = []
    sequence_cycle.append(sequence_cycle)

    with pytest.raises(ValueError, match="reference cycle"):
        ToolCallBlock(
            "call-1",
            "lookup",
            arguments=mapping_cycle,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="reference cycle"):
        ToolCallBlock(
            "call-1",
            "lookup",
            arguments={"sequence": sequence_cycle},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (lambda: TextBlock(1), TypeError, "text must be a string"),
        (lambda: ReasoningBlock(1), TypeError, "text must be a string"),
        (lambda: ImageUrl(1), TypeError, "url must be a string"),
        (lambda: ImageUrl(""), ValueError, "url must not be empty"),
        (lambda: ImageBytes(bytearray(b"x"), "image/png"), TypeError, "data must be bytes"),
        (lambda: ImageBytes(b"x", ""), ValueError, "media_type must not be empty"),
        (lambda: ImageBlock(object()), TypeError, "source must be"),
        (lambda: ImageBlock(ImageUrl("x"), alt_text=1), TypeError, "alt_text must be"),
        (lambda: ImageBlock(ImageUrl("x"), alt_text=""), ValueError, "alt_text must not"),
        (lambda: ToolCallBlock("", "tool"), ValueError, "id must not be empty"),
        (lambda: ToolCallBlock("id", ""), ValueError, "name must not be empty"),
        (lambda: ToolResultBlock("", content=()), ValueError, "tool_call_id must not"),
        (
            lambda: ToolResultBlock("id", content=1),  # type: ignore[arg-type]
            TypeError,
            "content must be an iterable",
        ),
        (
            lambda: ToolResultBlock("id", content=(object(),)),  # type: ignore[arg-type]
            TypeError,
            "only ContentBlock",
        ),
        (
            lambda: ToolResultBlock("id", content=(), is_error="yes"),  # type: ignore[arg-type]
            TypeError,
            "is_error must",
        ),
    ],
)
def test_content_blocks_validate_their_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
