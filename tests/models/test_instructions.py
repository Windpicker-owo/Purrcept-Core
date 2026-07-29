"""Verify system instructions snapshot typed content, keys, and stability hints."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.caching import PromptStability
from purrcept_core.models.content import ImageBlock, ImageUrl, TextBlock
from purrcept_core.models.instructions import SystemInstruction


def test_system_instruction_snapshots_content_and_supports_text_factory() -> None:
    blocks = [TextBlock("follow policy"), ImageBlock(ImageUrl("https://example.test/x"))]

    instruction = SystemInstruction(
        blocks,  # type: ignore[arg-type]
        key="policy",
        stability="growing",  # type: ignore[arg-type]
    )
    text_instruction = SystemInstruction.from_text(
        "be concise",
        stability=PromptStability.VOLATILE,
    )
    blocks.append(TextBlock("changed"))

    assert instruction.content == (
        TextBlock("follow policy"),
        ImageBlock(ImageUrl("https://example.test/x")),
    )
    assert instruction.key == "policy"
    assert instruction.stability is PromptStability.GROWING
    assert text_instruction == SystemInstruction(
        (TextBlock("be concise"),),
        stability=PromptStability.VOLATILE,
    )
    assert SystemInstruction.from_text("stable").stability is PromptStability.STABLE
    with pytest.raises(FrozenInstanceError):
        instruction.key = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: SystemInstruction(1),  # type: ignore[arg-type]
            TypeError,
            "content must be an iterable",
        ),
        (lambda: SystemInstruction(()), ValueError, "at least one ContentBlock"),
        (
            lambda: SystemInstruction((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only ContentBlock",
        ),
        (
            lambda: SystemInstruction((TextBlock("x"),), key=1),  # type: ignore[arg-type]
            TypeError,
            "key must be a string",
        ),
        (
            lambda: SystemInstruction((TextBlock("x"),), key=""),
            ValueError,
            "key must not be empty",
        ),
        (
            lambda: SystemInstruction(
                (TextBlock("x"),),
                stability="unknown",  # type: ignore[arg-type]
            ),
            ValueError,
            "Unsupported prompt stability",
        ),
    ],
)
def test_system_instruction_validates_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
