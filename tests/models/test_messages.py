"""Verify message constructors, text projection, immutability, and role validation."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.content import ImageBlock, ImageUrl, TextBlock, ToolResultBlock
from purrcept_core.models.messages import Message, MessageRole


def test_message_normalizes_role_content_and_metadata() -> None:
    blocks = [TextBlock("hello"), ImageBlock(ImageUrl("https://example.test/cat.png"))]
    tags = ["one"]
    metadata: dict[str, object] = {"trace": {"tags": tags}}

    message = Message(
        "user",  # type: ignore[arg-type]
        blocks,  # type: ignore[arg-type]
        name="customer",
        metadata=metadata,  # type: ignore[arg-type]
    )
    blocks.append(TextBlock("changed"))
    tags.append("changed")

    assert message.role is MessageRole.USER
    assert message.content[:2] == tuple(blocks[:2])
    assert message.metadata == {"trace": {"tags": ("one",)}}
    assert message.text == "hello"
    with pytest.raises(FrozenInstanceError):
        message.name = "changed"  # type: ignore[misc]


def test_text_factories_create_each_message_role() -> None:
    metadata = {"source": "test"}

    user = Message.user("question", name="alice", metadata=metadata)
    assistant = Message.assistant("answer")
    from_text = Message.from_text("assistant", "again")

    assert (user.role, user.text, user.name, user.metadata) == (
        MessageRole.USER,
        "question",
        "alice",
        metadata,
    )
    assert (assistant.role, assistant.text) == (MessageRole.ASSISTANT, "answer")
    assert from_text == Message.assistant("again")
    assert {role.value for role in MessageRole} == {"user", "assistant", "tool"}
    assert not hasattr(Message, "system")


def test_tool_factory_accepts_text_and_content_iterables() -> None:
    text_message = Message.tool("call-1", "result", is_error=True)
    block_message = Message.tool("call-2", [TextBlock("a"), TextBlock("b")])

    text_result = text_message.content[0]
    block_result = block_message.content[0]

    assert text_message.role is MessageRole.TOOL
    assert isinstance(text_result, ToolResultBlock)
    assert text_result.is_error is True
    assert text_message.text == "result"
    assert isinstance(block_result, ToolResultBlock)
    assert block_message.text == "ab"


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: Message("invalid", ()),  # type: ignore[arg-type]
            ValueError,
            "Unsupported message role",
        ),
        (
            lambda: Message.from_text("invalid", "text"),
            ValueError,
            "Unsupported message role",
        ),
        (
            lambda: Message(MessageRole.USER, ()),
            ValueError,
            "at least one ContentBlock",
        ),
        (
            lambda: Message(MessageRole.USER, 1),  # type: ignore[arg-type]
            TypeError,
            "content must be an iterable",
        ),
        (
            lambda: Message(MessageRole.USER, (object(),)),  # type: ignore[arg-type]
            TypeError,
            "only ContentBlock",
        ),
        (
            lambda: Message.user("text", name=1),  # type: ignore[arg-type]
            TypeError,
            "name must be a string",
        ),
        (lambda: Message.user("text", name=""), ValueError, "name must not be empty"),
    ],
)
def test_messages_reject_invalid_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
