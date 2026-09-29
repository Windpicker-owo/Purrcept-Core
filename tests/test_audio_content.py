"""Audio evidence stays immutable and separate from visible text."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models import AudioBlock, AudioBytes, Message, MessageRole


def test_audio_snapshot_is_not_transcribed_or_logged() -> None:
    source = AudioBytes(b"encoded-private-audio")
    message = Message(MessageRole.USER, (AudioBlock(source),))
    assert message.text == ""
    assert "encoded-private-audio" not in repr(message)
    with pytest.raises(FrozenInstanceError):
        source.data = b"changed"  # type: ignore[misc]


@pytest.mark.parametrize("data,mime", [(b"", "audio/wav"), (b"x", "audio/pcm")])
def test_invalid_audio_is_rejected(data: bytes, mime: str) -> None:
    with pytest.raises(ValueError):
        AudioBytes(data, mime)
