# Audio evidence

`purrcept_core.models.AudioBytes` holds an immutable encoded WAV or MP3 snapshot.
`AudioBlock` carries that snapshot in a user message without converting it into text:

```python
from pathlib import Path
from purrcept_core.models import AudioBlock, AudioBytes, Message, MessageRole

message = Message(MessageRole.USER, (
    AudioBlock(AudioBytes(Path("utterance.wav").read_bytes())),
))
```

Core owns no microphone, VAD, transcription, playback, or media storage. Applications
validate the container, duration, and resource budget. Backends map the block to a
provider's native audio input and must reject unsupported content explicitly.
An optional `alt_text` describes the recording; it does not constitute a transcript.
Audio bytes are omitted from the value's representation and from `Message.text`.
