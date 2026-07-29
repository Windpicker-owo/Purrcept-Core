"""Verify complete request and response snapshots plus their cross-field invariants."""

from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from purrcept_core.models.caching import CacheMode, PromptCachePolicy
from purrcept_core.models.content import ImageBlock, ImageUrl, TextBlock, ToolCallBlock
from purrcept_core.models.continuation import ModelContinuation
from purrcept_core.models.instructions import SystemInstruction
from purrcept_core.models.messages import Message, MessageRole
from purrcept_core.models.reminders import SystemReminder
from purrcept_core.models.requests import ModelRequest, ToolSpec
from purrcept_core.models.responses import FinishReason, ModelResponse, TokenUsage
from purrcept_core.models.settings import ModelSettings, ToolChoice


def _message() -> Message:
    return Message.user("hello")


def _instruction() -> SystemInstruction:
    return SystemInstruction.from_text("follow policy")


def _reminder() -> SystemReminder:
    return SystemReminder((TextBlock("cite sources"),))


def test_tool_spec_and_request_take_immutable_snapshots() -> None:
    properties: dict[str, object] = {"query": {"type": "string"}}
    tool = ToolSpec(
        "lookup",
        description="Look something up",
        parameters={  # type: ignore[dict-item]
            "type": "object",
            "properties": properties,
        },
    )
    messages = [_message()]
    instructions = [_instruction()]
    reminders = [_reminder()]
    tools = [tool]
    metadata: dict[str, object] = {"tags": ["one"]}
    options: dict[str, object] = {"thinking": {"budget": 10}}
    settings = ModelSettings(
        temperature=0.5,
        max_output_tokens=100,
        stop_sequences=("END",),
        tool_choice=ToolChoice.REQUIRED,
        parallel_tool_calls=True,
    )
    cache = PromptCachePolicy(
        mode=CacheMode.PREFER,
        ttl=timedelta(minutes=5),
    )
    continuation = ModelContinuation("provider", {"cursor": ["one"]})

    request = ModelRequest(
        messages,  # type: ignore[arg-type]
        instructions=instructions,  # type: ignore[arg-type]
        reminders=reminders,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
        settings=settings,
        cache=cache,
        continuation=continuation,
        metadata=metadata,  # type: ignore[arg-type]
        provider_options=options,  # type: ignore[arg-type]
    )
    messages.append(Message.user("changed"))
    instructions.clear()
    reminders.clear()
    tools.clear()
    properties["new"] = {"type": "number"}
    metadata["tags"] = []
    options.clear()

    assert request.messages == (_message(),)
    assert request.instructions == (_instruction(),)
    assert request.reminders == (_reminder(),)
    assert request.tools == (tool,)
    assert request.settings is settings
    assert request.cache is cache
    assert request.continuation is continuation
    assert tool.parameters == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    assert request.metadata == {"tags": ("one",)}
    assert request.provider_options == {"thinking": {"budget": 10}}
    assert not hasattr(request, "model")
    assert not hasattr(request, "temperature")
    with pytest.raises(FrozenInstanceError):
        request.settings = ModelSettings()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (lambda: ToolSpec(""), ValueError, "name must not be empty"),
        (
            lambda: ToolSpec("tool", description=1),  # type: ignore[arg-type]
            TypeError,
            "description must be a string",
        ),
        (
            lambda: ToolSpec("tool", description=""),
            ValueError,
            "description must not be empty",
        ),
        (
            lambda: ToolSpec("tool", parameters=[]),  # type: ignore[arg-type]
            TypeError,
            "parameters must be a mapping",
        ),
    ],
)
def test_tool_spec_validates_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "model_request",
    [
        ModelRequest((_message(),)),
        ModelRequest((), instructions=(_instruction(),)),
        ModelRequest((), reminders=(_reminder(),)),
    ],
)
def test_request_accepts_any_non_empty_prompt_channel(model_request: ModelRequest) -> None:
    assert model_request.messages or model_request.instructions or model_request.reminders
    assert model_request.tools == ()
    assert model_request.settings == ModelSettings()
    assert model_request.cache == PromptCachePolicy()
    assert model_request.continuation is None
    assert model_request.metadata == {}
    assert model_request.provider_options == {}


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (lambda: ModelRequest(()), ValueError, "at least one message"),
        (
            lambda: ModelRequest(1),  # type: ignore[arg-type]
            TypeError,
            "messages must be an iterable",
        ),
        (
            lambda: ModelRequest((object(),)),  # type: ignore[arg-type]
            TypeError,
            "only Message",
        ),
        (
            lambda: ModelRequest((_message(),), instructions=1),  # type: ignore[arg-type]
            TypeError,
            "instructions must be an iterable",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                instructions=(object(),),  # type: ignore[arg-type]
            ),
            TypeError,
            "only SystemInstruction",
        ),
        (
            lambda: ModelRequest((_message(),), reminders=1),  # type: ignore[arg-type]
            TypeError,
            "reminders must be an iterable",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                reminders=(object(),),  # type: ignore[arg-type]
            ),
            TypeError,
            "only SystemReminder",
        ),
        (
            lambda: ModelRequest((_message(),), tools=1),  # type: ignore[arg-type]
            TypeError,
            "tools must be an iterable",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                tools=(object(),),  # type: ignore[arg-type]
            ),
            TypeError,
            "only ToolSpec",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                tools=(ToolSpec("same"), ToolSpec("same")),
            ),
            ValueError,
            "unique names",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                instructions=(
                    SystemInstruction.from_text("one", key="same"),
                    SystemInstruction.from_text("two", key="same"),
                ),
            ),
            ValueError,
            "instructions must have unique",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                reminders=(
                    SystemReminder((TextBlock("one"),), key="same"),
                    SystemReminder((TextBlock("two"),), key="same"),
                ),
            ),
            ValueError,
            "reminders must have unique",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                settings=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "settings must be a ModelSettings",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                settings=ModelSettings(tool_choice=ToolChoice.REQUIRED),
            ),
            ValueError,
            "tools must not be empty",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                cache=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "cache must be a PromptCachePolicy",
        ),
        (
            lambda: ModelRequest(
                (_message(),),
                continuation=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "continuation must be a ModelContinuation",
        ),
    ],
)
def test_model_request_validates_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


def test_finish_reasons_only_describe_successful_completion() -> None:
    assert {reason.value for reason in FinishReason} == {
        "stop",
        "length",
        "tool_call",
        "content_filter",
        "other",
    }


def test_token_usage_tracks_cache_and_reasoning_breakdowns() -> None:
    usage = TokenUsage(
        7,
        5,
        cached_input_tokens=3,
        cache_write_input_tokens=2,
        reasoning_tokens=4,
    )

    assert usage.total_tokens == 12
    assert usage.cached_input_tokens == 3
    assert usage.cache_write_input_tokens == 2
    assert usage.reasoning_tokens == 4
    assert TokenUsage(0, 0) == TokenUsage(
        0,
        0,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_tokens=0,
    )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("input_tokens", True, TypeError),
        ("output_tokens", 1.5, TypeError),
        ("cached_input_tokens", -1, ValueError),
        ("cache_write_input_tokens", True, TypeError),
        ("reasoning_tokens", -1, ValueError),
    ],
)
def test_token_usage_validates_every_count(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    values[field_name] = value

    with pytest.raises(error_type, match=field_name):
        TokenUsage(**values)  # type: ignore[arg-type]


def test_response_exposes_tool_calls_continuation_and_frozen_metadata() -> None:
    call = ToolCallBlock("call-1", "lookup", arguments={"query": "cat"})
    message = Message(
        MessageRole.ASSISTANT,
        (
            TextBlock("working"),
            ImageBlock(ImageUrl("https://example.test/cat.png")),
            call,
        ),
    )
    continuation = ModelContinuation("provider", {"cursor": "next"})
    metadata: dict[str, object] = {"trace": ["one"]}

    response = ModelResponse(
        message,
        usage=TokenUsage(2, 3),
        finish_reason="tool_call",  # type: ignore[arg-type]
        model="example-model",
        response_id="response-1",
        continuation=continuation,
        provider_metadata=metadata,  # type: ignore[arg-type]
    )
    metadata["trace"] = []

    assert response.finish_reason is FinishReason.TOOL_CALL
    assert response.text == "working"
    assert response.tool_calls == (call,)
    assert response.continuation is continuation
    assert response.provider_metadata == {"trace": ("one",)}


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ModelResponse(object()),  # type: ignore[arg-type]
            TypeError,
            "message must be a Message",
        ),
        (
            lambda: ModelResponse(Message.assistant("x"), usage=object()),  # type: ignore[arg-type]
            TypeError,
            "usage must be a TokenUsage",
        ),
        (
            lambda: ModelResponse(
                Message.assistant("x"),
                finish_reason="invalid",  # type: ignore[arg-type]
            ),
            ValueError,
            "Unsupported finish reason",
        ),
        (
            lambda: ModelResponse(Message.assistant("x"), model=1),  # type: ignore[arg-type]
            TypeError,
            "model must be a string",
        ),
        (
            lambda: ModelResponse(Message.assistant("x"), model=""),
            ValueError,
            "model must not be empty",
        ),
        (
            lambda: ModelResponse(Message.assistant("x"), response_id=1),  # type: ignore[arg-type]
            TypeError,
            "response_id must be a string",
        ),
        (
            lambda: ModelResponse(Message.assistant("x"), response_id=""),
            ValueError,
            "response_id must not be empty",
        ),
        (
            lambda: ModelResponse(
                Message.assistant("x"),
                continuation=object(),  # type: ignore[arg-type]
            ),
            TypeError,
            "continuation must be a ModelContinuation",
        ),
        (
            lambda: ModelResponse(Message.user("not a response")),
            ValueError,
            "message role must be assistant",
        ),
    ],
)
def test_model_response_validates_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
