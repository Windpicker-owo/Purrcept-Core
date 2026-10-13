"""Exercise decision value contracts and the generic effect execution boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor, perform
from purrcept_core.models import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceProbability,
    ChoiceQuestion,
    Decide,
    DecisionBackend,
    DecisionModel,
    DecisionQuestion,
    DecisionRefusal,
    DecisionRequest,
    DecisionResponse,
    DecisionResponseError,
    ImageBlock,
    ImageUrl,
    JsonValue,
    ModelUnavailableError,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    ScoreLevel,
    ScoreProbability,
    ScoreQuestion,
    TextBlock,
    TokenUsage,
)


def questions() -> tuple[DecisionQuestion, ...]:
    return (
        PredicateQuestion("refund", "Is a refund requested?"),
        ChoiceQuestion("team", "Select a team.", (ChoiceOption(True), ChoiceOption("true"))),
        ScoreQuestion("urgency", "Rate urgency.", (ScoreLevel("low"), ScoreLevel("high"))),
    )


def response() -> DecisionResponse:
    return DecisionResponse(
        (
            PredicateAnswer("refund", 0.8),
            ChoiceAnswer(
                "team", True, (ChoiceProbability(True, 0.7), ChoiceProbability("true", 0.3)), 0.4
            ),
            ScoreAnswer(
                "urgency",
                0.2,
                (ScoreProbability(0, "low", 0.8), ScoreProbability(1, "high", 0.2)),
                0.6,
            ),
        ),
        model="decision-model",
        usage=TokenUsage(100, 0),
    )


def test_snapshots_and_typed_values() -> None:
    options = [ChoiceOption(True, "boolean"), ChoiceOption("true", "string")]
    levels = [ScoreLevel("low", "No impact"), ScoreLevel("high")]
    blocks = [TextBlock("evidence"), ImageBlock(ImageUrl("https://example.org/photo.png"))]
    metadata: dict[str, JsonValue] = {"nested": [1]}
    request = DecisionRequest(
        cast(tuple[TextBlock | ImageBlock, ...], blocks),
        (
            ChoiceQuestion("team", "Choose.", cast(tuple[ChoiceOption, ...], options)),
            ScoreQuestion("urgency", "Rate.", cast(tuple[ScoreLevel, ...], levels)),
        ),
        metadata=metadata,
        provider_options=metadata,
    )
    options.clear()
    levels.clear()
    blocks.clear()
    metadata.clear()
    assert len(request.input) == 2
    assert request.metadata["nested"] == (1,)
    assert request.provider_options["nested"] == (1,)
    with pytest.raises(FrozenInstanceError):
        request.input = "changed"
    result = response()
    result.validate_for(DecisionRequest("ticket", questions()))
    assert result.answer("team") == result.answers[1]
    with pytest.raises(KeyError):
        result.answer("missing")
    refusal = DecisionResponse((DecisionRefusal("refund"), *result.answers[1:]))
    refusal.validate_for(DecisionRequest("ticket", questions()))
    assert refusal.usage is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: PredicateQuestion(cast(str, 1), "question"),
        lambda: PredicateQuestion("q", "  "),
        lambda: ChoiceOption(cast(str, 1)),
        lambda: ChoiceOption("a", ""),
        lambda: ScoreLevel(""),
        lambda: ChoiceQuestion("q", "?", (ChoiceOption("a"),)),
        lambda: ChoiceQuestion("q", "?", (ChoiceOption("a"), ChoiceOption("a"))),
        lambda: ScoreQuestion("q", "?", (ScoreLevel("a"),)),
        lambda: DecisionRequest("", questions()),
        lambda: DecisionRequest(cast(str, 1), questions()),
        lambda: DecisionRequest(cast(str, b"invalid"), questions()),
        lambda: DecisionRequest((), questions()),
        lambda: DecisionRequest(cast(str, [object()]), questions()),
        lambda: DecisionRequest("x", ()),
        lambda: DecisionRequest("x", (questions()[0], questions()[0])),
        lambda: PredicateAnswer("q", cast(float, True)),
        lambda: PredicateAnswer("q", float("nan")),
        lambda: PredicateAnswer("q", -1),
        lambda: PredicateAnswer("q", 1.1),
        lambda: ScoreProbability(True, "level", 0.5),
        lambda: ScoreProbability(-1, "level", 0.5),
        lambda: ScoreProbability(cast(int, "0"), "level", 0.5),
        lambda: ChoiceAnswer(
            "q", "a", (ChoiceProbability("a", 0.5), ChoiceProbability("a", 0.5)), 1
        ),
        lambda: ScoreAnswer(
            "q", 0, (ScoreProbability(0, "a", 0.5), ScoreProbability(0, "a", 0.5)), 1
        ),
        lambda: DecisionResponse(()),
        lambda: DecisionResponse((PredicateAnswer("q", 0), PredicateAnswer("q", 1))),
        lambda: DecisionResponse((PredicateAnswer("q", 0),), usage=cast(TokenUsage, object())),
        lambda: DecisionResponse((PredicateAnswer("q", 0),), model=""),
    ],
)
def test_invalid_values(build: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        build()


@pytest.mark.parametrize(
    "result",
    [
        DecisionResponse((PredicateAnswer("missing", 0.5),)),
        DecisionResponse((PredicateAnswer("refund", 0.5),)),
        DecisionResponse(
            (
                PredicateAnswer("refund", 0.5),
                PredicateAnswer("team", 0.5),
                DecisionRefusal("urgency"),
            )
        ),
        DecisionResponse(
            (
                DecisionRefusal("refund"),
                ChoiceAnswer("team", "unknown", (ChoiceProbability(True, 1),), 1),
                DecisionRefusal("urgency"),
            )
        ),
        DecisionResponse(
            (
                DecisionRefusal("refund"),
                DecisionRefusal("team"),
                ScoreAnswer("urgency", 2, (ScoreProbability(0, "low", 1),), 1),
            )
        ),
        DecisionResponse(
            (
                DecisionRefusal("refund"),
                DecisionRefusal("team"),
                ScoreAnswer(
                    "urgency", 2, (ScoreProbability(0, "low", 1), ScoreProbability(1, "high", 0)), 1
                ),
            )
        ),
    ],
)
def test_response_must_correspond_to_request(result: DecisionResponse) -> None:
    with pytest.raises(DecisionResponseError):
        result.validate_for(DecisionRequest("ticket", questions()))


class RecordingBackend:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[DecisionRequest, str]] = []

    async def decide(self, request: DecisionRequest, *, model: str) -> DecisionResponse:
        self.calls.append((request, model))
        if isinstance(self.result, BaseException):
            raise self.result
        return cast(DecisionResponse, self.result)


async def run(model: DecisionModel) -> DecisionResponse:
    def flow() -> AgentFlow[DecisionResponse]:
        return (yield from perform(model.decide(DecisionRequest("ticket", questions()))))

    return await AgentDriver(InlineExecutor()).run(flow(), host=object())


async def test_decide_uses_existing_driver_without_streaming() -> None:
    backend = RecordingBackend(response())
    model = DecisionModel(backend, "exact-model")
    assert isinstance(backend, DecisionBackend)
    effect = model.decide(DecisionRequest("ticket", questions()))
    assert isinstance(effect, Decide)
    assert not backend.calls
    assert await run(model) == response()
    assert backend.calls[0][1] == "exact-model"
    assert "RecordingBackend" not in repr(model)


@pytest.mark.parametrize("error", [ModelUnavailableError("offline"), asyncio.CancelledError()])
async def test_error_and_cancellation_propagate(error: BaseException) -> None:
    with pytest.raises(type(error)) as captured:
        await run(DecisionModel(RecordingBackend(error), "model"))
    assert captured.value is error


async def test_invalid_backend_result() -> None:
    with pytest.raises(DecisionResponseError):
        await run(DecisionModel(RecordingBackend(object()), "model"))


def test_invalid_bindings() -> None:
    with pytest.raises(TypeError):
        DecisionModel(cast(DecisionBackend, object()), "model")
    model = DecisionModel(RecordingBackend(response()), "model")
    with pytest.raises(TypeError):
        Decide(cast(DecisionModel, object()), DecisionRequest("ticket", questions()))
    with pytest.raises(TypeError):
        model.decide(cast(DecisionRequest, object()))
