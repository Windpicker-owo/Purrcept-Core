"""Represent decision results and validate their relationship to a request.

Backends construct immutable answers after translating provider responses.
Values retain provider probabilities and confidence without recalibration,
thresholding, or renormalization. Refusal is a per-question result. Validation
can only inspect the adapter-visible response, not recover information already
discarded by an upstream SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from .._utils import JsonValue, empty_json_object, freeze_json_object
from ..errors import ModelError
from ..responses import TokenUsage
from ._validation import choice_value, number, optional_text, snapshot, text, unique
from .requests import ChoiceQuestion, DecisionRequest, PredicateQuestion, ScoreQuestion


class DecisionResponseError(ModelError):
    """The adapter-visible response violates the decision contract, without retry intent."""


@dataclass(frozen=True, slots=True)
class ChoiceProbability:
    """Provider probability associated with one typed categorical value."""

    value: str | bool
    probability: float

    def __post_init__(self) -> None:
        """Reject coercion and invalid probabilities at construction."""
        choice_value(self.value)
        number(self.probability, "probability", upper=1)


@dataclass(frozen=True, slots=True)
class ScoreProbability:
    """Provider probability for a zero-based rubric index and its original label."""

    index: int
    label: str
    probability: float

    def __post_init__(self) -> None:
        """Require an actual nonnegative integer index."""
        if (
            isinstance(self.index, bool)
            or not isinstance(cast(object, self.index), int)
            or self.index < 0
        ):
            raise ValueError("index must be a nonnegative integer.")
        text(self.label, "label")
        number(self.probability, "probability", upper=1)


@dataclass(frozen=True, slots=True)
class PredicateAnswer:
    """Probability of truth; callers choose their own action threshold."""

    name: str
    probability: float

    def __post_init__(self) -> None:
        """Validate identity and preserve the probability without thresholding."""
        text(self.name, "name")
        number(self.probability, "probability", upper=1)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """Selected value and full distribution, with provider-defined confidence.

    Confidence is not a portable estimate of correctness. No application policy
    should assume that different providers compute it using the same formula.
    """

    name: str
    choice: str | bool
    probabilities: tuple[ChoiceProbability, ...]
    confidence: float

    def __post_init__(self) -> None:
        """Snapshot the distribution without repairing provider values."""
        text(self.name, "name")
        choice_value(self.choice)
        number(self.confidence, "confidence", upper=1)
        probabilities = cast(
            tuple[ChoiceProbability, ...],
            snapshot(self.probabilities, (ChoiceProbability,), "probabilities"),
        )
        unique(((type(p.value), p.value) for p in probabilities), "probability values")
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A probability-weighted level index, not a percentage or chosen category."""

    name: str
    score: float
    probabilities: tuple[ScoreProbability, ...]
    confidence: float

    def __post_init__(self) -> None:
        """Snapshot level probabilities and preserve the provider's score."""
        text(self.name, "name")
        number(self.score, "score")
        number(self.confidence, "confidence", upper=1)
        probabilities = cast(
            tuple[ScoreProbability, ...],
            snapshot(self.probabilities, (ScoreProbability,), "probabilities"),
        )
        unique((p.index for p in probabilities), "probability indices")
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class DecisionRefusal:
    """No decision for this question; other questions may still have answers."""

    name: str

    def __post_init__(self) -> None:
        """Keep refusal associated with the original question."""
        text(self.name, "name")


DecisionAnswer: TypeAlias = PredicateAnswer | ChoiceAnswer | ScoreAnswer | DecisionRefusal
"""Closed result union that requires applications to account for refusal."""


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    """Answers in request order with optional operational usage and metadata.

    Construction validates local values; adapters and Decide call validate_for to
    check correspondence to the original request. Missing usage stays unknown.
    """

    answers: tuple[DecisionAnswer, ...]
    usage: TokenUsage | None = field(default=None, kw_only=True)
    model: str | None = field(default=None, kw_only=True)
    provider_metadata: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object, kw_only=True
    )

    def __post_init__(self) -> None:
        """Snapshot answers and diagnostic metadata, rejecting duplicate names."""
        answers = cast(
            tuple[DecisionAnswer, ...],
            snapshot(
                self.answers,
                (PredicateAnswer, ChoiceAnswer, ScoreAnswer, DecisionRefusal),
                "answers",
            ),
        )
        unique((answer.name for answer in answers), "answer names")
        if self.usage is not None and not isinstance(cast(object, self.usage), TokenUsage):
            raise TypeError("usage must be TokenUsage or None.")
        optional_text(self.model, "model")
        object.__setattr__(self, "answers", answers)
        object.__setattr__(
            self,
            "provider_metadata",
            freeze_json_object(self.provider_metadata, field_name="provider_metadata"),
        )

    def answer(self, name: str) -> DecisionAnswer:
        """Find a named result, raising KeyError when the request did not contain it."""
        for answer in self.answers:
            if answer.name == name:
                return answer
        raise KeyError(name)

    def validate_for(self, request: DecisionRequest) -> None:
        """Reject missing, reordered, mismatched, or out-of-domain answers.

        Distributions must cover exactly the declared options or levels. Numeric values
        are not recomputed or renormalized: rounding and calibration belong to the
        provider, and malformed results must not become application decisions.
        """
        if tuple(a.name for a in self.answers) != tuple(q.name for q in request.questions):
            raise DecisionResponseError("Answers must match question names and order.")
        for question, answer in zip(request.questions, self.answers, strict=True):
            if isinstance(answer, DecisionRefusal):
                continue
            if isinstance(question, PredicateQuestion) and isinstance(answer, PredicateAnswer):
                continue
            if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
                expected = {(type(c.value), c.value) for c in question.choices}
                actual = {(type(p.value), p.value) for p in answer.probabilities}
                if actual != expected or (type(answer.choice), answer.choice) not in expected:
                    raise DecisionResponseError("Choice answer does not match declared options.")
                continue
            if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
                expected_levels = {(i, level.label) for i, level in enumerate(question.levels)}
                actual_levels = {(p.index, p.label) for p in answer.probabilities}
                if actual_levels != expected_levels or answer.score > len(question.levels) - 1:
                    raise DecisionResponseError("Score answer does not match declared levels.")
                continue
            raise DecisionResponseError("Answer type does not match its question.")
