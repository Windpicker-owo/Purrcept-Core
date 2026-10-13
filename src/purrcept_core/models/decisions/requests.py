"""Describe independent typed questions over a shared evidence snapshot.

Applications construct these immutable values before creating a Decide effect.
Every question sees the same input, not other answers in the request. Dependent
questions require separate effects. This module owns portable value contracts;
provider limits, wire serialization, network resources, and business thresholds
belong to adapters and applications.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from .._utils import JsonValue, empty_json_object, freeze_json_object
from ..content import ImageBlock, TextBlock
from ._validation import choice_value, optional_text, snapshot, text, unique


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """One allowed categorical value; its optional description defines meaning."""

    value: str | bool
    description: str | None = None

    def __post_init__(self) -> None:
        """Preserve typed identity instead of coercing values."""
        choice_value(self.value)
        optional_text(self.description, "description")


@dataclass(frozen=True, slots=True)
class ScoreLevel:
    """One rubric level, whose numeric index comes from its request position."""

    label: str
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate the rubric's human-readable criteria."""
        text(self.label, "label")
        optional_text(self.description, "description")


@dataclass(frozen=True, slots=True)
class PredicateQuestion:
    """Ask for the probability that instructions describe a true condition.

    The name identifies this question's answer within a request. Instructions are
    explicit criteria, not a generated explanation or a threshold policy.
    """

    name: str
    instructions: str

    def __post_init__(self) -> None:
        """Require an explicit identity and question."""
        text(self.name, "name")
        text(self.instructions, "instructions")


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    """Select among at least two distinct, unordered categorical values."""

    name: str
    instructions: str
    choices: tuple[ChoiceOption, ...]

    def __post_init__(self) -> None:
        """Snapshot options while preserving string/boolean distinctions."""
        text(self.name, "name")
        text(self.instructions, "instructions")
        choices = cast(tuple[ChoiceOption, ...], snapshot(self.choices, (ChoiceOption,), "choices"))
        if len(choices) < 2:
            raise ValueError("choices requires at least two options.")
        unique(((type(item.value), item.value) for item in choices), "choice values")
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    """Rate evidence against at least two levels ordered from lowest to highest.

    Answers use the probability-weighted level index, so intermediate scores are
    valid. Labels describe the rubric; they do not supply arbitrary numeric weights.
    """

    name: str
    instructions: str
    levels: tuple[ScoreLevel, ...]

    def __post_init__(self) -> None:
        """Freeze the order that determines all subsequent score indices."""
        text(self.name, "name")
        text(self.instructions, "instructions")
        levels = cast(tuple[ScoreLevel, ...], snapshot(self.levels, (ScoreLevel,), "levels"))
        if len(levels) < 2:
            raise ValueError("levels requires at least two levels.")
        object.__setattr__(self, "levels", levels)


DecisionQuestion: TypeAlias = PredicateQuestion | ChoiceQuestion | ScoreQuestion
"""Portable question variants; provider-native structured criteria are excluded."""
DecisionInput: TypeAlias = str | tuple[TextBlock | ImageBlock, ...]
"""Shared evidence, with ordered text/image blocks and no conversation roles."""


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """An immutable evidence snapshot and independent, uniquely named questions.

    Metadata is for operational observers. Provider options are namespaced adapter
    hints, never a way to replace input or questions. Structured application data
    must be explicitly serialized to text before crossing this boundary.
    """

    input: DecisionInput
    questions: tuple[DecisionQuestion, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=empty_json_object, kw_only=True)
    provider_options: Mapping[str, JsonValue] = field(
        default_factory=empty_json_object, kw_only=True
    )

    def __post_init__(self) -> None:
        """Freeze caller-owned containers before any effect can be executed."""
        if isinstance(self.input, str):
            text(self.input, "input")
        else:
            blocks = snapshot(self.input, (TextBlock, ImageBlock), "input")
            object.__setattr__(self, "input", blocks)
        questions = cast(
            tuple[DecisionQuestion, ...],
            snapshot(
                self.questions, (PredicateQuestion, ChoiceQuestion, ScoreQuestion), "questions"
            ),
        )
        unique((question.name for question in questions), "question names")
        object.__setattr__(self, "questions", questions)
        for name in ("metadata", "provider_options"):
            object.__setattr__(self, name, freeze_json_object(getattr(self, name), field_name=name))
