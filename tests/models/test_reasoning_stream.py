"""Verify readable reasoning is a separate, validated streaming channel."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from purrcept_core.models import ReasoningDelta, TextDelta


def test_reasoning_delta_is_distinct_and_immutable() -> None:
    """Observers can display a provider summary without treating it as answer text."""

    event = ReasoningDelta("Compare the available results.", index=2, is_summary=True)
    assert not isinstance(event, TextDelta)
    assert event.index == 2
    assert event.is_summary
    with pytest.raises(FrozenInstanceError):
        event.delta = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"delta": 1}, TypeError),
        ({"delta": "x", "index": True}, TypeError),
        ({"delta": "x", "index": -1}, ValueError),
        ({"delta": "x", "is_summary": "yes"}, TypeError),
    ],
)
def test_reasoning_delta_validates_input(kwargs: dict[str, Any], error: type[Exception]) -> None:
    """Reject ambiguous channel identity and non-text payloads at the boundary."""

    with pytest.raises(error):
        ReasoningDelta(**kwargs)
