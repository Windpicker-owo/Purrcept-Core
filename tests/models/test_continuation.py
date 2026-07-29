"""Verify provider continuation snapshots are opaque, recursive, and immutable."""

from dataclasses import FrozenInstanceError

import pytest

from purrcept_core.models.continuation import ContinuationPolicy, ModelContinuation


def test_model_continuation_takes_a_deep_immutable_snapshot() -> None:
    ids = ["response-1"]
    data: dict[str, object] = {"ids": ids, "cursor": {"offset": 2}}

    continuation = ModelContinuation(
        "provider",
        data,  # type: ignore[arg-type]
    )
    ids.append("changed")
    data.clear()

    assert continuation.provider == "provider"
    assert continuation.data == {
        "ids": ("response-1",),
        "cursor": {"offset": 2},
    }
    with pytest.raises(TypeError):
        continuation.data["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        continuation.provider = "changed"  # type: ignore[misc]


def test_continuation_policy_has_stable_values() -> None:
    assert {policy.value for policy in ContinuationPolicy} == {
        "client_managed",
        "provider_managed",
        "auto",
    }


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: ModelContinuation(1, {}),  # type: ignore[arg-type]
            TypeError,
            "provider must be a string",
        ),
        (lambda: ModelContinuation("", {}), ValueError, "provider must not be empty"),
        (
            lambda: ModelContinuation("provider", []),  # type: ignore[arg-type]
            TypeError,
            "data must be a mapping",
        ),
    ],
)
def test_model_continuation_validates_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
