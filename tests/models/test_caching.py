"""Verify prompt-cache values normalize enum inputs and enforce immutable boundaries."""

from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from purrcept_core.models.caching import (
    CacheMode,
    PromptCachePolicy,
    PromptStability,
)


def test_prompt_cache_policy_normalizes_mode_and_preserves_ttl() -> None:
    ttl = timedelta(minutes=5)

    policy = PromptCachePolicy(
        mode="explicit",  # type: ignore[arg-type]
        key="shared-prefix",
        ttl=ttl,
        strict=True,
    )

    assert policy.mode is CacheMode.EXPLICIT
    assert policy.key == "shared-prefix"
    assert policy.ttl is ttl
    assert policy.strict is True
    with pytest.raises(FrozenInstanceError):
        policy.strict = False  # type: ignore[misc]


def test_prompt_cache_types_have_stable_values_and_defaults() -> None:
    policy = PromptCachePolicy()

    assert policy == PromptCachePolicy(mode=CacheMode.AUTO)
    assert policy.key is None
    assert policy.ttl is None
    assert policy.strict is False
    assert {mode.value for mode in CacheMode} == {
        "auto",
        "disabled",
        "prefer",
        "explicit",
    }
    assert {stability.value for stability in PromptStability} == {
        "stable",
        "growing",
        "volatile",
    }


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (
            lambda: PromptCachePolicy(mode="invalid"),  # type: ignore[arg-type]
            ValueError,
            "Unsupported prompt cache mode",
        ),
        (
            lambda: PromptCachePolicy(key=1),  # type: ignore[arg-type]
            TypeError,
            "key must be a string",
        ),
        (lambda: PromptCachePolicy(key=""), ValueError, "key must not be empty"),
        (
            lambda: PromptCachePolicy(ttl=1),  # type: ignore[arg-type]
            TypeError,
            "ttl must be a timedelta or None",
        ),
        (
            lambda: PromptCachePolicy(ttl=timedelta(0)),
            ValueError,
            "ttl must be greater than zero",
        ),
        (
            lambda: PromptCachePolicy(ttl=timedelta(seconds=-1)),
            ValueError,
            "ttl must be greater than zero",
        ),
        (
            lambda: PromptCachePolicy(strict=1),  # type: ignore[arg-type]
            TypeError,
            "strict must be a bool",
        ),
    ],
)
def test_prompt_cache_policy_validates_public_boundaries(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]
