"""Verify the provider-neutral model error taxonomy and diagnostic validation."""

import pytest

from purrcept_core.errors import PurrceptError
from purrcept_core.models.errors import (
    ModelAuthenticationError,
    ModelContextWindowError,
    ModelError,
    ModelRateLimitError,
    ModelRequestError,
    ModelTransientError,
    ModelUnavailableError,
)


def test_model_error_hierarchy_separates_permanent_and_transient_failures() -> None:
    authentication = ModelAuthenticationError(
        "bad key",
        provider="example",
        model="test-model",
    )
    context = ModelContextWindowError("too long")
    unavailable = ModelUnavailableError("offline")

    assert isinstance(authentication, (PurrceptError, ModelError))
    assert not isinstance(authentication, ModelRequestError)
    assert authentication.provider == "example"
    assert authentication.model == "test-model"
    assert isinstance(context, ModelRequestError)
    assert isinstance(unavailable, (ModelError, ModelTransientError))
    assert not isinstance(authentication, ModelTransientError)


def test_rate_limit_error_exposes_optional_retry_delay() -> None:
    default = ModelRateLimitError()
    delayed = ModelRateLimitError(
        "slow down",
        retry_after_seconds=3,
        provider="example",
        model="test-model",
    )

    assert str(default) == "Model rate limit exceeded."
    assert default.retry_after_seconds is None
    assert str(delayed) == "slow down"
    assert delayed.retry_after_seconds == 3.0
    assert delayed.provider == "example"
    assert delayed.model == "test-model"
    assert isinstance(delayed, ModelTransientError)


@pytest.mark.parametrize(
    ("factory", "error_type", "match"),
    [
        (lambda: ModelError(1), TypeError, "message must be a string"),
        (lambda: ModelError(""), ValueError, "message must not be empty"),
        (
            lambda: ModelError(provider=1),  # type: ignore[arg-type]
            TypeError,
            "provider must be a string",
        ),
        (
            lambda: ModelError(provider=""),
            ValueError,
            "provider must not be empty",
        ),
        (
            lambda: ModelError(model=1),  # type: ignore[arg-type]
            TypeError,
            "model must be a string",
        ),
        (
            lambda: ModelError(model=""),
            ValueError,
            "model must not be empty",
        ),
    ],
)
def test_model_error_validates_diagnostics(
    factory: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("retry_after_seconds", "error_type", "match"),
    [
        (True, TypeError, "must be a number"),
        ("soon", TypeError, "must be a number"),
        (float("inf"), ValueError, "must be finite"),
        (-0.1, ValueError, "greater than or equal"),
    ],
)
def test_rate_limit_error_validates_retry_delay(
    retry_after_seconds: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        ModelRateLimitError(
            retry_after_seconds=retry_after_seconds,  # type: ignore[arg-type]
        )
