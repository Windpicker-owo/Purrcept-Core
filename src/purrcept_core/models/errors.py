"""Classify model-backend failures for orchestration and retry policy.

Provider adapters raise these errors after translating transport failures.
Permanent request/authentication/context errors are separated from transient
rate-limit and availability errors, while cancellation and unexpected
implementation failures remain their original exception types.
"""

from __future__ import annotations

from math import isfinite

from ..errors import PurrceptError
from ._utils import require_non_empty_string, validate_optional_non_empty_string


class ModelError(PurrceptError):
    """Base class for normalized failures raised by model backends.

    Optional provider and model identifiers are diagnostic snapshots. The
    hierarchy itself carries retry meaning; the core does not automatically
    retry model calls.
    """

    def __init__(
        self,
        message: str = "Model generation failed.",
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        message = require_non_empty_string(message, field_name="message")
        validate_optional_non_empty_string(provider, field_name="provider")
        validate_optional_non_empty_string(model, field_name="model")
        self.provider = provider
        self.model = model
        super().__init__(message)


class ModelRequestError(ModelError):
    """A permanent failure caused by an invalid model request."""


class ModelAuthenticationError(ModelError):
    """A model provider rejected the runtime's credentials."""


class ModelContextWindowError(ModelRequestError):
    """A request exceeded the selected model's context window."""


class ModelTransientError(ModelError):
    """A model request may succeed when attempted again later."""


class ModelRateLimitError(ModelTransientError):
    """A provider rate limit prevented the request."""

    def __init__(
        self,
        message: str = "Model rate limit exceeded.",
        *,
        retry_after_seconds: float | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        if retry_after_seconds is not None:
            retry_after_seconds = _normalize_retry_after(retry_after_seconds)
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, provider=provider, model=model)


class ModelUnavailableError(ModelTransientError):
    """A provider or selected model is temporarily unavailable."""


def _normalize_retry_after(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("retry_after_seconds must be a number.")
    if not isfinite(value):
        raise ValueError("retry_after_seconds must be finite.")
    if value < 0:
        raise ValueError("retry_after_seconds must be greater than or equal to zero.")
    return float(value)


__all__ = [
    "ModelAuthenticationError",
    "ModelContextWindowError",
    "ModelError",
    "ModelRateLimitError",
    "ModelRequestError",
    "ModelTransientError",
    "ModelUnavailableError",
]
