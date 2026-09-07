"""Verify the root package exposes its declared stable public API."""

from __future__ import annotations

import purrcept_core


def test_package_name_and_version() -> None:
    assert purrcept_core.__name__ == "purrcept_core"
    assert purrcept_core.__version__ == "0.5.1"


def test_every_declared_public_name_is_available() -> None:
    missing = [name for name in purrcept_core.__all__ if not hasattr(purrcept_core, name)]

    assert missing == []


def test_required_v01_api_is_exported() -> None:
    required = {
        "AgentDriver",
        "AgentFlow",
        "AgentRun",
        "CallbackEventSink",
        "CompositeEventSink",
        "DispatchExecutor",
        "Effect",
        "EffectExecutor",
        "EffectMiddleware",
        "ExecutionContext",
        "FunctionMiddleware",
        "GeneratorRun",
        "InlineExecutor",
        "MiddlewareExecutor",
        "NullEventSink",
        "RetryMiddleware",
        "SafeEventSink",
        "TimeoutMiddleware",
        "perform",
    }

    assert required <= set(purrcept_core.__all__)
