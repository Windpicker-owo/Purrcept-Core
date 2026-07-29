"""Public API for the runtime-agnostic Purrcept agent core."""

from .context import ExecutionContext, ProgressCallback
from .driver import AgentDriver, Clock, IdFactory, Monotonic
from .effect import Effect
from .errors import (
    EventDispatchError,
    InvalidRunStateError,
    PurrceptError,
    UnsupportedEffectError,
)
from .events import (
    CallbackEventSink,
    CompositeEventSink,
    EffectEvent,
    EffectFailed,
    EffectProgress,
    EffectStarted,
    EffectSucceeded,
    EventSink,
    NullEventSink,
    RunCancelled,
    RunEvent,
    RunFailed,
    RunStarted,
    RunSucceeded,
    SafeEventSink,
)
from .executor import DispatchExecutor, EffectExecutor, InlineExecutor
from .flow import AgentFlow, perform
from .middleware import (
    EffectMiddleware,
    FunctionMiddleware,
    MiddlewareExecutor,
    NextExecutor,
    RetryDelay,
    RetryMiddleware,
    RetryPredicate,
    SleepFunction,
    TimeoutMiddleware,
)
from .run import (
    AgentRun,
    GeneratorRun,
    Returned,
    RunState,
    Yielded,
    as_agent_run,
)

__version__ = "0.5.0"
"""Version of the public Purrcept Core API exposed by this package."""

__all__ = [
    "AgentDriver",
    "AgentFlow",
    "AgentRun",
    "CallbackEventSink",
    "Clock",
    "CompositeEventSink",
    "DispatchExecutor",
    "Effect",
    "EffectEvent",
    "EffectExecutor",
    "EffectFailed",
    "EffectMiddleware",
    "EffectProgress",
    "EffectStarted",
    "EffectSucceeded",
    "EventDispatchError",
    "EventSink",
    "ExecutionContext",
    "FunctionMiddleware",
    "GeneratorRun",
    "IdFactory",
    "InlineExecutor",
    "InvalidRunStateError",
    "MiddlewareExecutor",
    "Monotonic",
    "NextExecutor",
    "NullEventSink",
    "ProgressCallback",
    "PurrceptError",
    "RetryDelay",
    "RetryMiddleware",
    "RetryPredicate",
    "Returned",
    "RunCancelled",
    "RunEvent",
    "RunFailed",
    "RunStarted",
    "RunState",
    "RunSucceeded",
    "SafeEventSink",
    "SleepFunction",
    "TimeoutMiddleware",
    "UnsupportedEffectError",
    "Yielded",
    "__version__",
    "as_agent_run",
    "perform",
]
