"""Drive synchronous agent state machines across an asynchronous runtime boundary.

The module owns run and effect lifecycle orchestration: it advances an
:class:`~purrcept_core.run.AgentRun`, delegates each yielded effect to an
executor, and emits ordered lifecycle events. Executors own capability
implementation, while agent flows retain ordinary Python exception semantics.

The primary entry point is :class:`AgentDriver`. Each call to
:meth:`AgentDriver.run` creates isolated run-local identifiers, timing state,
and progress channels. Event-dispatch failures and task cancellation are
infrastructure outcomes; they are never thrown into agent code as effect
failures.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping
from datetime import UTC, datetime
from inspect import isawaitable
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from .context import ExecutionContext, ProgressCallback
from .effect import Effect
from .errors import EventDispatchError, InvalidRunStateError
from .events import (
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
)
from .executor import EffectExecutor
from .flow import AgentFlow
from .run import AgentRun, Returned, Yielded, as_agent_run

HostT = TypeVar("HostT")
ResultT = TypeVar("ResultT")

Clock = Callable[[], datetime]
"""Wall-clock provider used to timestamp lifecycle events."""

Monotonic = Callable[[], float]
"""Monotonic clock used only to measure run and effect durations."""

IdFactory = Callable[[], str]
"""Factory producing trace identifiers for runs and effects."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


def _require_yielded(state: object) -> Yielded:
    """Narrow a custom run state before the driver reads its effect."""

    if isinstance(state, Yielded):
        return state
    raise TypeError(
        "AgentRun returned an unsupported state "
        f"{type(state).__module__}.{type(state).__qualname__}."
    )


class AgentDriver(Generic[HostT]):
    """Coordinate one or more independent agent runs with shared runtime services.

    Applications create a driver from an executor and optional event sink, then
    call :meth:`run` from an asyncio task. The driver owns orchestration only:
    it does not interpret effects, persist events, or retain per-run state after
    completion. The configured executor and sink are reused across calls, so
    their implementations own any concurrency restrictions.

    Ordinary executor exceptions are reported and thrown back into the active
    run. Cancellation, invalid custom run states, and event infrastructure
    failures close the run and propagate directly to the caller.
    """

    __slots__ = ("_clock", "_event_sink", "_executor", "_id_factory", "_monotonic")

    def __init__(
        self,
        executor: EffectExecutor[HostT],
        *,
        event_sink: EventSink | None = None,
        clock: Clock = _utc_now,
        monotonic: Monotonic = time.monotonic,
        id_factory: IdFactory = _new_id,
    ) -> None:
        self._executor = executor
        self._event_sink = event_sink if event_sink is not None else NullEventSink()
        self._clock = clock
        self._monotonic = monotonic
        self._id_factory = id_factory

    @property
    def executor(self) -> EffectExecutor[HostT]:
        """The executor used for every yielded effect."""

        return self._executor

    @property
    def event_sink(self) -> EventSink:
        """The sink receiving lifecycle events."""

        return self._event_sink

    async def run(
        self,
        run: AgentRun[ResultT] | AgentFlow[ResultT],
        *,
        host: HostT,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> ResultT:
        """Drive ``run`` to completion.

        Ordinary executor exceptions are emitted as ``EffectFailed`` and thrown
        back into the run. Cancellation and other ``BaseException`` subclasses
        are never delivered to agent code. The run is closed on every abnormal
        exit, and a close failure is attached to the primary failure rather than
        replacing it.
        """

        agent_run = as_agent_run(run)

        def close_preserving(primary_error: BaseException) -> None:
            """Close the run without allowing cleanup to replace its primary error."""

            try:
                agent_run.close()
            except BaseException as close_error:
                primary_error.add_note(
                    "AgentRun.close() also raised "
                    f"{type(close_error).__module__}.{type(close_error).__qualname__}: "
                    f"{close_error}"
                )

        # Resolve all preflight values before the first lifecycle event. A
        # preflight failure still closes a caller-supplied run that may own
        # resources, but it cannot produce a traceable RunFailed event because
        # the run identity or clock may itself be the failing dependency.
        try:
            resolved_run_id = self._id_factory() if run_id is None else run_id
            metadata_snapshot: Mapping[str, Any] = MappingProxyType(dict(metadata or {}))
            run_started_at = self._monotonic()
        except BaseException as error:
            close_preserving(error)
            raise

        try:
            # The main loop deliberately keeps every lifecycle transition
            # visible: announce the run, advance it, execute one effect, and
            # feed exactly one success or ordinary failure back into agent code.
            self._emit(
                RunStarted(
                    run_id=resolved_run_id,
                    occurred_at=self._clock(),
                    metadata=metadata_snapshot,
                )
            )
            state = agent_run.start()
            step_index = 0

            while True:
                if isinstance(state, Returned):
                    result = state.value
                    self._emit(
                        RunSucceeded(
                            run_id=resolved_run_id,
                            occurred_at=self._clock(),
                            result=result,
                            elapsed_seconds=self._monotonic() - run_started_at,
                        )
                    )
                    return result

                effect = _require_yielded(state).effect
                effect_id = self._id_factory()
                effect_started_at = self._monotonic()
                self._emit(
                    EffectStarted(
                        run_id=resolved_run_id,
                        occurred_at=self._clock(),
                        effect_id=effect_id,
                        step_index=step_index,
                        effect=effect,
                    )
                )
                progress_callback, finish_progress = self._make_progress_channel(
                    run_id=resolved_run_id,
                    effect_id=effect_id,
                    step_index=step_index,
                    effect=effect,
                )
                context = ExecutionContext(
                    run_id=resolved_run_id,
                    effect_id=effect_id,
                    step_index=step_index,
                    host=host,
                    metadata=metadata_snapshot,
                    _progress_callback=progress_callback,
                )

                try:
                    try:
                        value = await self._executor.execute(effect, context)
                    except BaseException as execution_error:
                        # Ordering: finalize the progress channel before choosing
                        # the primary effect outcome. A previously latched
                        # observer failure wins because agent code must never
                        # receive an effect result whose progress was not
                        # delivered consistently.
                        progress_error = finish_progress()
                        if progress_error is not None and progress_error is not execution_error:
                            progress_error.add_note(
                                "Effect execution also raised "
                                f"{type(execution_error).__module__}."
                                f"{type(execution_error).__qualname__}: "
                                f"{execution_error}"
                            )
                            raise progress_error from progress_error.__cause__
                        raise
                    else:
                        progress_error = finish_progress()
                        if progress_error is not None:
                            raise progress_error
                except asyncio.CancelledError:
                    raise
                except EventDispatchError:
                    # In practice this branch is reached when context.progress()
                    # cannot dispatch. It must never be thrown into the flow.
                    raise
                except Exception as error:
                    self._emit(
                        EffectFailed(
                            run_id=resolved_run_id,
                            occurred_at=self._clock(),
                            effect_id=effect_id,
                            step_index=step_index,
                            effect=effect,
                            error=error,
                            elapsed_seconds=self._monotonic() - effect_started_at,
                        )
                    )
                    state = agent_run.throw(error)
                else:
                    self._emit(
                        EffectSucceeded(
                            run_id=resolved_run_id,
                            occurred_at=self._clock(),
                            effect_id=effect_id,
                            step_index=step_index,
                            effect=effect,
                            result=value,
                            elapsed_seconds=self._monotonic() - effect_started_at,
                        )
                    )
                    state = agent_run.send(value)

                step_index += 1

        except asyncio.CancelledError as error:
            close_preserving(error)
            try:
                self._emit(
                    RunCancelled(
                        run_id=resolved_run_id,
                        occurred_at=self._clock(),
                        elapsed_seconds=self._monotonic() - run_started_at,
                    )
                )
            except EventDispatchError as dispatch_error:
                error.add_note(
                    "Dispatching RunCancelled also failed; the EventDispatchError "
                    "is attached as this cancellation's cause."
                )
                raise error from dispatch_error
            raise
        except EventDispatchError as error:
            close_preserving(error)
            raise
        except Exception as error:
            try:
                self._emit(
                    RunFailed(
                        run_id=resolved_run_id,
                        occurred_at=self._clock(),
                        error=error,
                        elapsed_seconds=self._monotonic() - run_started_at,
                    )
                )
            finally:
                close_preserving(error)
            raise
        except BaseException as error:
            close_preserving(error)
            raise

    def _make_progress_channel(
        self,
        *,
        run_id: str,
        effect_id: str,
        step_index: int,
        effect: Effect[Any],
    ) -> tuple[ProgressCallback, Callable[[], EventDispatchError | None]]:
        """Create a progress callback whose validity ends with the current effect.

        The returned finalizer deactivates retained contexts and exposes the
        first dispatch failure even when an executor caught that failure. This
        prevents an executor from accidentally turning an observer failure into
        a successful effect result.
        """

        active = True
        dispatch_error: EventDispatchError | None = None

        def report_progress(payload: object) -> None:
            """Emit progress while latching the first infrastructure failure."""

            nonlocal dispatch_error
            if not active:
                raise InvalidRunStateError(
                    f"Cannot report progress after effect {effect_id!r} has finished."
                )
            if dispatch_error is not None:
                raise dispatch_error
            try:
                self._emit(
                    EffectProgress(
                        run_id=run_id,
                        occurred_at=self._clock(),
                        effect_id=effect_id,
                        step_index=step_index,
                        effect=effect,
                        payload=payload,
                    )
                )
            except EventDispatchError as error:
                dispatch_error = error
                raise

        def finish_progress() -> EventDispatchError | None:
            """Close the channel and return any latched dispatch failure."""

            nonlocal active
            active = False
            return dispatch_error

        return report_progress, finish_progress

    def _emit(self, event: RunEvent) -> None:
        """Dispatch one event and translate sink contract violations uniformly."""

        try:
            result: object = self._event_sink.emit(event)
            if isawaitable(result):
                if isinstance(result, Coroutine):
                    result.close()
                raise TypeError("EventSink.emit() must be synchronous and return None.")
            if result is not None:
                raise TypeError(f"EventSink.emit() must return None, got {type(result).__name__}.")
        except EventDispatchError:
            raise
        except Exception as error:
            raise EventDispatchError(event) from error


__all__ = ["AgentDriver", "Clock", "IdFactory", "Monotonic"]
