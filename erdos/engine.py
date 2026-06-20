# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The run engine and event bus.

Every :class:`~erdos.apis.core.FLComponent` can react to lifecycle events
(``START_RUN``, ``BEFORE_AGGREGATE``, ``AFTER_AGGREGATE``, ``END_RUN`` …) by
overriding :meth:`~erdos.apis.core.FLComponent.handle_event`. The
:class:`RunEngine` keeps the set of registered components and fans an event out
to each one. This is what lets cross-cutting concerns — best-model selection,
early stopping, experiment tracking, audit logging — be added as *widgets*
without editing the controller's ``control_flow``.

The engine is published on the run-scope :class:`FLContext` under
``FLContext.ENGINE``, so any component holding a context can fire events or add
itself as a handler.
"""
from __future__ import annotations

from typing import List

from .apis.shareable import FLContext


class EventType:
    """Canonical lifecycle event names fired during a run."""

    START_RUN = "start_run"
    END_RUN = "end_run"
    BEFORE_ROUND = "before_round"
    AFTER_ROUND = "after_round"
    BEFORE_BROADCAST = "before_broadcast"
    AFTER_BROADCAST = "after_broadcast"
    BEFORE_AGGREGATE = "before_aggregate"
    AFTER_AGGREGATE = "after_aggregate"
    AFTER_MODEL_UPDATE = "after_model_update"
    BEFORE_TASK_EXECUTION = "before_task_execution"
    AFTER_TASK_EXECUTION = "after_task_execution"


class FLContextKey:
    """Well-known keys components publish on the context for event handlers."""

    ROUND_RECORD = "round_record"          # dict of this round's metrics
    GLOBAL_PARAMS = "global_params"        # the current global model params
    AGGREGATION_RESULT = "aggregation_result"  # Shareable from the aggregator
    CLIENT_RESULTS = "client_results"      # list[(name, Shareable)] this round
    SHOULD_STOP = "should_stop"            # set True to request early stop


class RunEngine:
    """Holds registered components and dispatches events to them."""

    def __init__(self) -> None:
        self._components: List[object] = []

    def register(self, *components: object) -> None:
        for c in components:
            if c not in self._components:
                self._components.append(c)

    # Alias used by widgets that add themselves dynamically.
    add_component = register

    def attach(self, fl_ctx: FLContext) -> None:
        """Publish this engine on the run-scope context."""
        fl_ctx.set_prop(FLContext.ENGINE, self)

    def fire_event(self, event_type: str, fl_ctx: FLContext) -> None:
        """Deliver ``event_type`` to every registered component in order."""
        for c in list(self._components):
            handler = getattr(c, "handle_event", None)
            if handler is None:
                continue
            try:
                handler(event_type, fl_ctx)
            except Exception as exc:  # noqa: BLE001 - a handler must not kill the run
                name = type(c).__name__
                import logging

                logging.getLogger("RunEngine").warning(
                    "Handler %s failed on event %s: %s", name, event_type, exc
                )
