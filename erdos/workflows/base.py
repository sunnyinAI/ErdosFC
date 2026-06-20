# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Base server-side workflow (the model-controller pattern).

:class:`BaseModelController` runs the round loop common to every weight-based
federated algorithm: scatter the global model, gather updates, aggregate, apply
the result, evaluate, persist, and fire lifecycle events the whole way through.
Concrete algorithms subclass it and override a small set of hooks:

* :meth:`prepare_task` — what to send clients each round (default: the global model);
* :meth:`update_model` — how to fold the aggregated result into the global model
  (default: replace for ``FULL``, add for ``DIFF``; FedOpt overrides this).

Cross-cutting concerns (best-model selection, early stopping, persistence,
experiment tracking) are *not* coded here — they are widgets on the event bus.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..apis import tensor
from ..apis.core import Aggregator, Controller, TaskName
from ..apis.model import FLModel, ParamsType
from ..apis.shareable import FLContext, Shareable
from ..engine import EventType, FLContextKey, RunEngine

EvaluateFn = Callable[[Dict[str, Any], int], Optional[Dict[str, float]]]
PersistFn = Callable[[Dict[str, Any], int], None]

# Result meta keys that are not user metrics.
_NON_METRIC_KEYS = {"num_samples", "client", "params_type", "metrics", "current_round", "total_rounds"}


class BaseModelController(Controller):
    """The shared scatter → gather → aggregate → apply round loop."""

    def __init__(
        self,
        num_rounds: int,
        initial_params: Dict[str, Any],
        aggregator: Aggregator,
        *,
        evaluate_fn: Optional[EvaluateFn] = None,
        persist_fn: Optional[PersistFn] = None,
        widgets: Optional[list] = None,
        federated_eval: bool = False,
        min_clients: Optional[int] = None,
        timeout: Optional[float] = None,
        train_task: str = TaskName.TRAIN,
        eval_task: str = TaskName.EVALUATE,
    ) -> None:
        super().__init__()
        self.num_rounds = int(num_rounds)
        self.initial_params = initial_params
        self.aggregator = aggregator
        self.evaluate_fn = evaluate_fn
        self.persist_fn = persist_fn
        self.widgets = list(widgets or [])
        self.federated_eval = federated_eval
        self.min_clients = min_clients
        self.timeout = timeout
        self.train_task = train_task
        self.eval_task = eval_task
        self.history: List[Dict[str, Any]] = []
        self.global_params: Optional[Dict[str, Any]] = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _clone(params: Dict[str, Any]) -> Dict[str, Any]:
        return tensor.clone_params(params)

    @staticmethod
    def _format(record: Dict[str, Any]) -> str:
        parts = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
        ]
        return ", ".join(parts)

    def _set_global(self, server, fl_ctx: FLContext, params: Dict[str, Any]) -> None:
        self.global_params = params
        server.global_params = params
        fl_ctx.set_prop(FLContextKey.GLOBAL_PARAMS, params)

    def _begin_run(self, fl_ctx: FLContext) -> RunEngine:
        """Attach an engine, register widgets, and publish run-scope props."""
        engine = fl_ctx.get_engine()
        if engine is None:
            engine = RunEngine()
            engine.attach(fl_ctx)
            engine.register(self)
        if self.widgets:
            engine.register(*self.widgets)
        fl_ctx.set_prop(FLContext.NUM_ROUNDS, self.num_rounds)
        return engine

    def _record_round(self, server, rnd: int, n_accepted: int, global_params, fl_ctx: FLContext) -> Dict[str, Any]:
        """Build the round metrics record, running any configured evaluation."""
        record: Dict[str, Any] = {"round": rnd + 1, "clients": n_accepted}
        if self.evaluate_fn is not None:
            record.update(self.evaluate_fn(global_params, rnd) or {})
        if self.federated_eval:
            record.update(self._federated_eval(server, global_params, fl_ctx))
        self.history.append(record)
        fl_ctx.set_prop(FLContextKey.ROUND_RECORD, record)
        self.logger.info("[round %d/%d] %s", rnd + 1, self.num_rounds, self._format(record))
        return record

    # -- overridable hooks -------------------------------------------------
    def prepare_task(self, global_params: Dict[str, Any], rnd: int) -> Shareable:
        """Build the Shareable scattered to clients this round."""
        return FLModel(
            params=self._clone(global_params),
            params_type=ParamsType.FULL,
            current_round=rnd,
            total_rounds=self.num_rounds,
        ).to_shareable()

    def update_model(
        self, global_params: Dict[str, Any], aggregated: Shareable, params_type: str
    ) -> Dict[str, Any]:
        """Fold the aggregated client result into the global model."""
        if params_type == ParamsType.DIFF:
            return tensor.add_params(global_params, aggregated.params)
        return dict(aggregated.params)

    # -- aggregation -------------------------------------------------------
    def aggregate_results(
        self, results: List[Tuple[str, Shareable]], fl_ctx: FLContext
    ) -> Tuple[Shareable, int, str]:
        self.aggregator.reset(fl_ctx)
        n_accepted = 0
        params_type = ParamsType.FULL
        for _name, res in results:
            if self.aggregator.accept(res, fl_ctx):
                n_accepted += 1
                params_type = FLModel.params_type_of(res)
        return self.aggregator.aggregate(fl_ctx), n_accepted, params_type

    # -- federated evaluation (the EVALUATE task, wired end-to-end) --------
    def _federated_eval(self, server, global_params: Dict[str, Any], fl_ctx: FLContext) -> Dict[str, float]:
        task = FLModel(
            params=self._clone(global_params), params_type=ParamsType.FULL
        ).to_shareable()
        results = server.broadcast(self.eval_task, task, fl_ctx, timeout=self.timeout)
        totals: Dict[str, float] = {}
        weight_sum = 0.0
        for _name, res in results:
            w = float(res.get_meta_prop("num_samples", 1.0))
            weight_sum += w
            metrics = dict(res.meta.get("metrics", {}))
            for k, v in res.meta.items():
                if k not in _NON_METRIC_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics.setdefault(k, v)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + w * float(v)
        if weight_sum <= 0:
            return {}
        return {f"fed_{k}": v / weight_sum for k, v in totals.items()}

    # -- the run loop ------------------------------------------------------
    def control_flow(self, server, fl_ctx: FLContext) -> Dict[str, Any]:
        self._begin_run(fl_ctx)

        global_params = self._clone(self.initial_params)
        self._set_global(server, fl_ctx, global_params)

        self.logger.info(
            "Starting %s: %d rounds over %d client(s)",
            type(self).__name__,
            self.num_rounds,
            len(server.clients),
        )
        self.fire_event(EventType.START_RUN, fl_ctx)

        for rnd in range(self.num_rounds):
            fl_ctx.set_prop(FLContext.CURRENT_ROUND, rnd)
            self.fire_event(EventType.BEFORE_ROUND, fl_ctx)

            # 1) scatter
            task = self.prepare_task(global_params, rnd)
            self.fire_event(EventType.BEFORE_BROADCAST, fl_ctx)
            results = server.broadcast(
                self.train_task, task, fl_ctx, min_responses=self.min_clients, timeout=self.timeout
            )
            fl_ctx.set_prop(FLContextKey.CLIENT_RESULTS, results)
            self.fire_event(EventType.AFTER_BROADCAST, fl_ctx)

            # 2) gather + aggregate
            self.fire_event(EventType.BEFORE_AGGREGATE, fl_ctx)
            aggregated, n_accepted, params_type = self.aggregate_results(results, fl_ctx)
            fl_ctx.set_prop(FLContextKey.AGGREGATION_RESULT, aggregated)
            self.fire_event(EventType.AFTER_AGGREGATE, fl_ctx)

            # 3) apply to the global model
            global_params = self.update_model(global_params, aggregated, params_type)
            self._set_global(server, fl_ctx, global_params)
            self.fire_event(EventType.AFTER_MODEL_UPDATE, fl_ctx)

            # 4) evaluate + record
            self._record_round(server, rnd, n_accepted, global_params, fl_ctx)

            # 5) persist (callback path) + lifecycle event for widgets
            if self.persist_fn is not None:
                self.persist_fn(global_params, rnd)
            self.fire_event(EventType.AFTER_ROUND, fl_ctx)

            if fl_ctx.get_prop(FLContextKey.SHOULD_STOP):
                self.logger.info("Early stop requested; ending after round %d.", rnd + 1)
                break

        self.fire_event(EventType.END_RUN, fl_ctx)
        self.logger.info("%s complete.", type(self).__name__)
        return global_params
