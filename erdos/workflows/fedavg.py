# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The FedAvg server-side workflow (controller).

Each round the controller:

1. **scatters** the current global model to every client as a training task,
2. **gathers** the returned local updates,
3. **aggregates** them with the configured :class:`Aggregator`,
4. optionally **evaluates** and **persists** the new global model.

This is the federated analogue of an ordinary training loop — the loop lives on
the server and each "step" is a full communication round.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..apis.core import Aggregator, Controller, TaskName
from ..apis.shareable import FLContext, Shareable

# evaluate_fn(global_params, round_index) -> dict of float metrics (or None)
EvaluateFn = Callable[[Dict[str, Any], int], Optional[Dict[str, float]]]
# persist_fn(global_params, round_index) -> None
PersistFn = Callable[[Dict[str, Any], int], None]


class FedAvg(Controller):
    """Run Federated Averaging for a fixed number of rounds."""

    def __init__(
        self,
        num_rounds: int,
        initial_params: Dict[str, Any],
        aggregator: Aggregator,
        evaluate_fn: Optional[EvaluateFn] = None,
        persist_fn: Optional[PersistFn] = None,
    ) -> None:
        super().__init__()
        self.num_rounds = int(num_rounds)
        self.initial_params = initial_params
        self.aggregator = aggregator
        self.evaluate_fn = evaluate_fn
        self.persist_fn = persist_fn
        self.history: List[Dict[str, Any]] = []
        self.global_params: Optional[Dict[str, Any]] = None

    @staticmethod
    def _clone(params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.detach().clone() for k, v in params.items()}

    @staticmethod
    def _format(record: Dict[str, Any]) -> str:
        parts = []
        for k, v in record.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        return ", ".join(parts)

    def control_flow(self, server, fl_ctx: FLContext) -> Dict[str, Any]:
        global_params = self._clone(self.initial_params)
        self.global_params = global_params
        server.global_params = global_params
        fl_ctx.set_prop(FLContext.NUM_ROUNDS, self.num_rounds)

        self.logger.info(
            "Starting FedAvg: %d rounds over %d client(s)",
            self.num_rounds,
            len(server.clients),
        )

        for rnd in range(self.num_rounds):
            fl_ctx.set_prop(FLContext.CURRENT_ROUND, rnd)

            # 1) scatter the current global model
            task = Shareable(
                params=self._clone(global_params),
                meta={FLContext.CURRENT_ROUND: rnd, FLContext.NUM_ROUNDS: self.num_rounds},
            )
            results = server.broadcast(TaskName.TRAIN, task, fl_ctx)

            # 2) gather + 3) aggregate
            self.aggregator.reset(fl_ctx)
            n_accepted = 0
            for _client_name, result in results:
                if self.aggregator.accept(result, fl_ctx):
                    n_accepted += 1
            aggregated = self.aggregator.aggregate(fl_ctx)
            global_params = aggregated.params
            self.global_params = global_params
            server.global_params = global_params

            # 4) evaluate + persist
            record: Dict[str, Any] = {"round": rnd + 1, "clients": n_accepted}
            if self.evaluate_fn is not None:
                metrics = self.evaluate_fn(global_params, rnd) or {}
                record.update(metrics)
            self.history.append(record)
            self.logger.info("[round %d/%d] %s", rnd + 1, self.num_rounds, self._format(record))

            if self.persist_fn is not None:
                self.persist_fn(global_params, rnd)

        self.logger.info("FedAvg complete.")
        return global_params
