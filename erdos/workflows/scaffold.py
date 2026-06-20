# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""SCAFFOLD — Stochastic Controlled Averaging (Karimireddy et al., 2020).

SCAFFOLD corrects client drift under heterogeneous (non-IID) data by carrying
*control variates*: a server control ``c`` (sent each round alongside the global
model) and a per-client control ``c_i`` (kept on the site). Each local step is
corrected by ``-c_i + c``; clients return both a model delta and a
control-variate delta, and the server updates the global model and ``c``.

The control variates ride in a ``"controls"`` section of the wire
:class:`Shareable` (serialized by the codec just like ``params``). Client support
lives in :class:`~erdos.executors.numpy_trainer.NumpyTrainer`, which applies the
correction when a task carries controls.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..apis import tensor
from ..apis.model import FLModel, ParamsType
from ..apis.shareable import FLContext
from ..engine import EventType, FLContextKey
from .base import BaseModelController


class Scaffold(BaseModelController):
    """SCAFFOLD controller (equal-weight averaging of client deltas).

    Args:
        server_lr: global learning rate η_g applied to the averaged model delta.
        All other arguments are inherited from :class:`BaseModelController`.
    """

    def __init__(self, *args: Any, server_lr: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_lr = float(server_lr)

    def control_flow(self, server, fl_ctx: FLContext) -> Dict[str, Any]:
        self._begin_run(fl_ctx)
        x = self._clone(self.initial_params)
        # Server control variate c, in float64 NumPy keyed like the model.
        c: Dict[str, np.ndarray] = {
            k: np.zeros(tensor.as_numpy(v).shape, dtype=np.float64) for k, v in x.items()
        }
        self._set_global(server, fl_ctx, x)
        n_clients = max(len(server.clients), 1)

        self.logger.info(
            "Starting SCAFFOLD: %d rounds over %d client(s)", self.num_rounds, n_clients
        )
        self.fire_event(EventType.START_RUN, fl_ctx)

        for rnd in range(self.num_rounds):
            fl_ctx.set_prop(FLContext.CURRENT_ROUND, rnd)
            self.fire_event(EventType.BEFORE_ROUND, fl_ctx)

            task = FLModel(
                params=self._clone(x),
                params_type=ParamsType.FULL,
                current_round=rnd,
                total_rounds=self.num_rounds,
            ).to_shareable()
            task["controls"] = {k: v.copy() for k, v in c.items()}

            self.fire_event(EventType.BEFORE_BROADCAST, fl_ctx)
            results = server.broadcast(
                self.train_task, task, fl_ctx, min_responses=self.min_clients, timeout=self.timeout
            )
            fl_ctx.set_prop(FLContextKey.CLIENT_RESULTS, results)
            self.fire_event(EventType.AFTER_BROADCAST, fl_ctx)

            # Sum client model-deltas and control-deltas (equal weight).
            self.fire_event(EventType.BEFORE_AGGREGATE, fl_ctx)
            dy_sum: Dict[str, np.ndarray] = {}
            dc_sum: Dict[str, np.ndarray] = {}
            cnt = 0
            for _name, res in results:
                cnt += 1
                for k, v in res.params.items():
                    dy_sum[k] = dy_sum.get(k, 0.0) + tensor.as_numpy(v).astype(np.float64)
                for k, v in res.get("controls", {}).items():
                    dc_sum[k] = dc_sum.get(k, 0.0) + tensor.as_numpy(v).astype(np.float64)
            self.fire_event(EventType.AFTER_AGGREGATE, fl_ctx)

            if cnt > 0:
                for k in x:
                    backend, dtype = tensor.restore_info(x[k])
                    new = tensor.as_numpy(x[k]).astype(np.float64) + self.server_lr * (dy_sum[k] / cnt)
                    x[k] = tensor.from_numpy(new, backend, dtype)
                for k in c:
                    if k in dc_sum:  # c <- c + (1/N) * sum(delta_c_i)
                        c[k] = c[k] + dc_sum[k] / n_clients
                self._set_global(server, fl_ctx, x)
            self.fire_event(EventType.AFTER_MODEL_UPDATE, fl_ctx)

            self._record_round(server, rnd, cnt, x, fl_ctx)
            if self.persist_fn is not None:
                self.persist_fn(x, rnd)
            self.fire_event(EventType.AFTER_ROUND, fl_ctx)
            if fl_ctx.get_prop(FLContextKey.SHOULD_STOP):
                self.logger.info("Early stop requested; ending after round %d.", rnd + 1)
                break

        self.fire_event(EventType.END_RUN, fl_ctx)
        self.logger.info("SCAFFOLD complete.")
        return x
