# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""FedOpt / FedAdam — server-side adaptive optimization.

Reddi et al. (2021), *Adaptive Federated Optimization*. Instead of replacing the
global model with the aggregated client average, FedOpt treats the aggregated
client *delta* as a pseudo-gradient and applies a stateful server optimizer
(Adam here) to it. This is a drop-in replacement for the model-update step, so it
reuses the entire :class:`~erdos.workflows.base.BaseModelController` round loop
and the standard :class:`FedAvgAggregator`; only :meth:`update_model` changes.

It works whether clients send ``FULL`` weights or ``DIFF`` updates — the delta is
recovered as ``aggregated - global`` in the FULL case.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..apis import tensor
from ..apis.model import ParamsType
from ..apis.shareable import Shareable
from .base import BaseModelController


class FedOpt(BaseModelController):
    """FedAdam: an Adam optimizer applied on the server to the aggregated delta.

    Args:
        server_lr: server learning rate (η).
        beta1, beta2: Adam moment decay rates.
        tau: numerical stability / adaptivity term (τ).
        All other arguments are inherited from :class:`BaseModelController`.
    """

    def __init__(
        self,
        *args: Any,
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        self._t = 0

    def update_model(
        self, global_params: Dict[str, Any], aggregated: Shareable, params_type: str
    ) -> Dict[str, Any]:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        new: Dict[str, Any] = {}
        for key, g in global_params.items():
            g_np = tensor.as_numpy(g).astype(np.float64)
            agg_np = tensor.as_numpy(aggregated.params[key]).astype(np.float64)
            # delta = average client (local - global)
            delta = agg_np if params_type == ParamsType.DIFF else agg_np - g_np

            m = self.beta1 * self._m.get(key, 0.0) + (1.0 - self.beta1) * delta
            v = self.beta2 * self._v.get(key, 0.0) + (1.0 - self.beta2) * (delta * delta)
            self._m[key], self._v[key] = m, v
            step = self.server_lr * (m / bc1) / (np.sqrt(v / bc2) + self.tau)

            backend, dtype = tensor.restore_info(g)
            new[key] = tensor.from_numpy(g_np + step, backend, dtype)
        return new
