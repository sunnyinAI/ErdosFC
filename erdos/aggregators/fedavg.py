# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Federated Averaging (FedAvg) aggregator.

Implements the weighted parameter averaging of McMahan et al. (2017),
*Communication-Efficient Learning of Deep Networks from Decentralized Data*.
Each client contribution is weighted by its number of local training samples.

The aggregator is intentionally free of any deep-learning import: it only relies
on the tensor arithmetic methods (``detach``, ``clone``, ``to``, ``+``, ``*``,
``/``) that both PyTorch tensors and NumPy-backed tensors expose, so the same
code works regardless of the trainer's framework.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..apis.core import Aggregator
from ..apis.shareable import FLContext, Shareable


class FedAvgAggregator(Aggregator):
    """Sample-count weighted average of client parameter dictionaries."""

    def __init__(self, weight_key: str = "num_samples") -> None:
        super().__init__()
        self.weight_key = weight_key
        self._sum: Optional[Dict[str, Any]] = None
        self._dtypes: Dict[str, Any] = {}
        self._total_weight: float = 0.0
        self._num_contributors: int = 0

    def reset(self, fl_ctx: FLContext) -> None:
        self._sum = None
        self._dtypes = {}
        self._total_weight = 0.0
        self._num_contributors = 0

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        params = shareable.params
        if not params:
            self.logger.warning("Ignoring empty contribution from %s", shareable.get_meta_prop("client"))
            return False

        weight = float(shareable.get_meta_prop(self.weight_key, 1.0))
        if weight <= 0:
            weight = 1.0

        if self._sum is None:
            self._sum = {}
            for key, value in params.items():
                self._dtypes[key] = value.dtype
                # float() promotes integer buffers so averaging is well-defined;
                # the original dtype is restored in aggregate().
                self._sum[key] = value.detach().float() * weight
        else:
            for key, value in params.items():
                self._sum[key] += value.detach().float() * weight

        self._total_weight += weight
        self._num_contributors += 1
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if self._sum is None or self._total_weight == 0.0:
            raise RuntimeError("FedAvgAggregator.aggregate() called with no accepted contributions.")

        averaged: Dict[str, Any] = {}
        for key, summed in self._sum.items():
            avg = summed / self._total_weight
            # Restore the original dtype (e.g. cast integer buffers back to int).
            averaged[key] = avg.to(self._dtypes[key])

        meta = {
            "num_contributors": self._num_contributors,
            "total_weight": self._total_weight,
        }
        self.logger.debug(
            "Aggregated %d contributions (total weight %.1f)",
            self._num_contributors,
            self._total_weight,
        )
        return Shareable(params=averaged, meta=meta)
