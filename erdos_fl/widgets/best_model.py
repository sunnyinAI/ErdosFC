# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Best-model selection widget."""
from __future__ import annotations

from typing import Optional

from ..apis import tensor
from ..apis.core import FLComponent
from ..apis.shareable import FLContext
from ..engine import EventType, FLContextKey


class BestModelSelector(FLComponent):
    """Track the global model with the best value of a round metric.

    After every round it reads the metric ``key_metric`` from the round record
    and, if it improved, keeps a private clone of the global model. Exposes
    :attr:`best_params`, :attr:`best_metric` and :attr:`best_round`.
    """

    def __init__(self, key_metric: str = "test_acc", mode: str = "max") -> None:
        super().__init__()
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.key_metric = key_metric
        self.mode = mode
        self.best_metric: Optional[float] = None
        self.best_round: Optional[int] = None
        self.best_params: Optional[dict] = None

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        return value > self.best_metric if self.mode == "max" else value < self.best_metric

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type != EventType.AFTER_ROUND:
            return
        record = fl_ctx.get_prop(FLContextKey.ROUND_RECORD) or {}
        if self.key_metric not in record:
            return
        value = float(record[self.key_metric])
        if self._is_better(value):
            self.best_metric = value
            self.best_round = record.get("round")
            params = fl_ctx.get_prop(FLContextKey.GLOBAL_PARAMS)
            self.best_params = tensor.clone_params(params) if params else None
            self.logger.info(
                "New best %s=%.4f at round %s", self.key_metric, value, self.best_round
            )
