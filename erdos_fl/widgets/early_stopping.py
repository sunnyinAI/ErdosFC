# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Early-stopping widget."""
from __future__ import annotations

from typing import Optional

from ..apis.core import FLComponent
from ..apis.shareable import FLContext
from ..engine import EventType, FLContextKey


class EarlyStopping(FLComponent):
    """Request an early stop when a metric stops improving.

    After each round, if ``key_metric`` has not improved by at least
    ``min_delta`` for ``patience`` consecutive rounds, it sets
    ``FLContextKey.SHOULD_STOP`` on the run context; the controller checks this
    flag and ends the run.
    """

    def __init__(
        self,
        key_metric: str = "test_acc",
        patience: int = 3,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.key_metric = key_metric
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.wait = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type != EventType.AFTER_ROUND:
            return
        record = fl_ctx.get_prop(FLContextKey.ROUND_RECORD) or {}
        if self.key_metric not in record:
            return
        value = float(record[self.key_metric])
        if self._improved(value):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                fl_ctx.set_prop(FLContextKey.SHOULD_STOP, True)
                self.logger.info(
                    "Early stop: %s did not improve for %d round(s) (best=%.4f)",
                    self.key_metric,
                    self.patience,
                    self.best,
                )
