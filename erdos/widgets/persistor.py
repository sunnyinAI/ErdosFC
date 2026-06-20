# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Model-persistence widget."""
from __future__ import annotations

from typing import Callable

from ..apis.core import FLComponent
from ..apis.shareable import FLContext
from ..engine import EventType, FLContextKey

# save_fn(global_params, round_index) -> None
SaveFn = Callable[[dict, int], None]


class ModelPersistor(FLComponent):
    """Persist the global model every ``every`` rounds via a user ``save_fn``.

    ``save_fn`` is framework-agnostic — pass ``lambda p, r: torch.save(p, path)``
    for PyTorch, or a NumPy/``np.savez`` saver — keeping the core free of any
    deep-learning dependency.
    """

    def __init__(self, save_fn: SaveFn, every: int = 1) -> None:
        super().__init__()
        self.save_fn = save_fn
        self.every = max(1, int(every))

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type != EventType.AFTER_ROUND:
            return
        record = fl_ctx.get_prop(FLContextKey.ROUND_RECORD) or {}
        rnd = int(record.get("round", 0))
        if rnd % self.every != 0:
            return
        params = fl_ctx.get_prop(FLContextKey.GLOBAL_PARAMS)
        if params is not None:
            self.save_fn(params, rnd)
            self.logger.debug("Persisted global model at round %d", rnd)
