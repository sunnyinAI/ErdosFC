# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The self-describing model payload.

:class:`FLModel` is Erdos FC's analogue of NVFlare's ``FLModel``/DXO: a single
structured container exchanged in both directions, carrying not just parameters
but *what they are*. The crucial field is :attr:`params_type`:

* ``FULL`` — ``params`` are complete model weights (replace the global model).
* ``DIFF`` — ``params`` are a weight *delta* (add to the global model).

That one distinction is what unlocks bandwidth-efficient updates, server-side
optimizers (FedOpt), diff-based privacy, and quantization without re-plumbing.
:class:`FLModel` converts to and from the wire :class:`Shareable`, so the
transport and codec never need to know about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .shareable import Shareable


class ParamsType:
    """How to interpret :attr:`FLModel.params`."""

    FULL = "FULL"  # complete weights
    DIFF = "DIFF"  # a delta to add to the global model


# Reserved Shareable.meta keys used to carry FLModel fields on the wire.
_PARAMS_TYPE = "params_type"
_METRICS = "metrics"
_CURRENT_ROUND = "current_round"
_TOTAL_ROUNDS = "total_rounds"


@dataclass
class FLModel:
    """A framework-agnostic model container exchanged between server and clients."""

    params: Dict[str, Any] = field(default_factory=dict)
    params_type: str = ParamsType.FULL
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    current_round: Optional[int] = None
    total_rounds: Optional[int] = None

    def to_shareable(self) -> Shareable:
        """Pack into a wire :class:`Shareable`."""
        meta = dict(self.meta)
        meta[_PARAMS_TYPE] = self.params_type
        if self.metrics:
            meta[_METRICS] = dict(self.metrics)
        if self.current_round is not None:
            meta[_CURRENT_ROUND] = self.current_round
        if self.total_rounds is not None:
            meta[_TOTAL_ROUNDS] = self.total_rounds
        return Shareable(params=dict(self.params), meta=meta)

    @classmethod
    def from_shareable(cls, shareable: Shareable) -> "FLModel":
        """Unpack from a wire :class:`Shareable`."""
        meta = dict(shareable.meta)
        return cls(
            params=shareable.params,
            params_type=meta.get(_PARAMS_TYPE, ParamsType.FULL),
            metrics=dict(meta.get(_METRICS, {})),
            meta=meta,
            current_round=meta.get(_CURRENT_ROUND),
            total_rounds=meta.get(_TOTAL_ROUNDS),
        )

    @staticmethod
    def params_type_of(shareable: Shareable) -> str:
        """Read the params_type of a Shareable without fully unpacking it."""
        return shareable.get_meta_prop(_PARAMS_TYPE, ParamsType.FULL)
