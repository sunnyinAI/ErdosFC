# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A simple differential-privacy-style filter.

``GaussianPrivacyFilter`` runs on the *client result path* — it transforms a
client's update just before it leaves the site, so the server never sees the
raw local weights. It supports optional global-L2 clipping followed by additive
Gaussian noise, the two ingredients of the Gaussian mechanism.

.. note::
   This is an **illustrative** mechanism intended to show where privacy logic
   plugs into Erdos FC. It is not a calibrated ``(epsilon, delta)`` guarantee;
   for production privacy use a properly accounted DP-SGD optimiser and a moments
   accountant. ``clip_norm`` defaults to ``None`` (no clipping) and ``sigma`` is
   small so the demo keeps converging.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..apis.core import Filter
from ..apis.shareable import FLContext, Shareable


class GaussianPrivacyFilter(Filter):
    """Optionally clip the update to an L2 ball, then add Gaussian noise.

    Args:
        sigma: standard deviation of the Gaussian noise added to each floating
            tensor. ``0`` disables noise.
        clip_norm: if set, the concatenation of all floating tensors is scaled so
            its global L2 norm does not exceed this value before noise is added.
        seed: optional seed for reproducible noise.
    """

    def __init__(
        self,
        sigma: float = 0.01,
        clip_norm: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator().manual_seed(int(seed))

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        params = shareable.params
        float_keys = [
            k for k, v in params.items()
            if torch.is_tensor(v) and v.is_floating_point()
        ]
        if not float_keys:
            return shareable

        # 1) Optional global L2 clipping.
        if self.clip_norm is not None:
            total_sq = sum(torch.sum(params[k].detach() ** 2).item() for k in float_keys)
            norm = total_sq ** 0.5
            if norm > self.clip_norm:
                scale = self.clip_norm / (norm + 1e-12)
                for k in float_keys:
                    params[k] = params[k] * scale

        # 2) Additive Gaussian noise.
        if self.sigma > 0:
            for k in float_keys:
                t = params[k]
                if self._generator is not None:
                    noise = torch.randn(
                        t.shape, generator=self._generator, dtype=t.dtype
                    ) * self.sigma
                else:
                    noise = torch.randn_like(t) * self.sigma
                params[k] = t + noise

        shareable.set_meta_prop("privacy_sigma", self.sigma)
        if self.clip_norm is not None:
            shareable.set_meta_prop("privacy_clip_norm", self.clip_norm)
        return shareable
