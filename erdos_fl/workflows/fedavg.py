# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The FedAvg server-side workflow (controller).

Each round the controller:

1. **scatters** the current global model to every client as a training task,
2. **gathers** the returned local updates,
3. **aggregates** them with the configured :class:`Aggregator`,
4. **applies** the result to the global model (replace for ``FULL`` updates, add
   for ``DIFF`` updates),
5. optionally **evaluates** (server-side ``evaluate_fn`` and/or a federated
   ``EVALUATE`` round) and fires lifecycle events for widgets.

This is the federated analogue of an ordinary training loop — the loop lives on
the server and each "step" is a full communication round. The shared machinery
lives in :class:`~erdos.workflows.base.BaseModelController`; FedAvg is just its
default configuration, so the same loop supports weight or weight-diff updates.
"""
from __future__ import annotations

from .base import BaseModelController

# Re-exported for backward compatibility (callbacks the controller accepts).
from .base import EvaluateFn, PersistFn  # noqa: F401


class FedAvg(BaseModelController):
    """Run Federated Averaging (McMahan et al., 2017) for a fixed number of rounds.

    Accepts the v0.1 keyword arguments unchanged (``num_rounds``,
    ``initial_params``, ``aggregator``, ``evaluate_fn``, ``persist_fn``) plus the
    v0.3 additions inherited from :class:`BaseModelController`: ``widgets``,
    ``federated_eval``, ``min_clients`` and ``timeout``.
    """
