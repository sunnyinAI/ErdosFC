# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The federation server.

The server owns the authoritative global model and provides the communication
primitive the controller uses to reach clients. In this reference runtime the
``broadcast`` is executed in-process and sequentially (see :mod:`erdos.simulator`);
the same ``Controller`` would run unchanged against a networked transport that
implemented the same ``broadcast`` contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .apis.core import FLComponent
from .apis.shareable import FLContext, Shareable


class Server(FLComponent):
    """Holds the global model and dispatches tasks to registered clients."""

    def __init__(self) -> None:
        super().__init__()
        self.clients: List[Any] = []
        self.global_params: Optional[Dict[str, Any]] = None

    def register_clients(self, clients: List[Any]) -> None:
        self.clients = list(clients)

    def broadcast(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
    ) -> List[Tuple[str, Shareable]]:
        """Send ``task_name`` + ``shareable`` to every client; collect results.

        Each client receives its own independent copy of the payload so that
        local mutation can never leak across sites.
        """
        results: List[Tuple[str, Shareable]] = []
        for client in self.clients:
            fl_ctx.set_prop(FLContext.CURRENT_CLIENT, client.name)
            task = Shareable(
                params={k: v.detach().clone() for k, v in shareable.params.items()},
                meta=dict(shareable.meta),
            )
            result = client.run_task(task_name, task, fl_ctx)
            results.append((client.name, result))
        return results
