# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Single-machine federation simulator.

The simulator wires a :class:`~erdos.server.Server`, a list of
:class:`~erdos.client.Client` objects and a
:class:`~erdos.apis.core.Controller` together and runs the whole federation
in one process. It is the fastest way to develop and debug a federated
workflow before deploying it across real, networked sites — the controller and
executors you test here are exactly the ones you would deploy.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .apis.core import Controller, FLComponent
from .apis.shareable import FLContext
from .client import Client
from .server import Server


class Simulator(FLComponent):
    """Run a federation locally for development and research."""

    def __init__(
        self,
        controller: Controller,
        clients: List[Client],
        server: Optional[Server] = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.clients = list(clients)
        self.server = server or Server()

    def run(self, fl_ctx: Optional[FLContext] = None) -> Any:
        fl_ctx = fl_ctx or FLContext()
        self.server.register_clients(self.clients)
        self.logger.info(
            "Simulating federation: 1 server, %d client(s)", len(self.clients)
        )
        return self.controller.control_flow(self.server, fl_ctx)
