# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Single-machine federation simulator.

The simulator wires a :class:`~erdos.server.Server`, a list of
:class:`~erdos.client.Client` objects and a
:class:`~erdos.apis.core.Controller` together and runs the whole federation in
one process. It is the fastest way to develop and debug a federated workflow
before deploying it across real, networked sites — the controller and executors
you test here are exactly the ones you would deploy.

Tasks still cross the real serialization boundary (every payload is encoded and
decoded by the server's :class:`~erdos.codec.base.Codec`), and each client task
runs in its own child :class:`FLContext` scope, so the in-process run exercises
the same code path a networked transport would.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .apis.core import Controller, FLComponent
from .apis.shareable import FLContext
from .client import Client
from .codec import Codec
from .engine import RunEngine
from .server import Server
from .transport import Transport


class Simulator(FLComponent):
    """Run a federation locally for development and research."""

    def __init__(
        self,
        controller: Controller,
        clients: List[Client],
        server: Optional[Server] = None,
        transport: Optional[Transport] = None,
        codec: Optional[Codec] = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.clients = list(clients)
        self.server = server or Server(transport=transport, codec=codec)

    def run(self, fl_ctx: Optional[FLContext] = None) -> Any:
        fl_ctx = fl_ctx or FLContext()
        self.server.register_clients(self.clients)

        # Stand up the run engine (event bus) so components and widgets can plug
        # into the lifecycle, and publish it on the run-scope context.
        engine = RunEngine()
        engine.register(self.server, self.controller, *self.clients)
        engine.attach(fl_ctx)

        self.logger.info(
            "Simulating federation: 1 server, %d client(s)", len(self.clients)
        )
        transport = self.server.transport
        if transport is not None:
            transport.start()
        try:
            return self.controller.control_flow(self.server, fl_ctx)
        finally:
            if transport is not None:
                transport.stop()
