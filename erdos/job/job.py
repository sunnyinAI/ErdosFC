# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The Job: a reproducible, launchable unit of federated work.

A :class:`Job` bundles *what the federation is* — a controller, the participating
clients, and the transport/codec choices — separately from *how it is launched*.
The same Job runs in the simulator today (:meth:`simulate`) and, once a networked
transport is attached, across real sites unchanged; :meth:`to_dict` emits a
manifest of the run for reproducibility. This mirrors NVFlare's FedJob, kept
Python-first rather than mandating JSON config files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..apis.core import Controller
from ..codec import Codec
from ..server import Server
from ..simulator import Simulator
from ..transport import InProcessTransport, Transport


@dataclass
class Job:
    """A federated job: a controller, its clients, and launch settings."""

    name: str
    controller: Controller
    clients: List[Any]
    server: Optional[Server] = None
    transport: Optional[Transport] = None
    codec: Optional[Codec] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def simulate(self, threads: int = 1, fl_ctx=None) -> Any:
        """Run the job in the local simulator (optionally multi-threaded)."""
        transport = self.transport
        if transport is None and threads > 1:
            transport = InProcessTransport(threads=threads, codec=self.codec)
        sim = Simulator(
            self.controller,
            self.clients,
            server=self.server,
            transport=transport,
            codec=self.codec,
        )
        return sim.run(fl_ctx)

    def to_dict(self) -> Dict[str, Any]:
        """A best-effort manifest of the job, for logging/reproducibility."""
        ctrl = self.controller
        manifest: Dict[str, Any] = {
            "name": self.name,
            "controller": type(ctrl).__name__,
            "num_rounds": getattr(ctrl, "num_rounds", None),
            "aggregator": type(getattr(ctrl, "aggregator", None)).__name__
            if getattr(ctrl, "aggregator", None) is not None
            else None,
            "widgets": [type(w).__name__ for w in getattr(ctrl, "widgets", [])],
            "n_clients": len(self.clients),
            "clients": [getattr(c, "name", "?") for c in self.clients],
            "transport": type(self.transport).__name__ if self.transport else "InProcessTransport",
            "codec": type(self.codec).__name__ if self.codec else "default",
        }
        manifest.update(self.meta)
        return manifest
