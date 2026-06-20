# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The federation server.

The server owns the authoritative global model and offers the one primitive the
controller uses to reach clients: :meth:`broadcast`. It does not talk to clients
directly — it encodes the task with a :class:`~erdos.codec.base.Codec`, hands the
bytes to a :class:`~erdos.transport.base.Transport`, and decodes the replies.
Swapping the transport (in-process, threaded, or networked) therefore changes
nothing above this line: the same controller and executors run unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .apis.core import FLComponent
from .apis.shareable import FLContext, Shareable
from .codec import Codec, default_codec
from .transport import InProcessTransport, TaskReply, Transport


class Server(FLComponent):
    """Holds the global model and dispatches tasks to clients via a transport."""

    def __init__(
        self,
        transport: Optional[Transport] = None,
        codec: Optional[Codec] = None,
    ) -> None:
        super().__init__()
        self.codec = codec or default_codec()
        self.transport = transport
        self.global_params: Optional[Dict[str, Any]] = None
        #: Replies from the most recent broadcast that failed (name -> error).
        self.last_failures: Dict[str, str] = {}

    # -- client / transport wiring ----------------------------------------
    def register_clients(self, clients: List[Any]) -> None:
        """Attach client runtimes. Builds a default in-process transport if none."""
        if self.transport is None:
            self.transport = InProcessTransport(clients, codec=self.codec)
        elif hasattr(self.transport, "register"):
            self.transport.register(clients)

    @property
    def clients(self) -> List[str]:
        """Names of reachable clients (kept for backward compatibility)."""
        return self.transport.client_names() if self.transport is not None else []

    # -- the communication primitive --------------------------------------
    def broadcast(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        *,
        targets: Optional[List[str]] = None,
        min_responses: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[Tuple[str, Shareable]]:
        """Send ``task_name`` + ``shareable`` to clients; return decoded replies.

        The payload is serialized once and each client receives an independent,
        decoded copy (so local mutation cannot leak across sites). Clients that
        fail or time out are recorded in :attr:`last_failures` instead of
        aborting the round. If ``min_responses`` is given and fewer clients
        succeed, a :class:`RuntimeError` is raised.
        """
        if self.transport is None:
            raise RuntimeError("Server has no transport; call register_clients() first.")

        payload = self.codec.encode(shareable)
        headers = {"task": task_name, "round": fl_ctx.get_prop(FLContext.CURRENT_ROUND)}
        replies: List[TaskReply] = self.transport.broadcast(
            task_name, payload, headers, fl_ctx, targets=targets, timeout=timeout
        )

        results: List[Tuple[str, Shareable]] = []
        self.last_failures = {}
        for r in replies:
            if r.success:
                results.append((r.client_name, self.codec.decode(r.payload)))
            else:
                self.last_failures[r.client_name] = r.error or "unknown error"

        if min_responses is not None and len(results) < min_responses:
            raise RuntimeError(
                f"Only {len(results)} client(s) responded to '{task_name}', "
                f"need {min_responses}. Failures: {self.last_failures}"
            )
        return results
