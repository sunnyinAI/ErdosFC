# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""In-process transport — the default for the simulator.

Clients live in the same process as the server, but tasks still cross the real
serialization boundary: each payload is *decoded* before the client sees it and
the result is *encoded* before it goes back, exactly as a networked transport
would. Each client task also runs in its own child :class:`FLContext` scope, so
no per-client state leaks through the shared run context.

With ``threads=1`` (the default) clients are visited sequentially. With
``threads>1`` they run on a thread pool — safe precisely because the context is
scoped per task and each payload is an independent decoded copy — and a
per-broadcast ``timeout`` lets the round proceed without stragglers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, List, Optional

from ..apis.shareable import FLContext
from ..codec import Codec, default_codec
from .base import TaskReply, Transport


class InProcessTransport(Transport):
    """Runs client tasks in-process, through the codec, optionally concurrently."""

    def __init__(
        self,
        clients: Optional[list] = None,
        codec: Optional[Codec] = None,
        threads: int = 1,
    ) -> None:
        super().__init__()
        self.codec = codec or default_codec()
        self.threads = max(1, int(threads))
        self._clients: Dict[str, object] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        if clients:
            self.register(clients)

    def register(self, clients: list) -> None:
        """Register client runtimes (objects exposing ``name`` and ``run_task``)."""
        for c in clients:
            self._clients[c.name] = c

    def client_names(self) -> List[str]:
        return list(self._clients)

    def start(self) -> None:
        if self.threads > 1 and self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.threads)

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _run_one(
        self, name: str, task_name: str, payload: bytes, headers: dict, fl_ctx: FLContext
    ) -> TaskReply:
        client = self._clients.get(name)
        if client is None:
            return TaskReply(name, error=f"unknown client {name!r}")
        try:
            # Each client gets an isolated child scope (the "client side").
            child = fl_ctx.new_child()
            child.set_prop(FLContext.CURRENT_CLIENT, name)
            if "round" in headers:
                child.set_prop(FLContext.CURRENT_ROUND, headers["round"])
            shareable = self.codec.decode(payload)
            result = client.run_task(task_name, shareable, child)
            return TaskReply(name, payload=self.codec.encode(result))
        except Exception as exc:  # noqa: BLE001 - one client must not abort the round
            self.logger.warning("Client %s failed on task %s: %s", name, task_name, exc)
            return TaskReply(name, error=f"{type(exc).__name__}: {exc}")

    def broadcast(
        self,
        task_name: str,
        payload: bytes,
        headers: dict,
        fl_ctx: FLContext,
        *,
        targets: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[TaskReply]:
        names = targets if targets is not None else self.client_names()
        if self.threads == 1:
            return [self._run_one(n, task_name, payload, headers, fl_ctx) for n in names]

        if self._pool is None:  # broadcast() called without start(); be forgiving
            self.start()
        futures = {
            n: self._pool.submit(self._run_one, n, task_name, payload, headers, fl_ctx)
            for n in names
        }
        replies: List[TaskReply] = []
        for n, fut in futures.items():
            try:
                replies.append(fut.result(timeout=timeout))
            except FuturesTimeout:
                fut.cancel()
                replies.append(TaskReply(n, error=f"timeout after {timeout}s"))
        return replies
