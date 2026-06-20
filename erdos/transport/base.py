# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The transport seam.

A :class:`Transport` is the one component that knows *how* a task reaches a
client. The server and controller depend only on this interface, never on a
concrete client object, so the same :class:`~erdos.workflows.fedavg.FedAvg`
controller runs unchanged whether tasks travel in-process
(:class:`~erdos.transport.inprocess.InProcessTransport`), across threads, or
over a network. Payloads cross this boundary as opaque ``bytes`` produced by a
:class:`~erdos.codec.base.Codec`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..apis.core import FLComponent
from ..apis.shareable import FLContext


@dataclass
class TaskReply:
    """One client's response to a broadcast task."""

    client_name: str
    payload: Optional[bytes] = None  # encoded result Shareable, or None on failure
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.payload is not None


class Transport(FLComponent, ABC):
    """Moves encoded tasks from the server to clients and collects replies."""

    @abstractmethod
    def client_names(self) -> List[str]:
        """Names of the clients currently reachable through this transport."""
        raise NotImplementedError

    @abstractmethod
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
        """Send ``payload`` for ``task_name`` to ``targets`` (default: all).

        Implementations must return one :class:`TaskReply` per *attempted*
        client. A client that errors or times out yields a reply with ``error``
        set rather than aborting the whole broadcast; the caller decides whether
        enough clients succeeded.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Bring the transport up (no-op for in-process)."""

    def stop(self) -> None:
        """Tear the transport down (no-op for in-process)."""

    def __enter__(self) -> "Transport":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
