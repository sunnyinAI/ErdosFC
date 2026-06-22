# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Transports: how tasks reach clients.

Always available: :class:`InProcessTransport` (the simulator default) and, from
v0.4, a threaded variant. A real networked :class:`TcpTransport` and an optional
gRPC transport are added in v0.6 behind the same :class:`Transport` contract.
"""
from __future__ import annotations

from .base import TaskReply, Transport
from .inprocess import InProcessTransport

__all__ = ["Transport", "TaskReply", "InProcessTransport"]
