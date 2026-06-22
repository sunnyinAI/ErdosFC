# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Base component classes for the Erdos FC programming model.

The design mirrors the controller/worker split used by mature federated
learning runtimes: a server-side :class:`Controller` defines *what* should
happen each round, while client-side :class:`Executor` objects define *how*
local work is performed. :class:`Aggregator` and :class:`Filter` are the two
extension points that sit on the server and on the message path respectively.

All components are deliberately framework-agnostic — none of these base
classes import a deep-learning library. Concrete PyTorch components live under
``erdos.executors`` and ``erdos.filters``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .shareable import FLContext, Shareable

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints
    from ..server import Server


class TaskName:
    """Canonical task names exchanged between server and clients."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    SUBMIT_MODEL = "submit_model"


class FLComponent:
    """Base class for every Erdos FC component.

    Provides a named logger and participates in the run's event bus: override
    :meth:`handle_event` to react to lifecycle events, or call :meth:`fire_event`
    to publish one. Both are no-ops unless a :class:`~erdos.engine.RunEngine` is
    attached to the context, so components work standalone too.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_event(self, event_type: str, fl_ctx: "FLContext") -> None:
        """React to a lifecycle event. Default: do nothing."""

    def fire_event(self, event_type: str, fl_ctx: "FLContext") -> None:
        """Fire a lifecycle event through the engine, if one is attached."""
        engine = fl_ctx.get_engine()
        if engine is not None:
            engine.fire_event(event_type, fl_ctx)


class Executor(FLComponent, ABC):
    """Runs on a client and performs the actual local computation for a task."""

    @abstractmethod
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Process ``shareable`` for ``task_name`` and return a result Shareable."""
        raise NotImplementedError


class Aggregator(FLComponent, ABC):
    """Combines client results (server-side) into a single global update."""

    @abstractmethod
    def reset(self, fl_ctx: FLContext) -> None:
        """Clear any state accumulated from a previous round."""
        raise NotImplementedError

    @abstractmethod
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accumulate one client contribution. Return ``True`` if accepted."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Produce the aggregated Shareable from everything accepted so far."""
        raise NotImplementedError


class Filter(FLComponent, ABC):
    """Transforms a Shareable in flight (e.g. privacy, compression, encryption)."""

    @abstractmethod
    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Return a (possibly modified) Shareable."""
        raise NotImplementedError


class Controller(FLComponent, ABC):
    """Server-side workflow: orchestrates rounds over the federation."""

    @abstractmethod
    def control_flow(self, server: "Server", fl_ctx: FLContext):
        """Drive the federated workflow to completion using ``server``."""
        raise NotImplementedError
