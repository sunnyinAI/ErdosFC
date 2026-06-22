# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Erdos Federated Computing (EFC).

A lightweight, extensible runtime for federated learning — many collaborators,
one model, no shared data. Named for Paul Erdos, whose mathematics was built
entirely on collaboration.

Typical usage::

    import erdos_fl

    clients = [erdos_fl.Client(f"site-{i}", trainer_i) for i in range(num_sites)]
    controller = erdos_fl.FedAvg(num_rounds=5, initial_params=w0,
                              aggregator=erdos_fl.FedAvgAggregator())
    erdos_fl.Simulator(controller, clients).run()
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Sunny Gupta"

# Framework-agnostic core.
from .apis.core import (
    Aggregator,
    Controller,
    Executor,
    Filter,
    FLComponent,
    TaskName,
)
from .apis.model import FLModel, ParamsType
from .apis.shareable import FLContext, Shareable
from .aggregators.fedavg import FedAvgAggregator
from .client import Client
from .codec import Codec, JsonCodec, MsgpackCodec, default_codec
from .engine import EventType, FLContextKey, RunEngine
from .executors.numpy_trainer import NumpyTrainer
from .filters.exclude import ExcludeVars
from .job import Job
from .server import Server
from .simulator import Simulator
from .transport import InProcessTransport, TaskReply, Transport
from .widgets import BestModelSelector, EarlyStopping, ModelPersistor
from .workflows.base import BaseModelController
from .workflows.fedavg import FedAvg

# PyTorch-dependent components are optional so that `import erdos_fl` works even
# in a minimal environment without torch installed.
try:  # pragma: no cover - depends on optional dependency
    from .executors.pt_trainer import PTTrainer
    from .filters.dp import GaussianPrivacyFilter

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    PTTrainer = None  # type: ignore[assignment]
    GaussianPrivacyFilter = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = [
    "__version__",
    # core
    "Shareable",
    "FLContext",
    "FLComponent",
    "Executor",
    "Aggregator",
    "Filter",
    "Controller",
    "TaskName",
    "FLModel",
    "ParamsType",
    # runtime
    "Server",
    "Client",
    "Simulator",
    "Job",
    # executors / filters (torch-free)
    "NumpyTrainer",
    "ExcludeVars",
    # transport + serialization
    "Transport",
    "InProcessTransport",
    "TaskReply",
    "Codec",
    "JsonCodec",
    "MsgpackCodec",
    "default_codec",
    # event bus
    "RunEngine",
    "EventType",
    "FLContextKey",
    # algorithms
    "FedAvg",
    "BaseModelController",
    "FedAvgAggregator",
    # widgets
    "BestModelSelector",
    "EarlyStopping",
    "ModelPersistor",
    # torch-dependent (may be None)
    "PTTrainer",
    "GaussianPrivacyFilter",
]
