# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Widgets: cross-cutting concerns that plug into the event bus.

A widget is an :class:`~erdos.apis.core.FLComponent` that reacts to lifecycle
events instead of being called by the controller. Best-model selection, early
stopping, persistence, and (v0.4) experiment tracking are all widgets, so they
can be added to a run without touching ``control_flow``.
"""
from __future__ import annotations

from .best_model import BestModelSelector
from .early_stopping import EarlyStopping
from .persistor import ModelPersistor

__all__ = ["BestModelSelector", "EarlyStopping", "ModelPersistor"]
