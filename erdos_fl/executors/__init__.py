# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Client-side executors (local training logic).

:class:`NumpyTrainer` is torch-free and always importable. :class:`PTTrainer`
needs PyTorch, so it is imported lazily and is ``None`` when torch is absent —
importing this subpackage never fails in a minimal environment.
"""
from .numpy_trainer import NumpyTrainer

try:  # pragma: no cover - depends on optional dependency
    from .pt_trainer import PTTrainer

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    PTTrainer = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = ["NumpyTrainer", "PTTrainer"]
