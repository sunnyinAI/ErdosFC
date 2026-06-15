# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Client-side executors (local training logic).

These import PyTorch; import this subpackage only where ``torch`` is available.
"""
from .pt_trainer import PTTrainer

__all__ = ["PTTrainer"]
