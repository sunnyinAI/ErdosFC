# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Filters that transform a Shareable in flight (privacy, compression, ...).

:class:`ExcludeVars` is torch-free and always importable. The torch-based
filters (:class:`GaussianPrivacyFilter`, and the v0.6 quantization filter) are
imported lazily and are ``None`` when torch is absent.
"""
from .exclude import ExcludeVars

try:  # pragma: no cover - depends on optional dependency
    from .dp import GaussianPrivacyFilter

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    GaussianPrivacyFilter = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = ["ExcludeVars", "GaussianPrivacyFilter"]
