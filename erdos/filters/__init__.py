# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Filters that transform a Shareable in flight (privacy, compression, ...).

These import PyTorch; import this subpackage only in environments that have
``torch`` installed.
"""
from .dp import GaussianPrivacyFilter

__all__ = ["GaussianPrivacyFilter"]
