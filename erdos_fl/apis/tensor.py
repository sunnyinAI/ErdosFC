# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Framework-agnostic tensor operations.

The Erdos FC core (aggregators, codec, controller, runtime) must not depend on
any one deep-learning library. This module is the single place where concrete
array types are touched: it knows how to move a tensor to/from a NumPy buffer
and how to rebuild it in its original framework and dtype. Everything else in
the core operates through these helpers, so a PyTorch tensor and a NumPy array
flow through the same code path.

NumPy is a hard dependency of Erdos FC and is always importable here; PyTorch is
optional and imported lazily, so this module works in a torch-free environment.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

_torch = None  # cached torch module, or False if unavailable


def _torch_mod():
    global _torch
    if _torch is None:
        try:  # pragma: no cover - depends on optional dependency
            import torch

            _torch = torch
        except ImportError:  # pragma: no cover
            _torch = False
    return _torch or None


def is_tensor(x: Any) -> bool:
    """True if ``x`` is a NumPy array or a PyTorch tensor."""
    if isinstance(x, np.ndarray):
        return True
    t = _torch_mod()
    return bool(t is not None and t.is_tensor(x))


def restore_info(x: Any) -> Tuple[str, str]:
    """Return ``(backend, dtype_token)`` needed to rebuild ``x`` after transport.

    ``backend`` is ``"numpy"`` or ``"torch"``; ``dtype_token`` is a plain string
    such as ``"float32"`` or ``"int64"`` (the ``torch.`` prefix is stripped).
    """
    if isinstance(x, np.ndarray):
        return "numpy", str(x.dtype)
    t = _torch_mod()
    if t is not None and t.is_tensor(x):
        return "torch", str(x.dtype).replace("torch.", "")
    raise TypeError(f"Not a supported tensor type: {type(x)!r}")


def as_numpy(x: Any) -> np.ndarray:
    """Return a contiguous NumPy view/copy of ``x`` suitable for serialization.

    ``bfloat16`` torch tensors (which NumPy cannot represent) are promoted to
    ``float32`` for transport; :func:`from_numpy` restores the original dtype.
    """
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    t = _torch_mod()
    if t is not None and t.is_tensor(x):
        x = x.detach().cpu()
        if x.dtype == t.bfloat16:
            x = x.to(t.float32)
        return np.ascontiguousarray(x.numpy())
    raise TypeError(f"Not a supported tensor type: {type(x)!r}")


def from_numpy(arr: np.ndarray, backend: str, dtype_token: str) -> Any:
    """Rebuild a tensor of ``backend``/``dtype_token`` from a NumPy buffer."""
    if backend == "numpy":
        # np.array makes a writable, contiguous copy (frombuffer is read-only).
        return np.array(arr, dtype=np.dtype(dtype_token))
    if backend == "torch":
        t = _torch_mod()
        if t is None:  # pragma: no cover - needs torch to decode a torch payload
            raise RuntimeError(
                "Payload was produced by PyTorch but torch is not installed here."
            )
        target = getattr(t, dtype_token)
        if target == t.bfloat16:
            base = t.from_numpy(np.array(arr, dtype="float32"))
            return base.to(t.bfloat16)
        return t.from_numpy(np.array(arr)).to(target)
    raise ValueError(f"Unknown tensor backend: {backend!r}")


def clone(x: Any) -> Any:
    """Deep-copy a tensor, preserving its framework and dtype."""
    if isinstance(x, np.ndarray):
        return x.copy()
    t = _torch_mod()
    if t is not None and t.is_tensor(x):
        return x.detach().clone()
    raise TypeError(f"Not a supported tensor type: {type(x)!r}")


def clone_params(params: dict) -> dict:
    """Deep-copy a ``{name: tensor}`` parameter mapping."""
    return {k: clone(v) for k, v in params.items()}


def is_floating(x: Any) -> bool:
    """True if ``x`` is a floating-point tensor."""
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    t = _torch_mod()
    if t is not None and t.is_tensor(x):
        return x.is_floating_point()
    return False


def _binary(a: Any, b: Any, op) -> Any:
    """Apply ``op`` in float64 NumPy space, restore ``a``'s framework/dtype."""
    backend, dtype = restore_info(a)
    out = op(as_numpy(a).astype(np.float64), as_numpy(b).astype(np.float64))
    return from_numpy(out, backend, dtype)


def add(a: Any, b: Any) -> Any:
    """Element-wise ``a + b``, result typed like ``a``."""
    return _binary(a, b, lambda x, y: x + y)


def subtract(a: Any, b: Any) -> Any:
    """Element-wise ``a - b``, result typed like ``a``."""
    return _binary(a, b, lambda x, y: x - y)


def add_params(base: dict, delta: dict) -> dict:
    """Return ``{k: base[k] + delta[k]}`` for every key in ``base``."""
    return {k: add(v, delta[k]) for k, v in base.items()}


def subtract_params(a: dict, b: dict) -> dict:
    """Return ``{k: a[k] - b[k]}`` for every key in ``a`` (e.g. local − global)."""
    return {k: subtract(v, b[k]) for k, v in a.items()}
