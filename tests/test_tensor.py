# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the framework-agnostic tensor adapter (NumPy path)."""
import numpy as np

from erdos.apis import tensor


def test_numpy_roundtrip_preserves_dtype_and_values():
    for arr in (
        np.array([1.5, -2.0, 3.25], dtype=np.float32),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        np.arange(6, dtype=np.float64).reshape(2, 3),
    ):
        backend, dtype = tensor.restore_info(arr)
        assert backend == "numpy"
        buf = tensor.as_numpy(arr)
        rebuilt = tensor.from_numpy(buf, backend, dtype)
        assert rebuilt.dtype == arr.dtype
        assert np.array_equal(rebuilt, arr)


def test_clone_is_independent():
    a = np.array([1.0, 2.0, 3.0])
    b = tensor.clone(a)
    b[0] = 99.0
    assert a[0] == 1.0


def test_is_tensor_and_is_floating():
    assert tensor.is_tensor(np.zeros(3))
    assert not tensor.is_tensor([1, 2, 3])
    assert tensor.is_floating(np.zeros(3, dtype=np.float32))
    assert not tensor.is_floating(np.zeros(3, dtype=np.int32))


def test_from_numpy_yields_writable_array():
    a = np.array([1.0, 2.0])
    out = tensor.from_numpy(tensor.as_numpy(a), "numpy", "float64")
    out[0] = 5.0  # must not raise (frombuffer arrays are read-only)
    assert out[0] == 5.0
