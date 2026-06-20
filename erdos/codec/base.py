# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The serialization boundary.

A :class:`Codec` turns a :class:`~erdos.apis.shareable.Shareable` into ``bytes``
and back. Every payload that crosses the :class:`~erdos.transport.base.Transport`
goes through a codec — even in the in-process simulator — so the code path you
debug locally is the same one a networked transport would use.

This is Erdos FC's analogue of NVFlare's FOBS: instead of ``pickle`` (which can
execute arbitrary code on load), payloads are reduced to a small, explicit
*portable form* made only of JSON-native scalars, lists, dicts, and a tagged
tensor record ``{__TENSOR__: {...}}``. Decoding rebuilds tensors via
:mod:`erdos.apis.tensor`; nothing in a payload can name a class to instantiate,
so a malicious payload cannot run code.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

from ..apis import tensor
from ..apis.shareable import Shareable

# Reserved key marking a serialized tensor inside the portable form.
TENSOR_TAG = "__erdos_tensor__"


def to_portable(obj: Any, *, binary: bool) -> Any:
    """Reduce ``obj`` to JSON-/msgpack-native data, tagging any tensors.

    With ``binary=True`` raw tensor bytes are kept as ``bytes`` (msgpack can
    carry them natively); otherwise they are base64-encoded into a ``str`` so
    the result is valid JSON.
    """
    if tensor.is_tensor(obj):
        arr = tensor.as_numpy(obj)
        backend, dtype_token = tensor.restore_info(obj)
        raw = arr.tobytes()
        return {
            TENSOR_TAG: {
                "b": backend,
                "d": dtype_token,
                "n": str(arr.dtype),  # buffer dtype
                "s": list(arr.shape),
                "x": raw if binary else base64.b64encode(raw).decode("ascii"),
            }
        }
    if isinstance(obj, dict):
        return {k: to_portable(v, binary=binary) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_portable(v, binary=binary) for v in obj]
    # Assume a transport-native scalar (str/int/float/bool/None/bytes).
    return obj


def from_portable(obj: Any) -> Any:
    """Inverse of :func:`to_portable`: rebuild tensors, leave scalars as-is."""
    import numpy as np

    if isinstance(obj, dict):
        rec = obj.get(TENSOR_TAG)
        if rec is not None:
            raw = rec["x"]
            if isinstance(raw, str):  # base64 (JSON path)
                raw = base64.b64decode(raw)
            arr = np.frombuffer(raw, dtype=np.dtype(rec["n"]))
            if rec["s"]:
                arr = arr.reshape(rec["s"])
            return tensor.from_numpy(arr, rec["b"], rec["d"])
        return {k: from_portable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_portable(v) for v in obj]
    return obj


class Codec(ABC):
    """Encodes/decodes a :class:`Shareable` to and from ``bytes``."""

    #: Short, stable identifier carried in transport headers so the receiver
    #: can pick the matching decoder.
    name: str = "codec"

    @abstractmethod
    def encode(self, shareable: Shareable) -> bytes:
        """Serialize ``shareable`` to bytes."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: bytes) -> Shareable:
        """Deserialize bytes back into a :class:`Shareable`."""
        raise NotImplementedError

    def clone(self, shareable: Shareable) -> Shareable:
        """Deep-copy a Shareable via an encode/decode round-trip.

        This is the framework-neutral replacement for ``tensor.clone`` of every
        param: it guarantees the copy is fully independent *and* exercises the
        real wire format.
        """
        return self.decode(self.encode(shareable))

    @staticmethod
    def _rebuild(payload: Any) -> Shareable:
        """Turn a decoded ``{"params":..., "meta":...}`` mapping into a Shareable."""
        data = from_portable(payload)
        s = Shareable()
        s[Shareable.PARAMS] = data.get(Shareable.PARAMS, {})
        s[Shareable.META] = data.get(Shareable.META, {})
        # Preserve any extra top-level sections (e.g. DXO fields added later).
        for k, v in data.items():
            if k not in (Shareable.PARAMS, Shareable.META):
                s[k] = v
        return s
