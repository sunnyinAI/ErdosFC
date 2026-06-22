# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pure-standard-library codec (the zero-dependency default).

Encodes a Shareable as UTF-8 JSON, with tensors reduced to a tagged record
whose raw bytes are base64-encoded. Needs nothing beyond the standard library
and NumPy, so it works in any Erdos FC install. For large models the optional
:class:`~erdos.codec.msgpack_codec.MsgpackCodec` (``pip install erdos-fc[fast]``)
is smaller and faster; both produce interchangeable semantics.
"""
from __future__ import annotations

import json

from ..apis.shareable import Shareable
from .base import Codec, to_portable


class JsonCodec(Codec):
    """JSON + base64 serialization. No third-party dependencies."""

    name = "json"

    def encode(self, shareable: Shareable) -> bytes:
        portable = to_portable(dict(shareable), binary=False)
        return json.dumps(portable, separators=(",", ":")).encode("utf-8")

    def decode(self, data: bytes) -> Shareable:
        payload = json.loads(data.decode("utf-8"))
        return self._rebuild(payload)
