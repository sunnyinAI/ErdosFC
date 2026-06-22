# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Optional MessagePack codec (``pip install erdos-fc[fast]``).

Same portable form as :class:`~erdos.codec.json_codec.JsonCodec`, but tensor
bytes are carried natively (no base64) inside a MessagePack envelope, so the
encoding is more compact and faster to parse for large models. This mirrors the
role MessagePack plays in NVFlare's FOBS. Importing this module raises
``ImportError`` if ``msgpack`` is not installed.
"""
from __future__ import annotations

import msgpack  # noqa: F401  (import error surfaces the missing optional dep)

from ..apis.shareable import Shareable
from .base import Codec, to_portable


class MsgpackCodec(Codec):
    """MessagePack serialization with native binary tensor payloads."""

    name = "msgpack"

    def encode(self, shareable: Shareable) -> bytes:
        portable = to_portable(dict(shareable), binary=True)
        return msgpack.packb(portable, use_bin_type=True)

    def decode(self, data: bytes) -> Shareable:
        payload = msgpack.unpackb(data, raw=False)
        return self._rebuild(payload)
