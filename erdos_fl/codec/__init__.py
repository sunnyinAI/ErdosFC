# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Serialization codecs for Erdos FC payloads.

:func:`default_codec` returns the best codec available in the current install:
the compact :class:`MsgpackCodec` if ``msgpack`` is present, otherwise the
zero-dependency :class:`JsonCodec`.
"""
from __future__ import annotations

from .base import Codec, from_portable, to_portable
from .json_codec import JsonCodec

try:  # pragma: no cover - depends on optional dependency
    from .msgpack_codec import MsgpackCodec

    _HAS_MSGPACK = True
except ImportError:  # pragma: no cover
    MsgpackCodec = None  # type: ignore[assignment]
    _HAS_MSGPACK = False

# Registry by codec name, used to decode a payload with the codec that wrote it.
_REGISTRY = {JsonCodec.name: JsonCodec}
if _HAS_MSGPACK:
    _REGISTRY[MsgpackCodec.name] = MsgpackCodec


def default_codec() -> Codec:
    """Return the preferred codec for this install."""
    if _HAS_MSGPACK:
        return MsgpackCodec()
    return JsonCodec()


def get_codec(name: str) -> Codec:
    """Construct a codec by its :attr:`Codec.name` (e.g. ``"json"``)."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown or unavailable codec {name!r}. Available: {sorted(_REGISTRY)}"
        )


__all__ = [
    "Codec",
    "JsonCodec",
    "MsgpackCodec",
    "default_codec",
    "get_codec",
    "to_portable",
    "from_portable",
]
