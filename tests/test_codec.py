# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the serialization codecs (JSON always; msgpack if installed)."""
import numpy as np
import pytest

from erdos_fl import JsonCodec, Shareable
from erdos_fl.codec import MsgpackCodec

CODECS = [JsonCodec]
if MsgpackCodec is not None:
    CODECS.append(MsgpackCodec)


def _sample_shareable():
    return Shareable(
        params={
            "w": np.array([[1.5, -2.0], [3.25, 4.0]], dtype=np.float32),
            "bias": np.array([0.1, 0.2], dtype=np.float64),
            "bn_count": np.array([7], dtype=np.int64),
        },
        meta={
            "num_samples": 1280,
            "train_loss": 0.4213,
            "client": "site-1",
            "flags": [True, False, None],
            "nested": {"round": 3, "tags": ["a", "b"]},
        },
    )


@pytest.mark.parametrize("codec_cls", CODECS)
def test_roundtrip_preserves_params_and_meta(codec_cls):
    codec = codec_cls()
    original = _sample_shareable()
    decoded = codec.decode(codec.encode(original))

    assert set(decoded.params) == set(original.params)
    for k, v in original.params.items():
        assert decoded.params[k].dtype == v.dtype
        assert np.array_equal(decoded.params[k], v)
    assert decoded.meta == original.meta


@pytest.mark.parametrize("codec_cls", CODECS)
def test_clone_is_independent(codec_cls):
    codec = codec_cls()
    original = _sample_shareable()
    clone = codec.clone(original)
    clone.params["w"][0, 0] = 999.0
    clone.meta["num_samples"] = 1
    assert original.params["w"][0, 0] == pytest.approx(1.5)
    assert original.meta["num_samples"] == 1280


@pytest.mark.skipif(MsgpackCodec is None, reason="msgpack not installed")
def test_codecs_agree():
    j, m = JsonCodec(), MsgpackCodec()
    original = _sample_shareable()
    dj = j.decode(j.encode(original))
    dm = m.decode(m.encode(original))
    for k in original.params:
        assert np.array_equal(dj.params[k], dm.params[k])
    assert dj.meta == dm.meta


@pytest.mark.parametrize("codec_cls", CODECS)
def test_encode_produces_bytes(codec_cls):
    assert isinstance(codec_cls().encode(_sample_shareable()), (bytes, bytearray))
