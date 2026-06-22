# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Smoke + correctness tests for Erdos FC.

Run with:  pytest -q
(The training tests are skipped automatically if PyTorch is not installed.)
"""
import sys
from pathlib import Path

import pytest

# Make the repo importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset  # noqa: E402

from erdos_fl import (  # noqa: E402
    Client,
    FedAvg,
    FedAvgAggregator,
    FLContext,
    PTTrainer,
    Shareable,
    Simulator,
)


def test_aggregator_weighted_mean():
    """FedAvg must weight contributions by num_samples."""
    agg = FedAvgAggregator()
    ctx = FLContext()
    agg.reset(ctx)
    agg.accept(Shareable(params={"w": torch.tensor([0.0, 10.0])}, meta={"num_samples": 1}), ctx)
    agg.accept(Shareable(params={"w": torch.tensor([10.0, 10.0])}, meta={"num_samples": 3}), ctx)
    out = agg.aggregate(ctx).params["w"]
    # (1*[0,10] + 3*[10,10]) / 4 == [7.5, 10]
    assert torch.allclose(out, torch.tensor([7.5, 10.0]))


def test_aggregator_preserves_integer_dtype():
    """Integer buffers must come back out as integers, not floats."""
    agg = FedAvgAggregator()
    ctx = FLContext()
    agg.reset(ctx)
    agg.accept(Shareable(params={"b": torch.tensor([2, 4], dtype=torch.int64)}, meta={"num_samples": 1}), ctx)
    agg.accept(Shareable(params={"b": torch.tensor([4, 8], dtype=torch.int64)}, meta={"num_samples": 1}), ctx)
    out = agg.aggregate(ctx).params["b"]
    assert out.dtype == torch.int64
    assert out.tolist() == [3, 6]


class _LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _toy_dataset(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x.sum(dim=1) > 0).long()
    return TensorDataset(x, y)


def test_simulation_runs_and_records_history():
    """A full 2-round FedAvg simulation should run and return a valid model."""
    clients = [
        Client(
            f"site-{i}",
            PTTrainer(_LinearModel(), _toy_dataset(seed=i), epochs=1, batch_size=16, device="cpu"),
        )
        for i in range(3)
    ]
    init = {k: v.detach().cpu().clone() for k, v in _LinearModel().state_dict().items()}
    controller = FedAvg(num_rounds=2, initial_params=init, aggregator=FedAvgAggregator())

    result = Simulator(controller, clients).run()

    assert len(controller.history) == 2
    assert controller.history[-1]["clients"] == 3
    assert set(result.keys()) == set(init.keys())


def test_privacy_filter_changes_update():
    """The Gaussian privacy filter should perturb a client's weights."""
    from erdos_fl import GaussianPrivacyFilter

    flt = GaussianPrivacyFilter(sigma=0.1, seed=0)
    ctx = FLContext()
    original = torch.zeros(100)
    out = flt.process(Shareable(params={"w": original.clone()}, meta={}), ctx)
    assert not torch.allclose(out.params["w"], original)
    assert out.get_meta_prop("privacy_sigma") == pytest.approx(0.1)
