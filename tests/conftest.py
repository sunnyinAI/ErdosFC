# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared test fixtures/helpers for the Erdos FC suite.

Makes the repo importable without installation, and provides a tiny, torch-free
NumPy executor so the full federated round-trip (codec + transport + aggregator)
can be exercised in any environment.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from erdos_fl import Client, Executor, FLModel, ParamsType, Shareable, TaskName  # noqa: E402


class NumpyConstTrainer(Executor):
    """A trivial executor that returns fixed NumPy params and a sample count.

    Independent of the incoming global model, so aggregation results are exactly
    predictable in tests. Optionally answers the EVALUATE task with a fixed
    accuracy, so federated-evaluation wiring can be tested without torch.
    """

    def __init__(self, params, num_samples=1, eval_accuracy=None):
        super().__init__()
        self._params = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        self.num_samples = int(num_samples)
        self.eval_accuracy = eval_accuracy

    def execute(self, task_name, shareable, fl_ctx):
        if task_name == TaskName.EVALUATE:
            if self.eval_accuracy is None:
                raise ValueError("this trainer was not configured for evaluation")
            return Shareable(
                params={},
                meta={"num_samples": self.num_samples, "accuracy": float(self.eval_accuracy)},
            )
        if task_name != TaskName.TRAIN:
            raise ValueError(f"unsupported task {task_name!r}")
        return Shareable(
            params={k: v.copy() for k, v in self._params.items()},
            meta={"num_samples": self.num_samples},
        )


class NumpyDiffTrainer(Executor):
    """Returns a fixed weight *delta* as a DIFF payload (for FedOpt/diff tests)."""

    def __init__(self, delta, num_samples=1):
        super().__init__()
        self._delta = {k: np.asarray(v, dtype=np.float64) for k, v in delta.items()}
        self.num_samples = int(num_samples)

    def execute(self, task_name, shareable, fl_ctx):
        if task_name != TaskName.TRAIN:
            raise ValueError(f"unsupported task {task_name!r}")
        return FLModel(
            params={k: v.copy() for k, v in self._delta.items()},
            params_type=ParamsType.DIFF,
            meta={"num_samples": self.num_samples},
        ).to_shareable()


def numpy_clients(specs):
    """Build clients from ``[(name, params_dict, num_samples), ...]``."""
    return [
        Client(name, NumpyConstTrainer(params, ns)) for name, params, ns in specs
    ]
