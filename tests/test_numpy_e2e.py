# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""End-to-end FedAvg on the torch-free NumPy path.

Proves the framework-agnostic claim: a full federation (controller + codec +
transport + aggregator) runs with no deep-learning library installed.
"""
import numpy as np

from conftest import numpy_clients

from erdos import FedAvg, FedAvgAggregator, Simulator


def test_numpy_fedavg_weighted_average():
    # Two sites return fixed weights; FedAvg should produce the sample-weighted mean.
    clients = numpy_clients(
        [("site-1", {"w": [0.0, 10.0]}, 1), ("site-2", {"w": [10.0, 10.0]}, 3)]
    )
    init = {"w": np.array([1.0, 1.0])}
    controller = FedAvg(num_rounds=3, initial_params=init, aggregator=FedAvgAggregator())

    result = Simulator(controller, clients).run()

    # (1*[0,10] + 3*[10,10]) / 4 == [7.5, 10]
    assert np.allclose(result["w"], np.array([7.5, 10.0]))
    assert len(controller.history) == 3
    assert controller.history[-1]["clients"] == 2


def test_history_records_evaluate_fn_metrics():
    clients = numpy_clients([("s1", {"w": [2.0]}, 1), ("s2", {"w": [4.0]}, 1)])
    seen = {}

    def evaluate(params, rnd):
        seen["last"] = float(params["w"][0])
        return {"mean_w": float(params["w"][0])}

    controller = FedAvg(
        num_rounds=2,
        initial_params={"w": np.array([0.0])},
        aggregator=FedAvgAggregator(),
        evaluate_fn=evaluate,
    )
    Simulator(controller, clients).run()
    assert controller.history[-1]["mean_w"] == 3.0  # mean of 2 and 4
