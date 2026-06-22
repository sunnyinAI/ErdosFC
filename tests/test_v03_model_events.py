# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""v0.3 — FLModel payload, DIFF updates, widgets, and federated eval."""
import numpy as np
import pytest

from conftest import NumpyConstTrainer, NumpyDiffTrainer, numpy_clients

from erdos_fl import (
    BestModelSelector,
    Client,
    EarlyStopping,
    FedAvg,
    FedAvgAggregator,
    FLModel,
    ParamsType,
    Shareable,
    Simulator,
)


def test_flmodel_roundtrips_through_shareable():
    m = FLModel(
        params={"w": np.array([1.0, 2.0])},
        params_type=ParamsType.DIFF,
        metrics={"train_loss": 0.5},
        current_round=2,
        total_rounds=10,
    )
    back = FLModel.from_shareable(m.to_shareable())
    assert back.params_type == ParamsType.DIFF
    assert back.metrics["train_loss"] == 0.5
    assert back.current_round == 2
    assert np.array_equal(back.params["w"], np.array([1.0, 2.0]))
    assert FLModel.params_type_of(m.to_shareable()) == ParamsType.DIFF


def test_diff_updates_are_added_to_global():
    # Each client returns delta; FedAvg must ADD the averaged delta to the global.
    clients = [
        Client("a", NumpyDiffTrainer({"w": [2.0]}, num_samples=1)),
        Client("b", NumpyDiffTrainer({"w": [4.0]}, num_samples=1)),
    ]
    controller = FedAvg(
        num_rounds=2, initial_params={"w": np.array([0.0])}, aggregator=FedAvgAggregator()
    )
    result = Simulator(controller, clients).run()
    # avg delta = 3 per round -> after 2 rounds global = 6
    assert np.allclose(result["w"], np.array([6.0]))


def test_best_model_selector_widget():
    clients = numpy_clients([("s1", {"w": [1.0]}, 1), ("s2", {"w": [1.0]}, 1)])
    accs = iter([0.5, 0.9, 0.7])  # peak at round 2

    def evaluate(params, rnd):
        return {"test_acc": next(accs)}

    best = BestModelSelector(key_metric="test_acc", mode="max")
    controller = FedAvg(
        num_rounds=3,
        initial_params={"w": np.array([0.0])},
        aggregator=FedAvgAggregator(),
        evaluate_fn=evaluate,
        widgets=[best],
    )
    Simulator(controller, clients).run()
    assert best.best_round == 2
    assert best.best_metric == 0.9
    assert best.best_params is not None


def test_early_stopping_widget_halts_run():
    clients = numpy_clients([("s1", {"w": [1.0]}, 1)])
    accs = iter([0.5, 0.5, 0.5, 0.5, 0.5])  # never improves

    def evaluate(params, rnd):
        return {"test_acc": next(accs)}

    stopper = EarlyStopping(key_metric="test_acc", patience=2, mode="max")
    controller = FedAvg(
        num_rounds=5,
        initial_params={"w": np.array([0.0])},
        aggregator=FedAvgAggregator(),
        evaluate_fn=evaluate,
        widgets=[stopper],
    )
    Simulator(controller, clients).run()
    # First round sets the baseline; 2 more without improvement -> stop after round 3.
    assert len(controller.history) == 3


def test_federated_evaluate_task_is_wired():
    clients = [
        Client("s1", NumpyConstTrainer({"w": [1.0]}, num_samples=1, eval_accuracy=0.6)),
        Client("s2", NumpyConstTrainer({"w": [1.0]}, num_samples=3, eval_accuracy=0.9)),
    ]
    controller = FedAvg(
        num_rounds=1,
        initial_params={"w": np.array([0.0])},
        aggregator=FedAvgAggregator(),
        federated_eval=True,
    )
    Simulator(controller, clients).run()
    # weighted mean accuracy = (0.6*1 + 0.9*3) / 4 = 0.825
    assert controller.history[-1]["fed_accuracy"] == pytest.approx(0.825)
