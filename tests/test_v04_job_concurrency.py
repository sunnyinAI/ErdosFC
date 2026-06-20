# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""v0.4 — NumpyTrainer, ExcludeVars, concurrency/timeout, Job, and CLI."""
import time
from textwrap import dedent

import numpy as np
import pytest

from conftest import NumpyConstTrainer, numpy_clients

from erdos import (
    Client,
    Executor,
    ExcludeVars,
    FedAvg,
    FedAvgAggregator,
    FLContext,
    InProcessTransport,
    Job,
    NumpyTrainer,
    Server,
    Shareable,
    Simulator,
    TaskName,
)


def _make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    x0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n0, 2))
    x1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n - n0, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n0 + [1] * (n - n0))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def _federated_numpy(params_transfer):
    x, y = _make_blobs(240, seed=1)
    shards = [(x[:120], y[:120]), (x[120:], y[120:])]
    clients = [
        Client(f"s{i}", NumpyTrainer(shard, epochs=5, lr=0.5, params_transfer=params_transfer))
        for i, shard in enumerate(shards)
    ]
    controller = FedAvg(
        num_rounds=12,
        initial_params=NumpyTrainer.init_params(2, 2),
        aggregator=FedAvgAggregator(),
        federated_eval=True,
    )
    Simulator(controller, clients).run()
    return controller.history[-1]["fed_accuracy"]


def test_numpy_trainer_full_learns():
    assert _federated_numpy("full") > 0.9


def test_numpy_trainer_diff_learns():
    assert _federated_numpy("diff") > 0.9


def test_exclude_vars_filter():
    flt = ExcludeVars(["*.running_*", "drop_me"])
    s = Shareable(
        params={
            "fc.weight": np.zeros(3),
            "bn.running_mean": np.zeros(3),
            "drop_me": np.zeros(1),
        },
        meta={},
    )
    out = flt.process(s, FLContext())
    assert set(out.params) == {"fc.weight"}
    assert set(out.get_meta_prop("excluded_vars")) == {"bn.running_mean", "drop_me"}


class _SlowTrainer(Executor):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def execute(self, task_name, shareable, fl_ctx):
        time.sleep(self.delay)
        return Shareable(params={"w": np.array([1.0])}, meta={"num_samples": 1})


def test_threaded_transport_runs_all_clients():
    transport = InProcessTransport(threads=4)
    server = Server(transport=transport)
    server.register_clients(numpy_clients([(f"s{i}", {"w": [float(i)]}, 1) for i in range(4)]))
    transport.start()
    try:
        results = server.broadcast(
            TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), FLContext()
        )
    finally:
        transport.stop()
    assert len(results) == 4


def test_timeout_drops_straggler():
    transport = InProcessTransport(threads=2)
    server = Server(transport=transport)
    server.register_clients(
        [Client("fast", NumpyConstTrainer({"w": [1.0]}, 1)), Client("slow", _SlowTrainer(0.5))]
    )
    transport.start()
    try:
        results = server.broadcast(
            TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), FLContext(), timeout=0.05
        )
    finally:
        transport.stop()
    assert [n for n, _ in results] == ["fast"]
    assert "slow" in server.last_failures


def test_job_simulate_and_manifest():
    clients = numpy_clients([("s1", {"w": [2.0]}, 1), ("s2", {"w": [4.0]}, 1)])
    controller = FedAvg(
        num_rounds=2, initial_params={"w": np.array([0.0])}, aggregator=FedAvgAggregator()
    )
    job = Job(name="demo", controller=controller, clients=clients)
    result = job.simulate(threads=2)
    assert np.allclose(result["w"], np.array([3.0]))
    manifest = job.to_dict()
    assert manifest["name"] == "demo"
    assert manifest["controller"] == "FedAvg"
    assert manifest["n_clients"] == 2


def test_cli_simulate(tmp_path, capsys):
    from erdos.cli import main

    job_file = tmp_path / "myjob.py"
    job_file.write_text(
        dedent(
            """
            import numpy as np
            from erdos import Client, FedAvg, FedAvgAggregator, Job, NumpyTrainer

            def build_job():
                x = np.random.default_rng(0).normal(size=(40, 2))
                y = (x.sum(1) > 0).astype(int)
                clients = [Client(f"s{i}", NumpyTrainer((x, y), epochs=2)) for i in range(2)]
                ctrl = FedAvg(num_rounds=3, initial_params=NumpyTrainer.init_params(2, 2),
                              aggregator=FedAvgAggregator())
                return Job(name="cli-demo", controller=ctrl, clients=clients)
            """
        )
    )
    assert main(["simulate", str(job_file)]) == 0
    assert main(["version"]) == 0
    assert "cli-demo" in capsys.readouterr().out
