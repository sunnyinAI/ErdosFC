# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tests for the in-process transport and the server broadcast contract."""
import numpy as np
import pytest

from conftest import NumpyConstTrainer, numpy_clients

from erdos_fl import Client, Executor, FLContext, Server, Shareable, TaskName


class _Boom(Executor):
    def execute(self, task_name, shareable, fl_ctx):
        raise RuntimeError("kaboom")


def test_broadcast_returns_decoded_results():
    server = Server()
    server.register_clients(
        numpy_clients([("s1", {"w": [1.0, 2.0]}, 1), ("s2", {"w": [3.0, 4.0]}, 1)])
    )
    ctx = FLContext()
    task = Shareable(params={"w": np.array([0.0, 0.0])}, meta={})
    results = server.broadcast(TaskName.TRAIN, task, ctx)
    assert {n for n, _ in results} == {"s1", "s2"}
    by_name = dict(results)
    assert np.array_equal(by_name["s1"].params["w"], np.array([1.0, 2.0]))
    # Every result is tagged with its originating client.
    assert by_name["s1"].get_meta_prop("client") == "s1"


def test_failed_client_does_not_abort_round():
    clients = numpy_clients([("ok", {"w": [1.0]}, 1)]) + [Client("bad", _Boom())]
    server = Server()
    server.register_clients(clients)
    ctx = FLContext()
    results = server.broadcast(TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), ctx)
    assert [n for n, _ in results] == ["ok"]
    assert "bad" in server.last_failures
    assert "kaboom" in server.last_failures["bad"]


def test_min_responses_enforced():
    clients = numpy_clients([("ok", {"w": [1.0]}, 1)]) + [Client("bad", _Boom())]
    server = Server()
    server.register_clients(clients)
    with pytest.raises(RuntimeError, match="need 2"):
        server.broadcast(
            TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), FLContext(), min_responses=2
        )


def test_per_client_context_isolation():
    """CURRENT_CLIENT must be set per-task and never leak into the run scope."""
    server = Server()
    server.register_clients(numpy_clients([("s1", {"w": [1.0]}, 1), ("s2", {"w": [2.0]}, 1)]))
    run_ctx = FLContext()
    server.broadcast(TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), run_ctx)
    assert run_ctx.get_prop(FLContext.CURRENT_CLIENT) is None


def test_targets_subset():
    server = Server()
    server.register_clients(
        numpy_clients([("s1", {"w": [1.0]}, 1), ("s2", {"w": [2.0]}, 1), ("s3", {"w": [3.0]}, 1)])
    )
    results = server.broadcast(
        TaskName.TRAIN, Shareable(params={"w": np.array([0.0])}), FLContext(), targets=["s2"]
    )
    assert [n for n, _ in results] == ["s2"]
