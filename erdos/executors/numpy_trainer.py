# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A torch-free training executor (multinomial logistic regression).

``NumpyTrainer`` trains a softmax classifier with mini-batch SGD using only
NumPy. It exists to prove — and keep proving in CI — that Erdos FC's core is
genuinely framework-agnostic: a full federation (codec, transport, aggregator,
controller, FULL/DIFF updates, FedProx, evaluation) runs with no deep-learning
library installed. Its model is ``{"W": (n_features, n_classes), "b": (n_classes,)}``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..apis import tensor
from ..apis.core import Executor, TaskName
from ..apis.model import FLModel, ParamsType
from ..apis.shareable import FLContext, Shareable

Dataset = Tuple[np.ndarray, np.ndarray]  # (X, y)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyTrainer(Executor):
    """Local softmax-regression SGD trainer for one site, using only NumPy.

    Args:
        dataset: ``(X, y)`` with ``X`` shape ``(N, n_features)`` and integer
            labels ``y`` shape ``(N,)``.
        epochs: local epochs per round.
        lr: learning rate.
        batch_size: mini-batch size.
        params_transfer: ``"full"`` or ``"diff"`` (delta from the global model).
        fedprox_mu: if > 0, add the FedProx proximal term to the local gradient.
        eval_dataset: optional held-out data for the EVALUATE task.
        seed: RNG seed for shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        epochs: int = 1,
        lr: float = 0.1,
        batch_size: int = 32,
        params_transfer: str = "full",
        fedprox_mu: float = 0.0,
        eval_dataset: Optional[Dataset] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if params_transfer not in ("full", "diff"):
            raise ValueError("params_transfer must be 'full' or 'diff'")
        self.X = np.asarray(dataset[0], dtype=np.float64)
        self.y = np.asarray(dataset[1], dtype=np.int64)
        self.eval_dataset = eval_dataset
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.params_transfer = params_transfer
        self.fedprox_mu = float(fedprox_mu)
        self._rng = np.random.default_rng(seed)
        self._c_i: Optional[Dict[str, np.ndarray]] = None  # SCAFFOLD client control variate

    @staticmethod
    def init_params(n_features: int, n_classes: int) -> Dict[str, np.ndarray]:
        """Zero-initialized global model, convenient for ``initial_params``."""
        return {
            "W": np.zeros((n_features, n_classes), dtype=np.float64),
            "b": np.zeros((n_classes,), dtype=np.float64),
        }

    # -- task entry point --------------------------------------------------
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        if task_name == TaskName.TRAIN:
            return self._train(shareable)
        if task_name == TaskName.EVALUATE:
            return self._evaluate(shareable)
        if task_name == TaskName.SUBMIT_MODEL:
            return FLModel(params=self._current, params_type=ParamsType.FULL,
                           meta={"num_samples": len(self.y)}).to_shareable()
        raise ValueError(f"NumpyTrainer received unsupported task '{task_name}'")

    # -- training ----------------------------------------------------------
    def _train(self, shareable: Shareable) -> Shareable:
        if shareable.get("controls"):
            return self._train_scaffold(shareable)
        W = np.array(shareable.params["W"], dtype=np.float64)
        b = np.array(shareable.params["b"], dtype=np.float64)
        W0, b0 = W.copy(), b.copy()  # global snapshot (for diff + FedProx)
        n = len(self.y)

        total_loss, steps = 0.0, 0
        for _epoch in range(self.epochs):
            order = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = self.X[idx], self.y[idx]
                probs = _softmax(xb @ W + b)
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0
                total_loss += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean())
                steps += 1
                grad = probs - onehot
                dW = xb.T @ grad / len(yb)
                db = grad.mean(axis=0)
                if self.fedprox_mu > 0:
                    dW += self.fedprox_mu * (W - W0)
                    db += self.fedprox_mu * (b - b0)
                W -= self.lr * dW
                b -= self.lr * db

        self._current = {"W": W, "b": b}
        local = {"W": W, "b": b}
        if self.params_transfer == "diff":
            update = tensor.subtract_params(local, {"W": W0, "b": b0})
            params_type = ParamsType.DIFF
        else:
            update, params_type = local, ParamsType.FULL

        avg_loss = total_loss / max(steps, 1)
        return FLModel(
            params=update,
            params_type=params_type,
            metrics={"train_loss": avg_loss},
            meta={"num_samples": n, "train_loss": avg_loss},
        ).to_shareable()

    def _train_scaffold(self, shareable: Shareable) -> Shareable:
        """SCAFFOLD local update (Karimireddy et al., 2020), option II.

        Corrects each local step by ``-c_i + c`` (the client and server control
        variates), and returns the model delta plus the control-variate delta so
        the server can update both the global model and the server control.
        """
        x = {"W": np.array(shareable.params["W"], dtype=np.float64),
             "b": np.array(shareable.params["b"], dtype=np.float64)}
        c = {k: np.asarray(v, dtype=np.float64) for k, v in shareable["controls"].items()}
        if self._c_i is None:
            self._c_i = {k: np.zeros_like(v) for k, v in x.items()}
        ci = self._c_i
        y = {k: v.copy() for k, v in x.items()}
        n = len(self.y)

        total_loss, steps = 0.0, 0
        for _epoch in range(self.epochs):
            order = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = self.X[idx], self.y[idx]
                probs = _softmax(xb @ y["W"] + y["b"])
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0
                total_loss += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean())
                steps += 1
                grad = probs - onehot
                gW = xb.T @ grad / len(yb)
                gb = grad.mean(axis=0)
                y["W"] -= self.lr * (gW - ci["W"] + c["W"])
                y["b"] -= self.lr * (gb - ci["b"] + c["b"])

        k_lr = max(steps, 1) * self.lr
        ci_new = {k: ci[k] - c[k] + (x[k] - y[k]) / k_lr for k in x}
        delta_y = {k: y[k] - x[k] for k in x}
        delta_c = {k: ci_new[k] - ci[k] for k in x}
        self._c_i = ci_new
        self._current = y

        avg_loss = total_loss / max(steps, 1)
        out = FLModel(
            params=delta_y,
            params_type=ParamsType.DIFF,
            metrics={"train_loss": avg_loss},
            meta={"num_samples": n, "train_loss": avg_loss},
        ).to_shareable()
        out["controls"] = delta_c
        return out

    # -- evaluation --------------------------------------------------------
    def _evaluate(self, shareable: Shareable) -> Shareable:
        W = np.asarray(shareable.params["W"], dtype=np.float64)
        b = np.asarray(shareable.params["b"], dtype=np.float64)
        X, y = (self.eval_dataset if self.eval_dataset is not None else (self.X, self.y))
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        logits = X @ W + b
        preds = logits.argmax(axis=1)
        acc = float((preds == y).mean())
        probs = _softmax(logits)
        loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
        return Shareable(
            params={}, meta={"num_samples": int(len(y)), "accuracy": acc, "eval_loss": loss}
        )

    #: Latest locally-trained params (set after a train round).
    _current: Dict[str, np.ndarray] = {}
