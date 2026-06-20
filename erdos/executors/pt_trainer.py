# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A PyTorch training executor.

``PTTrainer`` is the "Client API" of Erdos FC: it wraps an ordinary PyTorch
training loop so that an existing centralized script becomes a federated client
with almost no changes. Each round it loads the global weights, trains locally
on the site's private dataset, and returns its update.

What the update contains is configurable:

* ``params_transfer="full"`` (default) sends the complete local weights;
* ``params_transfer="diff"`` sends only ``local - global`` as a ``DIFF`` —
  smaller on the wire and required by server-side optimizers (FedOpt) and
  diff-based privacy.

Setting ``fedprox_mu > 0`` adds the FedProx proximal term to the local loss, and
the executor also handles the ``EVALUATE`` and ``SUBMIT_MODEL`` tasks, so a
client can be scored federally or contribute its model to cross-site evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..apis import tensor
from ..apis.core import Executor, TaskName
from ..apis.model import FLModel, ParamsType
from ..apis.shareable import FLContext, Shareable


class PTTrainer(Executor):
    """Local SGD trainer for a single client/site.

    Args:
        model: a ``torch.nn.Module``. Its architecture must match the global model.
        dataset: this site's *private* training data.
        epochs: number of local epochs per round.
        lr: SGD learning rate.
        batch_size: local mini-batch size.
        momentum: SGD momentum.
        loss_fn: loss function (defaults to ``CrossEntropyLoss``).
        device: ``"cpu"``, ``"cuda"``, or ``None`` to auto-select.
        params_transfer: ``"full"`` to send whole weights, ``"diff"`` to send the
            ``local - global`` delta (a ``DIFF`` payload).
        fedprox_mu: if > 0, add the FedProx proximal term ``(mu/2)||w - w_global||^2``
            to the local loss (Li et al., 2020).
        eval_dataset: optional held-out data for the ``EVALUATE`` task (falls back
            to ``dataset``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 32,
        momentum: float = 0.9,
        loss_fn: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
        params_transfer: str = "full",
        fedprox_mu: float = 0.0,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__()
        if params_transfer not in ("full", "diff"):
            raise ValueError("params_transfer must be 'full' or 'diff'")
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.momentum = float(momentum)
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()
        self.params_transfer = params_transfer
        self.fedprox_mu = float(fedprox_mu)

    # -- weight (de)serialization -----------------------------------------
    def _load_global(self, params: Dict[str, Any]) -> None:
        state = {k: v.to(self.device) for k, v in params.items()}
        self.model.load_state_dict(state, strict=True)

    def _export_local(self) -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # -- task entry point --------------------------------------------------
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        if task_name == TaskName.TRAIN:
            return self._train(shareable, fl_ctx)
        if task_name == TaskName.EVALUATE:
            return self._evaluate(shareable, fl_ctx)
        if task_name == TaskName.SUBMIT_MODEL:
            return FLModel(
                params=self._export_local(),
                params_type=ParamsType.FULL,
                meta={"num_samples": len(self.dataset)},
            ).to_shareable()
        raise ValueError(f"PTTrainer received unsupported task '{task_name}'")

    # -- tasks -------------------------------------------------------------
    def _train(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        self._load_global(shareable.params)
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # Snapshot the global trainable params for the FedProx proximal term.
        prox_ref = (
            [p.detach().clone() for p in self.model.parameters()]
            if self.fedprox_mu > 0
            else None
        )

        running_loss, n_batches = 0.0, 0
        for _epoch in range(self.epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                if prox_ref is not None:
                    prox = sum(
                        ((p - p0) ** 2).sum() for p, p0 in zip(self.model.parameters(), prox_ref)
                    )
                    loss = loss + 0.5 * self.fedprox_mu * prox
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        local = self._export_local()
        if self.params_transfer == "diff":
            global_in = {k: v.detach().cpu() for k, v in shareable.params.items()}
            update, params_type = tensor.subtract_params(local, global_in), ParamsType.DIFF
        else:
            update, params_type = local, ParamsType.FULL

        return FLModel(
            params=update,
            params_type=params_type,
            metrics={"train_loss": avg_loss},
            current_round=shareable.get_meta_prop(FLContext.CURRENT_ROUND),
            meta={"num_samples": len(self.dataset), "train_loss": avg_loss},
        ).to_shareable()

    def _evaluate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        self._load_global(shareable.params)
        self.model.eval()
        dataset = self.eval_dataset if self.eval_dataset is not None else self.dataset
        loader = DataLoader(dataset, batch_size=self.batch_size)
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss_sum += float(self.loss_fn(outputs, targets)) * inputs.size(0)
                correct += int((outputs.argmax(1) == targets).sum().item())
                total += int(targets.size(0))
        total = max(total, 1)
        return Shareable(
            params={},
            meta={
                "num_samples": total,
                "accuracy": correct / total,
                "eval_loss": loss_sum / total,
            },
        )
