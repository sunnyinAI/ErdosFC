# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A PyTorch training executor.

``PTTrainer`` is the "Client API" of Erdos FC: it wraps an ordinary PyTorch
training loop so that an existing centralized script becomes a federated client
with almost no changes. Each round it loads the global weights, trains locally
on the site's private dataset, and returns the updated weights together with the
local sample count (used by :class:`~erdos.aggregators.fedavg.FedAvgAggregator`
to weight the average).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..apis.core import Executor, TaskName
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
    ) -> None:
        super().__init__()
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.dataset = dataset
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.momentum = float(momentum)
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    # -- weight (de)serialization -----------------------------------------
    def _load_global(self, params: Dict[str, Any]) -> None:
        state = {k: v.to(self.device) for k, v in params.items()}
        self.model.load_state_dict(state, strict=True)

    def _export_local(self) -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # -- task entry point --------------------------------------------------
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        if task_name != TaskName.TRAIN:
            raise ValueError(f"PTTrainer received unsupported task '{task_name}'")

        self._load_global(shareable.params)
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        running_loss, n_batches = 0.0, 0
        for _epoch in range(self.epochs):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        return Shareable(
            params=self._export_local(),
            meta={
                "num_samples": len(self.dataset),
                "train_loss": avg_loss,
                "round": shareable.get_meta_prop(FLContext.CURRENT_ROUND),
            },
        )
