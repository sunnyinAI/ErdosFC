# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Data loading + partitioning for the hello-pytorch-mnist example.

Tries to download MNIST via torchvision. If that is unavailable (no torchvision,
no network, etc.) it falls back to a small *synthetic* image-classification
dataset so the example always runs end-to-end with no external dependencies.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset, Subset, TensorDataset


def _try_load_mnist(root: str):
    """Return (train, test, 'MNIST') or None if MNIST cannot be obtained."""
    try:
        from torchvision import datasets, transforms

        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(root, train=True, download=True, transform=tf)
        test = datasets.MNIST(root, train=False, download=True, transform=tf)
        return train, test, "MNIST"
    except Exception as exc:  # noqa: BLE001 - any failure -> use the fallback
        print(f"[data] MNIST unavailable ({exc.__class__.__name__}); using synthetic data.")
        return None


def _make_synthetic(
    n_train: int = 4000,
    n_test: int = 1000,
    num_classes: int = 10,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, str]:
    """A learnable toy dataset: each class stamps a bright block at a fixed spot."""
    gen = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        labels = torch.randint(0, num_classes, (n,), generator=gen)
        images = torch.randn(n, 1, 28, 28, generator=gen) * 0.3
        for i in range(n):
            c = int(labels[i])
            row = (c // 5) * 13
            col = (c % 5) * 5
            images[i, 0, row:row + 12, col:col + 5] += 2.5
        return TensorDataset(images, labels)

    return make(n_train), make(n_test), "synthetic"


def partition_iid(dataset: Dataset, num_clients: int, seed: int = 0) -> List[Subset]:
    """Split ``dataset`` into ``num_clients`` disjoint IID shards."""
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    shard = n // num_clients
    shards: List[Subset] = []
    for i in range(num_clients):
        start = i * shard
        end = (i + 1) * shard if i < num_clients - 1 else n
        shards.append(Subset(dataset, perm[start:end]))
    return shards


def load_data(num_clients: int, data_root: str = "data", seed: int = 0):
    """Return (list[client_dataset], test_dataset, source_name)."""
    loaded = _try_load_mnist(data_root)
    if loaded is None:
        train, test, source = _make_synthetic(seed=seed)
    else:
        train, test, source = loaded
    client_datasets = partition_iid(train, num_clients, seed=seed)
    return client_datasets, test, source
