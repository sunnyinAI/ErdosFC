# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A small convolutional network for 28x28 single-channel images (MNIST)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Two conv blocks + two linear layers. No BatchNorm (keeps FedAvg simple)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)                 # -> 32*7*7 = 1568
        x = F.relu(self.fc1(x))
        return self.fc2(x)
