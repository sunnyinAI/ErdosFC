"""Trajectories — Learning Layer.

A trajectory is the recorded execution of a run: each step's input, output,
model, and token usage. Trajectories are what the Evaluator scores and the
Optimizer learns from, and they serialize to JSONL for dataset synthesis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrajectoryStep:
    name: str
    agent: str
    prompt: str
    output: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    approved: bool | None = None


@dataclass
class Trajectory:
    run_id: str
    task: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_output: str = ""
    created_at: float = field(default_factory=time.time)

    def add(self, step: TrajectoryStep) -> TrajectoryStep:
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "created_at": self.created_at,
            "final_output": self.final_output,
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str | Path) -> None:
        """Append this trajectory as one JSONL row (dataset synthesis format)."""
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.to_dict(), default=str) + "\n")
