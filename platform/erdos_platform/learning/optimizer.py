"""Optimizer — Learning Layer.

Turns a graded trajectory into "refined memories": short, actionable lessons
written back into an agent's Memory so it does better next time. This is the
continual-learning loop — agents that improve release over release — kept
deterministic here and swappable for an LLM-driven refiner.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..intelligence.memory import Memory
from .evaluation import EvaluationReport
from .trajectory import Trajectory


@dataclass
class RefinedMemory:
    lesson: str
    source_metric: str
    score: float


class Optimizer:
    """Synthesizes lessons from low-scoring metrics and persists them."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def refine(self, trajectory: Trajectory, report: EvaluationReport) -> list[RefinedMemory]:
        lessons: list[RefinedMemory] = []
        for metric in report.metrics:
            if metric.score >= self.threshold:
                continue
            lessons.append(
                RefinedMemory(
                    lesson=self._lesson_for(metric.name, trajectory),
                    source_metric=metric.name,
                    score=metric.score,
                )
            )
        return lessons

    def apply(self, memory: Memory, lessons: list[RefinedMemory]) -> None:
        for lesson in lessons:
            memory.remember(lesson.lesson, tag="refined")

    @staticmethod
    def _lesson_for(metric: str, trajectory: Trajectory) -> str:
        hints = {
            "Task Completion": "Ensure every step produces output before handing off.",
            "Role Adherence": "Stay within the role and model defined on the agent card.",
            "Tool Correctness": "Confirm write tools are approved before invoking them.",
            "Faithfulness": "Ground the final answer in the prior steps' outputs.",
            "Answer Relevancy": "Return a substantive, on-topic answer — avoid stubs.",
            "Safety": f"Avoid emitting sensitive data; prior run on '{trajectory.task[:40]}' triggered redaction.",
        }
        return hints.get(metric, f"Improve {metric}.")
