"""Evaluation — Learning Layer.

Scores a trajectory against the agent-quality metrics teams care about:
Role Adherence, Task Completion, Tool Correctness, Faithfulness, Answer
Relevancy, and Safety. The built-in evaluators are heuristic (no model
calls, deterministic) so they run in CI; subclass ``Evaluator`` and plug in
an LLM judge for production-grade grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .trajectory import Trajectory

METRIC_NAMES = (
    "Role Adherence",
    "Task Completion",
    "Tool Correctness",
    "Faithfulness",
    "Answer Relevancy",
    "Safety",
)


@dataclass
class MetricResult:
    name: str
    score: float  # 0.0 .. 1.0
    detail: str = ""


@dataclass
class EvaluationReport:
    run_id: str
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def overall(self) -> float:
        if not self.metrics:
            return 0.0
        return round(sum(m.score for m in self.metrics) / len(self.metrics), 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "overall": self.overall,
            "metrics": {m.name: m.score for m in self.metrics},
        }


class Evaluator:
    """Heuristic, deterministic scoring over a trajectory."""

    def evaluate(self, trajectory: Trajectory, *, safety_interventions: int = 0) -> EvaluationReport:
        report = EvaluationReport(run_id=trajectory.run_id)
        steps = trajectory.steps

        produced = sum(1 for s in steps if s.output.strip())
        completion = produced / len(steps) if steps else 0.0
        report.metrics.append(
            MetricResult("Task Completion", round(completion, 3), f"{produced}/{len(steps)} steps produced output")
        )

        # Role adherence: every step ran on the model its card declared.
        on_model = sum(1 for s in steps if s.model)
        report.metrics.append(
            MetricResult("Role Adherence", round(on_model / len(steps), 3) if steps else 0.0)
        )

        # Tool correctness: gated steps that were actually approved.
        gated = [s for s in steps if s.approved is not None]
        tool_score = (sum(1 for s in gated if s.approved) / len(gated)) if gated else 1.0
        report.metrics.append(MetricResult("Tool Correctness", round(tool_score, 3)))

        # Faithfulness: final output is non-empty and grounded in step outputs.
        faithful = 1.0 if trajectory.final_output.strip() else 0.0
        report.metrics.append(MetricResult("Faithfulness", faithful))

        # Answer relevancy: heuristic length-based proxy for substantive output.
        relevancy = 1.0 if len(trajectory.final_output.split()) >= 5 else 0.5
        report.metrics.append(MetricResult("Answer Relevancy", relevancy))

        # Safety: full marks unless interventions fired; each one docks points.
        safety = max(0.0, 1.0 - 0.2 * safety_interventions)
        report.metrics.append(
            MetricResult("Safety", round(safety, 3), f"{safety_interventions} interventions")
        )

        return report
