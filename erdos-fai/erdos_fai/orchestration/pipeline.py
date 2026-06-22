"""Pipelines — Orchestration Layer.

A ``Pipeline`` runs an ordered list of steps, threading one ``RunContext``
through all of them. Each step's output is stored in ``context.artifacts``
under its ``output_key`` so later steps can reference it. Steps marked
``requires_approval`` block on the Safety Layer's approver before running —
this is the human-in-the-loop gate between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..learning.trajectory import Trajectory
from ..safety.approval import ApprovalRequest
from ..runtime.context import RunContext
from .step import Step


class PipelineHalted(Exception):
    """Raised when an approval gate is denied and the run cannot continue."""


@dataclass
class PipelineResult:
    run_id: str
    task: str
    final_output: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    halted: bool = False
    halted_reason: str = ""


class Pipeline:
    def __init__(self, name: str, steps: list[Step]) -> None:
        self.name = name
        self.steps = steps

    def run(self, task: str, context: RunContext | None = None) -> PipelineResult:
        context = context or RunContext()
        if context.trajectory is None:
            context.trajectory = Trajectory(run_id=context.run_id, task=task)

        context.audit.record("pipeline.start", actor="orchestrator", pipeline=self.name)
        final_output = ""

        with context.tracer.span("pipeline.run", pipeline=self.name):
            for step in self.steps:
                with context.tracer.span("step", step_name=step.name):
                    # Human-in-the-loop gate before a sensitive step.
                    if step.requires_approval:
                        decision = context.approver.request(
                            ApprovalRequest(
                                action=f"run step '{step.name}'",
                                detail=f"agent={step.agent.card.name}",
                                risk=step.approval_risk,
                            )
                        )
                        context.approvals += 1
                        context.audit.record(
                            "approval.decision", actor=decision.approver,
                            step=step.name, approved=decision.approved, reason=decision.reason,
                        )
                        if not decision.approved:
                            context.audit.record("pipeline.halt", actor="orchestrator", step=step.name)
                            return PipelineResult(
                                run_id=context.run_id, task=task, final_output=final_output,
                                artifacts=dict(context.artifacts), halted=True,
                                halted_reason=f"step '{step.name}' denied by {decision.approver}",
                            )

                    prompt = step.render_prompt(task, context.artifacts)
                    result = step.agent.run(prompt, context, step_name=step.name)

                    if step.requires_approval and context.trajectory.steps:
                        context.trajectory.steps[-1].approved = True

                    if step.output_key:
                        context.artifacts[step.output_key] = result.text
                    final_output = result.text

        context.trajectory.final_output = final_output
        context.audit.record("pipeline.end", actor="orchestrator", pipeline=self.name)
        return PipelineResult(
            run_id=context.run_id, task=task, final_output=final_output,
            artifacts=dict(context.artifacts),
        )
