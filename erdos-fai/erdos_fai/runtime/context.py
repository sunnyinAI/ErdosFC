"""Run context — Runtime.

The shared state threaded through a pipeline run: the active policy engine and
approver (Safety), the tracer, cost tracker, and audit log (Insights), the
trajectory recorder (Learning), and the artifact bag that lets later steps
read earlier outputs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..insights.audit import AuditLog
from ..insights.cost import CostTracker
from ..insights.tracing import Tracer
from ..learning.trajectory import Trajectory
from ..safety.approval import Approver, AutoApprover
from ..safety.gate import WriteGate
from ..safety.policy import PolicyEngine


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    approver: Approver = field(default_factory=AutoApprover)
    tracer: Tracer = field(default_factory=Tracer)
    cost: CostTracker = field(default_factory=CostTracker)
    audit: AuditLog = field(default_factory=AuditLog)
    trajectory: Trajectory | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    safety_interventions: int = 0
    approvals: int = 0
    gate: WriteGate = field(init=False)

    def __post_init__(self) -> None:
        self.gate = WriteGate(policy=self.policy, approver=self.approver, audit=self.audit)

    def note_intervention(self, count: int = 1) -> None:
        self.safety_interventions += count
