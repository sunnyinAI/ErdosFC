"""Dry-run simulation — Execution Layer.

Run a workflow template *without touching any real system* — the "simulate
before you deploy" mode. The :class:`Simulator` walks a template's actions,
resolves which connectors they touch, marks which would write (and therefore
require approval), evaluates the risk signals you feed it, and returns a
:class:`SimulationReport` of exactly what *would* happen. Nothing is executed:
``executed`` is always ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .connectors import CONNECTORS, ConnectorRegistry
from .risk import RiskModel, RiskScore, Signal
from .templates import WorkflowTemplate


@dataclass
class PlannedAction:
    label: str
    connector: str | None
    write: bool
    would_require_approval: bool


@dataclass
class SimulationReport:
    template_id: str
    template_name: str
    planned_actions: list[PlannedAction] = field(default_factory=list)
    notifications_planned: int = 0
    writes_gated: int = 0
    missing_connectors: list[str] = field(default_factory=list)
    risk: RiskScore | None = None
    executed: bool = False  # invariant: a dry run never executes

    def render(self) -> str:
        lines = [
            f"DRY RUN — {self.template_name}  (executed={self.executed})",
        ]
        if self.risk is not None:
            lines.append(f"  {self.risk.render()}")
            lines.append(f"  would escalate: {'yes' if self.risk.breached else 'no'}")
        lines.append("  planned actions:")
        for a in self.planned_actions:
            tags = []
            if a.write:
                tags.append("write")
            if a.would_require_approval:
                tags.append("needs-approval")
            via = f" via {a.connector}" if a.connector else ""
            suffix = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"    • {a.label}{via}{suffix}")
        lines.append(f"  writes gated: {self.writes_gated} · notifications planned: {self.notifications_planned}")
        if self.missing_connectors:
            lines.append(f"  ⚠ missing connectors: {', '.join(self.missing_connectors)}")
        return "\n".join(lines)


class Simulator:
    """Plans a template's execution without performing any side effects."""

    def __init__(self, registry: ConnectorRegistry = CONNECTORS) -> None:
        self.registry = registry

    def simulate(
        self,
        template: WorkflowTemplate,
        signals: list[Signal] | None = None,
    ) -> SimulationReport:
        report = SimulationReport(template_id=template.id, template_name=template.name)
        report.missing_connectors = template.validate(self.registry)

        for action in template.actions:
            requires_approval = bool(action.write and template.requires_approval)
            report.planned_actions.append(
                PlannedAction(
                    label=action.label,
                    connector=action.connector,
                    write=action.write,
                    would_require_approval=requires_approval,
                )
            )
            if action.write:
                report.writes_gated += 1
            # Messaging/notification connectors imply a notification would be sent.
            if action.connector and action.connector in self.registry:
                if self.registry.get(action.connector).category == "Messaging & Notifications":
                    report.notifications_planned += 1

        if signals:
            report.risk = RiskModel(threshold=template.risk_threshold).evaluate(signals)

        return report
