"""Workflow templates — Execution Layer.

Healthcare- and operations-native templates so a team starts from a governed
pattern instead of a blank canvas. A :class:`WorkflowTemplate` declares its
trigger, the signals it watches, the connectors it touches, the risk threshold
that decides escalation, its SLA / escalation ladder, and the ordered actions it
takes — including which actions write (and therefore require approval). The
bundled :data:`TEMPLATES` library covers the clinical and operational use cases
the platform is built for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .connectors import CONNECTORS, ConnectorRegistry
from .routing import SLA, EscalationLevel, EscalationPolicy

CLINICAL = "clinical"
OPERATIONAL = "operational"


@dataclass
class TemplateAction:
    label: str
    connector: str | None = None   # connector id this action touches, if any
    write: bool = False            # side-effecting => Safety-gated


@dataclass
class WorkflowTemplate:
    id: str
    name: str
    category: str
    summary: str
    trigger: str
    signals: list[str] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    actions: list[TemplateAction] = field(default_factory=list)
    risk_threshold: float = 0.6
    sla: SLA = field(default_factory=SLA)
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    requires_approval: bool = True

    def write_actions(self) -> list[TemplateAction]:
        return [a for a in self.actions if a.write]

    def validate(self, registry: ConnectorRegistry = CONNECTORS) -> list[str]:
        """Return a list of referenced connector ids missing from the registry."""
        return [cid for cid in self.connectors if cid not in registry]

    def describe(self) -> str:
        lines = [
            f"{self.name}  [{self.category}]",
            f"  trigger:    {self.trigger}",
            f"  signals:    {', '.join(self.signals) or '—'}",
            f"  connectors: {', '.join(self.connectors) or '—'}",
            f"  risk ≥ {self.risk_threshold:.2f} → escalate · ack {self.sla.ack_minutes:.0f}m / resolve {self.sla.resolve_minutes:.0f}m",
            "  actions:",
        ]
        for a in self.actions:
            flag = " (write)" if a.write else ""
            via = f" via {a.connector}" if a.connector else ""
            lines.append(f"    • {a.label}{flag}{via}")
        return "\n".join(lines)


class TemplateLibrary:
    def __init__(self, templates: list[WorkflowTemplate] | None = None) -> None:
        self._by_id: dict[str, WorkflowTemplate] = {}
        for t in templates or []:
            self.register(t)

    def register(self, template: WorkflowTemplate) -> WorkflowTemplate:
        self._by_id[template.id] = template
        return template

    def get(self, template_id: str) -> WorkflowTemplate:
        return self._by_id[template_id]

    def all(self) -> list[WorkflowTemplate]:
        return list(self._by_id.values())

    def by_category(self, category: str) -> list[WorkflowTemplate]:
        return [t for t in self._by_id.values() if t.category == category]

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


def _ladder(*levels: tuple[float, str, str]) -> EscalationPolicy:
    return EscalationPolicy([EscalationLevel(after_minutes=m, role=r, channel=c) for (m, r, c) in levels])


TEMPLATES = TemplateLibrary([
    WorkflowTemplate(
        id="inpatient-deterioration",
        name="Inpatient Deterioration & Early Escalation",
        category=CLINICAL,
        summary="Watch vitals and labs for deterioration trends and escalate to rapid response before a code.",
        trigger="New vitals or lab result for an admitted patient",
        signals=["heart_rate", "resp_rate", "spo2", "lactate_trend", "news2"],
        connectors=["epic", "fhir", "lis", "twilio-sms", "pagerduty"],
        risk_threshold=0.7,
        sla=SLA(ack_minutes=10, resolve_minutes=60),
        escalation=_ladder((10, "charge_nurse", "in_app"), (20, "rapid_response", "sms"), (30, "attending", "pagerduty")),
        actions=[
            TemplateAction("Evaluate deterioration risk from connected signals"),
            TemplateAction("Create rapid-response review task", connector="epic", write=True),
            TemplateAction("Notify charge nurse", connector="twilio-sms", write=True),
            TemplateAction("Escalate to rapid response on SLA breach", connector="pagerduty", write=True),
        ],
    ),
    WorkflowTemplate(
        id="discharge-navigation",
        name="Discharge Navigation",
        category=CLINICAL,
        summary="Surface discharge barriers and route prioritized remediation tasks to nurses and case managers.",
        trigger="Discharge order placed or readiness review scheduled",
        signals=["barrier_count", "los_excess_days", "pending_consults"],
        connectors=["epic", "fhir", "outlook-mail", "qgenda"],
        risk_threshold=0.5,
        sla=SLA(ack_minutes=30, resolve_minutes=240),
        escalation=_ladder((30, "case_manager", "in_app"), (120, "discharge_lead", "email")),
        actions=[
            TemplateAction("Detect discharge barriers from chart + orders"),
            TemplateAction("Prioritize barriers by impact on length of stay"),
            TemplateAction("Create remediation tasks for nurses", connector="epic", write=True),
            TemplateAction("Email summary to discharge lead", connector="outlook-mail", write=True),
        ],
    ),
    WorkflowTemplate(
        id="critical-results",
        name="Critical Results Management",
        category=CLINICAL,
        summary="Track critical lab and imaging results to closed-loop acknowledgement with audit.",
        trigger="Critical lab value or imaging finding posted",
        signals=["criticality", "result_age_minutes", "unacked"],
        connectors=["lis", "pacs-dicom", "ris", "twilio-sms"],
        risk_threshold=0.8,
        sla=SLA(ack_minutes=15, resolve_minutes=60),
        escalation=_ladder((15, "ordering_provider", "sms"), (30, "covering_provider", "sms")),
        actions=[
            TemplateAction("Score result criticality and staleness"),
            TemplateAction("Route to ordering provider with read-back", connector="twilio-sms", write=True),
            TemplateAction("Log closed-loop acknowledgement"),
        ],
    ),
    WorkflowTemplate(
        id="care-transitions",
        name="Care Transitions & Follow-up",
        category=CLINICAL,
        summary="Assess discharge readiness and coordinate post-discharge follow-up appointments.",
        trigger="Patient flagged for transition of care",
        signals=["readmission_risk", "followup_gap_days", "med_reconciliation"],
        connectors=["epic", "fhir", "epic-cadence", "surescripts"],
        risk_threshold=0.55,
        sla=SLA(ack_minutes=60, resolve_minutes=480),
        escalation=_ladder((60, "transition_coordinator", "in_app")),
        actions=[
            TemplateAction("Compute readmission risk and follow-up gaps"),
            TemplateAction("Book follow-up appointment", connector="epic-cadence", write=True),
            TemplateAction("Reconcile medications", connector="surescripts", write=True),
        ],
    ),
    WorkflowTemplate(
        id="transfer-planning",
        name="Transfer & Capacity Planning",
        category=OPERATIONAL,
        summary="Surface expected capacity from admissions and discharges to plan transfers ahead of demand.",
        trigger="Hourly capacity recompute or transfer request",
        signals=["expected_admits", "expected_discharges", "occupied_beds"],
        connectors=["epic", "hl7v2", "teamup", "tableau"],
        risk_threshold=0.6,
        sla=SLA(ack_minutes=30, resolve_minutes=180),
        escalation=_ladder((30, "bed_manager", "in_app"), (90, "house_supervisor", "sms")),
        actions=[
            TemplateAction("Forecast net capacity from ADT events"),
            TemplateAction("Reserve beds for expected transfers", connector="teamup", write=True),
            TemplateAction("Publish capacity outlook", connector="tableau"),
        ],
    ),
    WorkflowTemplate(
        id="capacity-management",
        name="Hospital Capacity & Patient Flow",
        category=OPERATIONAL,
        summary="Optimize staff allocation against predicted census across ER, OR, and inpatient units.",
        trigger="Census or staffing change",
        signals=["census_forecast", "staff_ratio", "boarding_count"],
        connectors=["epic", "qgenda", "workday", "power-bi"],
        risk_threshold=0.6,
        sla=SLA(ack_minutes=60, resolve_minutes=240),
        escalation=_ladder((60, "nurse_manager", "in_app"), (120, "operations_director", "email")),
        actions=[
            TemplateAction("Forecast census and staffing gaps"),
            TemplateAction("Propose staff reallocation", connector="qgenda", write=True),
            TemplateAction("Publish flow dashboard", connector="power-bi"),
        ],
    ),
    WorkflowTemplate(
        id="procurement-approval",
        name="Procurement Approval Routing",
        category=OPERATIONAL,
        summary="Route cross-system purchase requests through policy-based, multi-level approvals.",
        trigger="Purchase request submitted",
        signals=["amount", "budget_variance", "vendor_risk"],
        connectors=["sap", "coupa", "teams", "okta"],
        risk_threshold=0.65,
        sla=SLA(ack_minutes=120, resolve_minutes=1440),
        escalation=_ladder((120, "department_head", "in_app"), (480, "finance", "email")),
        actions=[
            TemplateAction("Validate request against budget and policy"),
            TemplateAction("Route for approval by amount tier", connector="teams", write=True),
            TemplateAction("Create purchase order on approval", connector="coupa", write=True),
        ],
    ),
    WorkflowTemplate(
        id="utilization-review",
        name="Insurance Utilization Review",
        category=OPERATIONAL,
        summary="Optimize utilization-review creation and prior-authorization for case managers.",
        trigger="Admission or service requiring review",
        signals=["los_vs_benchmark", "denial_risk", "auth_status"],
        connectors=["epic", "availity", "change-healthcare", "outlook-mail"],
        risk_threshold=0.6,
        sla=SLA(ack_minutes=120, resolve_minutes=720),
        escalation=_ladder((120, "case_manager", "in_app"), (360, "ur_lead", "email")),
        actions=[
            TemplateAction("Estimate denial risk and review need"),
            TemplateAction("Submit prior authorization", connector="availity", write=True),
            TemplateAction("Create case-manager review task", connector="epic", write=True),
        ],
    ),
])
