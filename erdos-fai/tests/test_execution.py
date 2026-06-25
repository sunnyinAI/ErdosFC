"""Execution Layer tests — offline, deterministic."""

from __future__ import annotations

import pytest

from erdos_fai.execution import (
    CONNECTORS,
    TEMPLATES,
    AccessControl,
    AccessDenied,
    CLINICIAN,
    EscalationLevel,
    EscalationPolicy,
    RBACApprover,
    RiskModel,
    Role,
    Signal,
    Simulator,
    SLA,
    TaskRouter,
    User,
    VIEWER,
)
from erdos_fai.execution.rbac import WORKFLOW_RUN
from erdos_fai.insights.audit import AuditLog
from erdos_fai.safety.approval import ApprovalRequest


# ── Connectors ────────────────────────────────────────────────────────────
def test_connector_registry_has_50_plus():
    assert len(CONNECTORS) >= 50
    assert "epic" in CONNECTORS
    assert CONNECTORS.get("epic").write is True


def test_connectors_grouped_by_category():
    cats = CONNECTORS.categories()
    assert "EHR" in cats and "Messaging & Notifications" in cats
    assert all(CONNECTORS.by_category(c) for c in cats)


def test_connector_to_mcp_bridge():
    mcp = CONNECTORS.get("fhir").to_mcp()
    assert mcp.name == "fhir"
    assert mcp.url.startswith("mcp://")


# ── Risk ──────────────────────────────────────────────────────────────────
def test_risk_breaches_threshold():
    model = RiskModel(threshold=0.5)
    score = model.evaluate([Signal("a", 1.0), Signal("b", 1.0)])
    assert score.score == pytest.approx(1.0)
    assert score.breached and score.band == "critical"


def test_risk_below_threshold():
    score = RiskModel(threshold=0.9).evaluate([Signal("a", 0.1)])
    assert not score.breached and score.band == "low"


def test_risk_empty_signals_is_zero():
    score = RiskModel().evaluate([])
    assert score.score == 0.0 and not score.breached


# ── Routing ───────────────────────────────────────────────────────────────
def test_task_creation_records_audit_and_outputs():
    audit = AuditLog()
    router = TaskRouter(audit=audit)
    task = router.create_task(title="t", owner="nurse", risk="high", next_steps=["x"])
    out = task.outputs()
    assert out["action_id"] == task.action_id
    assert out["assigned_owner"] == "nurse"
    assert task.audit_event_id is not None
    # create + notify both audited
    assert len(audit.records) >= 2 and audit.verify()


def test_escalation_fires_on_sla_breach():
    router = TaskRouter()
    policy = EscalationPolicy([
        EscalationLevel(10, "charge_nurse"),
        EscalationLevel(20, "attending", "sms"),
    ])
    task = router.create_task(title="t", owner="nurse", sla=SLA(ack_minutes=10))
    fired = router.escalate(task, policy, elapsed_minutes=25)
    assert len(fired) == 2
    assert task.escalation_status.startswith("L2")


def test_acknowledged_task_does_not_escalate():
    router = TaskRouter()
    policy = EscalationPolicy([EscalationLevel(5, "charge_nurse")])
    task = router.create_task(title="t", owner="nurse")
    router.acknowledge(task, by="nurse")
    assert router.escalate(task, policy, elapsed_minutes=99) == []


# ── RBAC ──────────────────────────────────────────────────────────────────
def test_access_control_require_raises():
    ac = AccessControl()
    guest = User("g", "guest", roles=[VIEWER])
    assert ac.can(guest, "workflow.view")
    with pytest.raises(AccessDenied):
        ac.require(guest, WORKFLOW_RUN)


def test_rbac_approver_gates_by_risk():
    clinician = RBACApprover(User("u", "doc", roles=[CLINICIAN]))
    viewer = RBACApprover(User("v", "guest", roles=[VIEWER]))
    high = ApprovalRequest(action="write", detail="", risk="high")
    assert clinician.request(high).approved is True
    assert viewer.request(high).approved is False


def test_admin_role_grants_everything():
    admin = User("a", "root", roles=[Role("administrator", frozenset({"admin"}))])
    assert admin.can("anything.at.all")


# ── Templates ─────────────────────────────────────────────────────────────
def test_template_library_populated_and_valid():
    assert len(TEMPLATES) >= 5
    for t in TEMPLATES:
        # every referenced connector exists in the registry
        assert t.validate(CONNECTORS) == [], f"{t.id} has missing connectors"


def test_template_has_write_actions():
    t = TEMPLATES.get("inpatient-deterioration")
    assert t.write_actions()
    assert "epic" in t.connectors


# ── Simulation (dry-run) ──────────────────────────────────────────────────
def test_simulation_never_executes_and_gates_writes():
    t = TEMPLATES.get("discharge-navigation")
    report = Simulator().simulate(t, signals=[Signal("barrier_count", 0.9)])
    assert report.executed is False
    assert report.writes_gated == len(t.write_actions())
    assert all(a.would_require_approval for a in report.planned_actions if a.write)
    assert report.risk is not None
