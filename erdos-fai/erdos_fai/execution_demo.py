"""Bundled demo: the Execution Layer end to end.

Takes the *Inpatient Deterioration* workflow template and runs it through the
whole execution surface — **offline, no API key**:

  • Connectors   resolve the systems the template touches
  • Simulation   dry-run it first (nothing is executed; writes are gated)
  • Risk         score live signals against the template's threshold
  • RBAC         a clinician approves the high-risk write; a viewer could not
  • Routing      create a review task, notify, escalate on SLA breach, resolve
  • Insights     every step lands in a tamper-evident audit trail
"""

from __future__ import annotations

from .execution import (
    CONNECTORS,
    CLINICIAN,
    VIEWER,
    AccessControl,
    RBACApprover,
    RiskModel,
    Signal,
    Simulator,
    TaskRouter,
    TEMPLATES,
    User,
)
from .insights.audit import AuditLog
from .safety.approval import ApprovalRequest


def run_demo() -> int:
    template = TEMPLATES.get("inpatient-deterioration")
    audit = AuditLog()

    print("Erdos-FAI — Execution Layer demo\n" + "=" * 44)
    print(f"\nTemplate: {template.name}")
    print(f"  systems: {', '.join(c.name for c in (CONNECTORS.get(i) for i in template.connectors))}")

    # Live signals for one deteriorating patient (normalized ranges per signal).
    signals = [
        Signal("heart_rate", 132, weight=1.0, floor=60, ceiling=140),
        Signal("resp_rate", 28, weight=1.2, floor=12, ceiling=30),
        Signal("spo2", 89, weight=1.4, floor=100, ceiling=85),     # lower is worse → inverted range
        Signal("lactate_trend", 0.8, weight=1.1),
        Signal("news2", 7, weight=1.5, floor=0, ceiling=9),
    ]

    # 1. Dry run — prove what would happen before touching anything live.
    print("\n── Simulation (dry run) ───────────────────────")
    report = Simulator().simulate(template, signals=signals)
    print(report.render())

    # 2. Go live. Score the risk against the template threshold.
    score = RiskModel(threshold=template.risk_threshold).evaluate(signals)
    print("\n── Live execution ─────────────────────────────")
    print(f"  {score.render()}")

    if not score.breached:
        print("  risk below threshold — no escalation.")
        return 0

    # 3. RBAC: only an authorized clinician may approve the high-risk action.
    access = AccessControl()
    clinician = User("u-emily", "dr.emily", roles=[CLINICIAN])
    viewer = User("u-guest", "guest", roles=[VIEWER])
    req = ApprovalRequest(action="open rapid-response task", detail=template.name, risk="high")
    print(f"  viewer can approve high risk?    {access.can_approve(viewer, 'high')}")
    decision = RBACApprover(clinician, access).request(req)
    print(f"  clinician approval:              {decision.approved} ({decision.reason})")

    # 4. Routing: create the review task, then escalate on SLA breach.
    router = TaskRouter(audit=audit, default_channel="in_app")
    task = router.create_task(
        title="Rapid response: deterioration detected",
        owner="charge_nurse",
        sla=template.sla,
        risk="high",
        source=template.id,
        next_steps=["Bedside assessment", "Notify attending", "Consider ICU transfer"],
    )
    print(f"\n  task created: {task.action_id} → {task.assigned_owner}")
    fired = router.escalate(task, template.escalation, elapsed_minutes=22)  # nurse never acked
    print(f"  escalated to: {', '.join(l.role for l in fired)}  (status={task.escalation_status})")
    router.acknowledge(task, by="rapid_response")
    router.resolve(task, by="dr.emily")

    print("\n  task outputs:")
    for k, v in task.outputs().items():
        print(f"    {k:<18} {v}")

    print(f"\nAudit records: {len(audit.records)} (hash chain "
          f"{'verified ✓' if audit.verify() else 'BROKEN ✗'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
