"""Bundled demo: a healthcare-triage agent pipeline.

Three agents run in sequence — Intake → Triage → Referral — and the run
exercises every layer of the platform:

  • Intelligence  composable agent cards with skills + tools
  • Orchestration a 3-step pipeline passing artifacts between agents
  • Safety        PHI/PII redaction + a human-in-the-loop gate before the
                  referral note is "written to the EHR"
  • Insights      tracing, cost/ROI, and a tamper-evident audit trail
  • Learning      trajectory recording, evaluation, and refined memories

Runs fully offline with the EchoProvider — no API key required. Set
``ANTHROPIC_API_KEY`` (and install the ``anthropic`` extra) to run it on Claude.
"""

from __future__ import annotations

from . import (
    Agent,
    AgentCard,
    AutoApprover,
    Evaluator,
    Optimizer,
    Pipeline,
    Policy,
    PolicyEngine,
    RunContext,
    Skill,
    Step,
    build_dashboard,
)
from .insights.metrics import Dashboard  # noqa: F401  (re-exported for callers)

# A deliberately messy, PHI-laden intake message.
SAMPLE_COMPLAINT = (
    "Patient John Doe (MRN: 4821990, DOB 03/14/1971), reachable at "
    "john.doe@example.com or 415-555-0199, reports crushing chest pain "
    "radiating to the left arm for the past 30 minutes, with shortness of breath."
)


def build_pipeline() -> Pipeline:
    intake = Agent(
        AgentCard(
            name="Intake",
            system="You are a clinical intake agent. Extract structured fields from a complaint.",
            skills=[
                Skill(
                    "structured-intake",
                    "Return chief complaint, duration, and associated symptoms as bullet points.",
                )
            ],
        )
    )
    triage = Agent(
        AgentCard(
            name="Triage",
            system="You are a triage nurse agent. Assign an acuity level (1-5) and justify it.",
            skills=[Skill("esi", "Use the Emergency Severity Index. Chest pain + dyspnea is high acuity.")],
        )
    )
    referral = Agent(
        AgentCard(
            name="Referral",
            system="You are a referral agent. Draft a concise hand-off note for the care team.",
        )
    )

    return Pipeline(
        "healthcare-triage",
        [
            Step("intake", intake, prompt="Intake this complaint:\n{input}", output_key="intake"),
            Step("triage", triage, prompt="Given this intake, assign acuity:\n{intake}", output_key="triage"),
            Step(
                "referral",
                referral,
                prompt="Write the EHR referral note from:\nINTAKE: {intake}\nTRIAGE: {triage}",
                output_key="referral",
                requires_approval=True,  # HITL gate before writing to the EHR
                approval_risk="high",
            ),
        ],
    )


def run_demo() -> int:
    # Safety: a HIPAA-style policy that redacts PHI/PII and gates writes.
    policy = PolicyEngine(Policy(name="HIPAA", version="v2.4"))
    context = RunContext(policy=policy, approver=AutoApprover(approve=True, name="dr.smith"))

    print("Erdos-FAI — healthcare-triage demo\n" + "=" * 44)
    print(f"\nEnforced policies ({policy.policy.name} {policy.policy.version}): "
          + ", ".join(policy.policy.enforced()))

    pipeline = build_pipeline()
    result = pipeline.run(SAMPLE_COMPLAINT, context)

    print("\n── Pipeline output (PHI redacted) ─────────────")
    for key in ("intake", "triage", "referral"):
        if key in result.artifacts:
            print(f"\n[{key}]\n{result.artifacts[key]}")

    # Learning: evaluate the trajectory and synthesize refined memories.
    report = Evaluator().evaluate(context.trajectory, safety_interventions=context.safety_interventions)
    lessons = Optimizer().refine(context.trajectory, report)

    print("\n── Learning: evaluation ───────────────────────")
    for metric in report.metrics:
        print(f"  {metric.name:<18} {metric.score:.2f}")
    print(f"  {'OVERALL':<18} {report.overall:.2f}")
    if lessons:
        print("  refined memories:")
        for lesson in lessons:
            print(f"   • ({lesson.source_metric}) {lesson.lesson}")

    # Insights: dashboard snapshot (assume this replaced ~25 min of manual work).
    dashboard = build_dashboard(context, human_minutes=25)
    print("\n── Insights ───────────────────────────────────")
    print(dashboard.render())

    print(f"\nAudit records: {len(context.audit.records)} (hash chain "
          f"{'verified ✓' if context.audit.verify() else 'BROKEN ✗'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
