<h1 align="center">Erdos-FAI</h1>

<p align="center">
  <em>Enterprise Agentic Technology — the execution layer for governed AI agents.</em><br>
  Composable agent cards · multi-agent orchestration · human-in-the-loop safety · continual learning · real-time observability · connected execution.
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/release-0.1.0-blue">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="deps" src="https://img.shields.io/badge/core%20deps-stdlib%20only-success">
</p>

---

## Why "Erdos"?

Paul Erdős did his life's work through **collaboration** — 500+ co-authors, one shared body of knowledge. The Erdos-FAI applies that idea to AI agents: many specialized agents collaborating through one governed runtime to do real enterprise work, safely.

## What it is

Erdos is an **agent lifecycle & execution platform**. You build agents, test them, deploy them into pipelines, let them improve from their own runs, govern every step, and connect them to the systems you already run — from a single, observable, audited system. It is organized as **six tightly-integrated layers**:

| # | Layer | What it gives you | Package |
|---|-------|-------------------|---------|
| 01 | **Intelligence** | Composable *agent cards* — prompt + skills + memory + tools + MCP servers wired into one reusable, versioned object. | `erdos_fai.intelligence` |
| 02 | **Orchestration** | Multi-agent *pipelines* — chain agents into steps that pass artifacts forward, with human-in-the-loop gates between them. | `erdos_fai.orchestration` |
| 03 | **Safety** | PHI/PII redaction, a policy engine, and **write gates** that block side-effecting actions behind multi-approver human sign-off. | `erdos_fai.safety` |
| 04 | **Learning** | Trajectory recording, agent **evaluation** (role adherence, task completion, faithfulness, safety…), and an optimizer that writes *refined memories* back to agents. | `erdos_fai.learning` |
| 05 | **Insights** | Real-time **cost & ROI**, event-replay **tracing**, and a tamper-evident (hash-chained) **audit trail**. | `erdos_fai.insights` |
| 06 | **Execution** | **60+ connectors** into real systems, governed workflow **templates**, **risk scoring**, task **routing** with SLA escalation, **RBAC**, and **dry-run** simulation. | `erdos_fai.execution` |

The **core is dependency-light** — pure standard library, so it reads and runs anywhere. It ships with an offline `EchoProvider` so a full pipeline runs with **no API key**, and an `AnthropicProvider` that calls Claude (default `claude-opus-4-8`) when you want live model calls.

## Install

```bash
cd erdos-fai
pip install -e .                      # core (standard library only)
pip install -e ".[anthropic]"         # + the Claude SDK for live model calls
```

## Quickstart

```python
from erdos_fai import Agent, AgentCard, Pipeline, Step, RunContext, Skill

# 01 Intelligence — a composable agent card
triage = Agent(AgentCard(
    name="Triage",
    system="You triage incoming support tickets and assign a priority.",
    skills=[Skill("priority", "P1 = outage, P2 = degraded, P3 = question.")],
))

# 02 Orchestration — one-step pipeline (add more steps to chain agents)
pipe = Pipeline("support", [Step("triage", triage, output_key="triage")])

result = pipe.run("Customer reports the whole dashboard is down")
print(result.final_output)
```

### Run the bundled demo

A 3-agent **healthcare-triage** pipeline (Intake → Triage → Referral) that exercises the first five layers — PHI redaction, a HITL gate before the referral is "written to the EHR", cost/ROI, an audit trail, and post-run evaluation (run `erdos-fai execute` for the sixth, Execution, layer):

```bash
erdos-fai demo
# or:  python -m examples.healthcare_triage.run
```

```text
── Insights ───────────────────────────────────
┌─ Erdos Insights ─────────────────────────────
│ total cost           $0.0000
│ safety interventions 4
│ approvals            1
│ dollars saved        $31.25
│ audit chain          verified ✓
└──────────────────────────────────────────────
```

## Running on Claude

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
erdos-fai demo            # now runs on claude-opus-4-8
```

Agents pick a provider automatically (`default_provider()` → Claude when a key is present, else the offline echo provider), or you can pass one explicitly:

```python
from erdos_fai import Agent, AgentCard, AnthropicProvider

agent = Agent(AgentCard(name="Analyst", model="claude-opus-4-8"),
              provider=AnthropicProvider())
```

## The six layers, in code

```python
from erdos_fai import (
    AgentCard, Skill, Tool,                 # 01 Intelligence
    Pipeline, Step,                         # 02 Orchestration
    Policy, PolicyEngine, WriteGate,        # 03 Safety
    Evaluator, Optimizer,                   # 04 Learning
    CostTracker, Tracer, AuditLog,          # 05 Insights
    CONNECTORS, TEMPLATES, RiskModel,       # 06 Execution
    Simulator, TaskRouter, RBACApprover,    # 06 Execution
)
```

- **Safety** — `Policy(redact_phi=True, require_approval_for_writes=True)` drives a `PolicyEngine` that scrubs PHI/PII on every input and output, and a `WriteGate` that requires an `Approver` (auto / console / your callback) before any `Tool(..., write=True)` runs.
- **Learning** — after a run, `Evaluator().evaluate(context.trajectory)` scores six agent-quality metrics; `Optimizer().refine(...)` turns low scores into refined memories you persist back onto the agent card.
- **Insights** — `build_dashboard(context, human_minutes=...)` rolls cost, latency, safety interventions, approvals, ROI, and audit-chain status into one snapshot.
- **Execution** — `CONNECTORS` is a registry of 60+ pre-built integrations; `TEMPLATES` are governed workflow patterns; `RiskModel` scores signals against a threshold; `Simulator().simulate(template)` dry-runs a workflow (nothing executes, writes are gated); `TaskRouter` creates review tasks, notifies, and escalates on SLA breach; `RBACApprover` only clears an approval if the user holds the permission for that risk.

### Execution Layer in one snippet

```python
from erdos_fai import TEMPLATES, Simulator, Signal, RiskModel, TaskRouter, AuditLog

wf = TEMPLATES.get("inpatient-deterioration")
signals = [Signal("news2", 7, floor=0, ceiling=9), Signal("spo2", 89, floor=100, ceiling=85)]

# 1. Dry-run first — nothing touches a live system.
print(Simulator().simulate(wf, signals=signals).render())

# 2. Score risk; if it breaches the template threshold, route + escalate.
score = RiskModel(threshold=wf.risk_threshold).evaluate(signals)
if score.breached:
    router = TaskRouter(audit=AuditLog())
    task = router.create_task(title="Rapid response", owner="charge_nurse",
                              sla=wf.sla, risk="high", source=wf.id)
    router.escalate(task, wf.escalation, elapsed_minutes=22)
    print(task.outputs())   # action_id, assigned_owner, escalation_status, audit_event_id
```

## Project layout

```
erdos-fai/
  erdos_fai/
    intelligence/   01  agent cards, skills, memory, tools, MCP
    orchestration/  02  steps + pipelines
    safety/         03  redaction, policy, approval, write gates
    learning/       04  trajectories, evaluation, optimizer
    insights/       05  tracing, cost, audit, metrics
    execution/      06  connectors, templates, risk, routing, rbac, simulation
    runtime/            agent engine, run context, LLM providers
    cli.py              `erdos-fai` command
    examples_demo.py    healthcare-triage pipeline demo
    execution_demo.py   execution-layer demo (connectors → routing)
  examples/
    healthcare_triage/  runnable multi-agent demo
  tests/                pytest suite (offline, deterministic)
```

## CLI

```bash
erdos-fai info          # layers, active provider, connector & template counts
erdos-fai demo          # healthcare-triage pipeline (all 6 layers)
erdos-fai execute       # Execution Layer demo: dry-run → risk → RBAC → route → escalate
erdos-fai connectors    # list the 60+ pre-built connector catalog
erdos-fai templates     # list the governed workflow template library
```

## Tests

```bash
cd erdos-fai
pip install -e ".[dev]"
pytest -q
```

## License

Apache License 2.0. © 2026 Sunny Gupta.

> Erdos-FAI is an independent, educational agent-lifecycle framework. It is
> inspired by the architecture of modern enterprise agent platforms and is not
> affiliated with or endorsed by any of them.
