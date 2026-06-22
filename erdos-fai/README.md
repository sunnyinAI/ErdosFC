<h1 align="center">Erdos-FAI</h1>

<p align="center">
  <em>Enterprise Agentic Technology — the AI platform your engineers wish they had time to build.</em><br>
  Composable agent cards · multi-agent orchestration · human-in-the-loop safety · continual learning · real-time observability.
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

Erdos is an **agent lifecycle platform**. You build agents, test them, deploy them into pipelines, let them improve from their own runs, and govern every step — from a single, observable, audited system. It is organized as **five tightly-integrated layers**:

| # | Layer | What it gives you | Package |
|---|-------|-------------------|---------|
| 01 | **Intelligence** | Composable *agent cards* — prompt + skills + memory + tools + MCP servers wired into one reusable, versioned object. | `erdos_fai.intelligence` |
| 02 | **Orchestration** | Multi-agent *pipelines* — chain agents into steps that pass artifacts forward, with human-in-the-loop gates between them. | `erdos_fai.orchestration` |
| 03 | **Safety** | PHI/PII redaction, a policy engine, and **write gates** that block side-effecting actions behind multi-approver human sign-off. | `erdos_fai.safety` |
| 04 | **Learning** | Trajectory recording, agent **evaluation** (role adherence, task completion, faithfulness, safety…), and an optimizer that writes *refined memories* back to agents. | `erdos_fai.learning` |
| 05 | **Insights** | Real-time **cost & ROI**, event-replay **tracing**, and a tamper-evident (hash-chained) **audit trail**. | `erdos_fai.insights` |

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

A 3-agent **healthcare-triage** pipeline (Intake → Triage → Referral) that exercises all five layers — PHI redaction, a HITL gate before the referral is "written to the EHR", cost/ROI, an audit trail, and post-run evaluation:

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

## The five layers, in code

```python
from erdos_fai import (
    AgentCard, Skill, Tool,                 # 01 Intelligence
    Pipeline, Step,                         # 02 Orchestration
    Policy, PolicyEngine, WriteGate,        # 03 Safety
    Evaluator, Optimizer,                   # 04 Learning
    CostTracker, Tracer, AuditLog,          # 05 Insights
)
```

- **Safety** — `Policy(redact_phi=True, require_approval_for_writes=True)` drives a `PolicyEngine` that scrubs PHI/PII on every input and output, and a `WriteGate` that requires an `Approver` (auto / console / your callback) before any `Tool(..., write=True)` runs.
- **Learning** — after a run, `Evaluator().evaluate(context.trajectory)` scores six agent-quality metrics; `Optimizer().refine(...)` turns low scores into refined memories you persist back onto the agent card.
- **Insights** — `build_dashboard(context, human_minutes=...)` rolls cost, latency, safety interventions, approvals, ROI, and audit-chain status into one snapshot.

## Project layout

```
erdos-fai/
  erdos_fai/
    intelligence/   01  agent cards, skills, memory, tools, MCP
    orchestration/  02  steps + pipelines
    safety/         03  redaction, policy, approval, write gates
    learning/       04  trajectories, evaluation, optimizer
    insights/       05  tracing, cost, audit, metrics
    runtime/            agent engine, run context, LLM providers
    cli.py              `erdos-fai` command
  examples/
    healthcare_triage/  runnable multi-agent demo
  tests/                pytest suite (offline, deterministic)
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
