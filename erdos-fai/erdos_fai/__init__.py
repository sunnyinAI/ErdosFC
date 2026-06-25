"""Erdos-FAI — enterprise agentic technology in five layers.

Erdos is the agent lifecycle stack: composable agent cards (Intelligence),
multi-agent orchestration (Orchestration), human-in-the-loop safety (Safety),
trajectory-driven improvement (Learning), and real-time observability
(Insights). Build, test, deploy, reinforce, and govern agents from one runtime.

Quick start::

    from erdos_fai import AgentCard, Agent, Pipeline, Step, RunContext

    triage = Agent(AgentCard(name="Triage", system="You triage support tickets."))
    pipe = Pipeline("support", [Step("triage", triage, output_key="triage")])
    result = pipe.run("Customer can't log in")
    print(result.final_output)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Intelligence Layer
from .intelligence import (
    DEFAULT_MODEL,
    AgentCard,
    MCPServer,
    Memory,
    Skill,
    Tool,
    ToolLibrary,
)

# Orchestration Layer
from .orchestration import Pipeline, PipelineHalted, PipelineResult, Step

# Safety Layer
from .safety import (
    Approver,
    AutoApprover,
    CallbackApprover,
    ConsoleApprover,
    Policy,
    PolicyEngine,
    Redactor,
    WriteGate,
)

# Learning Layer
from .learning import Evaluator, Optimizer, Trajectory

# Insights Layer
from .insights import AuditLog, CostTracker, Dashboard, Tracer

# Execution Layer
from .execution import (
    CONNECTORS,
    TEMPLATES,
    AccessControl,
    Connector,
    ConnectorRegistry,
    EscalationPolicy,
    RBACApprover,
    RiskModel,
    Role,
    Signal,
    Simulator,
    SLA,
    Task,
    TaskRouter,
    TemplateLibrary,
    User,
    WorkflowTemplate,
)

# Runtime
from .runtime import (
    Agent,
    AnthropicProvider,
    EchoProvider,
    LLMProvider,
    RunContext,
    default_provider,
)


def build_dashboard(context: RunContext, *, human_minutes: float = 0.0) -> Dashboard:
    """Roll a finished run's context up into an Insights dashboard snapshot."""
    cost = context.cost
    return Dashboard(
        runs=1,
        total_cost_usd=cost.total_cost,
        total_tokens=cost.total_tokens,
        latency_ms=context.tracer.total_ms(),
        safety_interventions=context.safety_interventions,
        approvals=context.approvals,
        dollars_saved=cost.dollars_saved(human_minutes) if human_minutes else None,
        audit_ok=context.audit.verify(),
    )


__all__ = [
    "__version__",
    "build_dashboard",
    # Intelligence
    "AgentCard",
    "DEFAULT_MODEL",
    "MCPServer",
    "Memory",
    "Skill",
    "Tool",
    "ToolLibrary",
    # Orchestration
    "Pipeline",
    "PipelineHalted",
    "PipelineResult",
    "Step",
    # Safety
    "Approver",
    "AutoApprover",
    "CallbackApprover",
    "ConsoleApprover",
    "Policy",
    "PolicyEngine",
    "Redactor",
    "WriteGate",
    # Learning
    "Evaluator",
    "Optimizer",
    "Trajectory",
    # Insights
    "AuditLog",
    "CostTracker",
    "Dashboard",
    "Tracer",
    # Execution
    "CONNECTORS",
    "TEMPLATES",
    "AccessControl",
    "Connector",
    "ConnectorRegistry",
    "EscalationPolicy",
    "RBACApprover",
    "RiskModel",
    "Role",
    "Signal",
    "Simulator",
    "SLA",
    "Task",
    "TaskRouter",
    "TemplateLibrary",
    "User",
    "WorkflowTemplate",
    # Runtime
    "Agent",
    "AnthropicProvider",
    "EchoProvider",
    "LLMProvider",
    "RunContext",
    "default_provider",
]
