"""Execution Layer — connect to real systems and drive governed workflows.

Where the first five layers *build, test, and govern* agents, the Execution
Layer turns governed pipelines into real-world follow-through:

  • Connectors   pre-built integrations into EHRs, labs, ERPs, messaging…
  • Templates    healthcare- and operations-native workflow patterns
  • Risk         trend/threshold scoring that decides when to escalate
  • Routing      tasks, SLA timers, escalation ladders, notifications
  • RBAC         role-based access, including who may approve which risk
  • Simulation   dry-run a workflow before it touches anything live
"""

from .connectors import CATEGORIES, CONNECTORS, Connector, ConnectorRegistry
from .rbac import (
    ADMINISTRATOR,
    CASE_MANAGER,
    CLINICIAN,
    NURSE,
    OPERATOR,
    PERMISSIONS,
    ROLES,
    VIEWER,
    AccessControl,
    AccessDenied,
    RBACApprover,
    Role,
    User,
)
from .risk import BANDS, RiskModel, RiskScore, Signal, band_for
from .routing import (
    CHANNELS,
    EscalationLevel,
    EscalationPolicy,
    Notification,
    SLA,
    Task,
    TaskRouter,
)
from .simulation import PlannedAction, SimulationReport, Simulator
from .templates import (
    CLINICAL,
    OPERATIONAL,
    TEMPLATES,
    TemplateAction,
    TemplateLibrary,
    WorkflowTemplate,
)

__all__ = [
    # Connectors
    "CATEGORIES",
    "CONNECTORS",
    "Connector",
    "ConnectorRegistry",
    # Risk
    "BANDS",
    "RiskModel",
    "RiskScore",
    "Signal",
    "band_for",
    # Routing
    "CHANNELS",
    "EscalationLevel",
    "EscalationPolicy",
    "Notification",
    "SLA",
    "Task",
    "TaskRouter",
    # RBAC
    "ADMINISTRATOR",
    "CASE_MANAGER",
    "CLINICIAN",
    "NURSE",
    "OPERATOR",
    "PERMISSIONS",
    "ROLES",
    "VIEWER",
    "AccessControl",
    "AccessDenied",
    "RBACApprover",
    "Role",
    "User",
    # Templates
    "CLINICAL",
    "OPERATIONAL",
    "TEMPLATES",
    "TemplateAction",
    "TemplateLibrary",
    "WorkflowTemplate",
    # Simulation
    "PlannedAction",
    "SimulationReport",
    "Simulator",
]
