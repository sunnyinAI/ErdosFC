"""Role-based access control — Execution Layer.

Granular, declarative permissions over who may do what: run workflows, manage
connectors, and — critically — approve gated writes at a given risk level. The
:class:`RBACApprover` plugs straight into the Safety Layer's :class:`Approver`
contract, so the human-in-the-loop gate only clears when the acting user
actually holds the permission for that risk band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..safety.approval import ApprovalDecision, ApprovalRequest, Approver

# Permission identifiers.
WORKFLOW_VIEW = "workflow.view"
WORKFLOW_RUN = "workflow.run"
CONNECTOR_MANAGE = "connector.manage"
TASK_ASSIGN = "task.assign"
APPROVE_LOW = "approve.low"
APPROVE_MEDIUM = "approve.medium"
APPROVE_HIGH = "approve.high"
ADMIN = "admin"  # wildcard — grants everything

PERMISSIONS = (
    WORKFLOW_VIEW, WORKFLOW_RUN, CONNECTOR_MANAGE, TASK_ASSIGN,
    APPROVE_LOW, APPROVE_MEDIUM, APPROVE_HIGH, ADMIN,
)

# Risk level → the permission required to approve it.
_APPROVE_FOR_RISK = {"low": APPROVE_LOW, "medium": APPROVE_MEDIUM, "high": APPROVE_HIGH}


class AccessDenied(Exception):
    """Raised when a user lacks a required permission."""


@dataclass(frozen=True)
class Role:
    name: str
    permissions: frozenset[str] = frozenset()

    def grants(self, permission: str) -> bool:
        return ADMIN in self.permissions or permission in self.permissions


@dataclass
class User:
    id: str
    name: str
    roles: list[Role] = field(default_factory=list)

    @property
    def permissions(self) -> set[str]:
        out: set[str] = set()
        for role in self.roles:
            out |= set(role.permissions)
        return out

    def can(self, permission: str) -> bool:
        return any(r.grants(permission) for r in self.roles)


# Predefined roles, least → most privileged.
VIEWER = Role("viewer", frozenset({WORKFLOW_VIEW}))
NURSE = Role("nurse", frozenset({WORKFLOW_VIEW, WORKFLOW_RUN, TASK_ASSIGN, APPROVE_LOW}))
CLINICIAN = Role("clinician", frozenset({WORKFLOW_VIEW, WORKFLOW_RUN, TASK_ASSIGN, APPROVE_LOW, APPROVE_MEDIUM, APPROVE_HIGH}))
CASE_MANAGER = Role("case_manager", frozenset({WORKFLOW_VIEW, WORKFLOW_RUN, TASK_ASSIGN, APPROVE_LOW, APPROVE_MEDIUM}))
OPERATOR = Role("operator", frozenset({WORKFLOW_VIEW, WORKFLOW_RUN, CONNECTOR_MANAGE, TASK_ASSIGN, APPROVE_LOW, APPROVE_MEDIUM}))
ADMINISTRATOR = Role("administrator", frozenset({ADMIN}))

ROLES = (VIEWER, NURSE, CLINICIAN, CASE_MANAGER, OPERATOR, ADMINISTRATOR)


class AccessControl:
    """Checks a user against required permissions."""

    def can(self, user: User, permission: str) -> bool:
        return user.can(permission)

    def require(self, user: User, permission: str) -> None:
        if not user.can(permission):
            raise AccessDenied(f"{user.name!r} lacks permission {permission!r}")

    def can_approve(self, user: User, risk: str) -> bool:
        return user.can(_APPROVE_FOR_RISK.get(risk, APPROVE_HIGH))


class RBACApprover(Approver):
    """An :class:`Approver` that clears only if the user may approve that risk."""

    def __init__(self, user: User, access: AccessControl | None = None) -> None:
        self.user = user
        self.access = access or AccessControl()

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        if self.access.can_approve(self.user, req.risk):
            return ApprovalDecision(
                approved=True, approver=self.user.name,
                reason=f"authorized for {req.risk}-risk approval",
            )
        return ApprovalDecision(
            approved=False, approver=self.user.name,
            reason=f"not authorized to approve {req.risk}-risk actions",
        )
