"""Write gates — Safety Layer.

Wraps a side-effecting Tool so it cannot run until policy + a human approver
allow it. Every decision (allowed, denied, auto) is written to the audit log,
giving you the multi-approver, admin-configurable write-gate behavior the
governance suite promises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..intelligence.tools import Tool
from .approval import Approver, ApprovalRequest
from .policy import PolicyEngine


class GateDenied(Exception):
    """Raised when a gated action is denied by policy or an approver."""


@dataclass
class GateOutcome:
    allowed: bool
    approver: str
    reason: str
    result: Any = None


class WriteGate:
    """Guards execution of write tools behind policy + approval."""

    def __init__(self, policy: PolicyEngine, approver: Approver, audit=None) -> None:
        self.policy = policy
        self.approver = approver
        self.audit = audit

    def guard(self, tool: Tool, *, risk: str = "medium", **kwargs: Any) -> GateOutcome:
        if not self.policy.needs_approval(is_write=tool.write):
            result = tool(**kwargs)
            if self.audit:
                self.audit.record("tool.execute", actor="agent", tool=tool.name, gated=False)
            return GateOutcome(allowed=True, approver="none", reason="not gated", result=result)

        decision = self.approver.request(
            ApprovalRequest(
                action=f"execute write tool `{tool.name}`",
                detail=", ".join(f"{k}={v!r}" for k, v in kwargs.items()) or "(no args)",
                risk=risk,
            )
        )
        if self.audit:
            self.audit.record(
                "approval.decision",
                actor=decision.approver,
                tool=tool.name,
                approved=decision.approved,
                reason=decision.reason,
            )

        if not decision.approved:
            raise GateDenied(f"`{tool.name}` denied by {decision.approver}: {decision.reason}")

        result = tool(**kwargs)
        if self.audit:
            self.audit.record("tool.execute", actor="agent", tool=tool.name, gated=True)
        return GateOutcome(
            allowed=True,
            approver=decision.approver,
            reason=decision.reason,
            result=result,
        )
