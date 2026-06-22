"""Human-in-the-loop approval — Safety Layer.

Approvers decide whether a gated action may proceed. Swap implementations
per environment: ``AutoApprover`` for tests/CI, ``ConsoleApprover`` for an
interactive run, ``CallbackApprover`` to bridge into your own UI or a
multi-approver workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class ApprovalRequest:
    action: str
    detail: str
    risk: str = "medium"  # "low" | "medium" | "high"


@dataclass
class ApprovalDecision:
    approved: bool
    approver: str
    reason: str = ""


class Approver(ABC):
    @abstractmethod
    def request(self, req: ApprovalRequest) -> ApprovalDecision: ...


class AutoApprover(Approver):
    """Approves (or denies) without a human — for tests and demos."""

    def __init__(self, approve: bool = True, name: str = "auto") -> None:
        self.approve = approve
        self.name = name

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            approved=self.approve,
            approver=self.name,
            reason="auto-approved" if self.approve else "auto-denied",
        )


class CallbackApprover(Approver):
    """Delegates the decision to a callable (e.g. your app's approval UI)."""

    def __init__(self, fn: Callable[[ApprovalRequest], bool], name: str = "callback") -> None:
        self.fn = fn
        self.name = name

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        approved = bool(self.fn(req))
        return ApprovalDecision(approved=approved, approver=self.name)


class ConsoleApprover(Approver):
    """Prompts a human on stdin. Blocks until they answer y/n."""

    def __init__(self, name: str = "console") -> None:
        self.name = name

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        print(f"\n⚠️  Approval required [{req.risk} risk]: {req.action}")
        print(f"    {req.detail}")
        answer = input("    Approve? [y/N] ").strip().lower()
        approved = answer in {"y", "yes"}
        return ApprovalDecision(
            approved=approved,
            approver=self.name,
            reason="approved by operator" if approved else "denied by operator",
        )
