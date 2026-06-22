"""Policy enforcement — Safety Layer.

A ``Policy`` is the declarative rule set (redact PHI/PII, block terms, gate
writes); ``PolicyEngine`` applies it to text flowing in and out of agents and
decides whether side-effecting actions need approval. Mirrors the
"policies / HIPAA · v2.4" enforce-list model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .redaction import Finding, Redactor


@dataclass
class Policy:
    name: str = "default"
    version: str = "v1"
    redact_pii: bool = True
    redact_phi: bool = True
    blocked_terms: list[str] = field(default_factory=list)
    require_approval_for_writes: bool = True

    def enforced(self) -> list[str]:
        rules = []
        if self.redact_phi:
            rules.append("PHI redaction")
        if self.redact_pii:
            rules.append("PII detection")
        if self.blocked_terms:
            rules.append("content blocklist")
        if self.require_approval_for_writes:
            rules.append("write gating")
        rules.append("audit trail")
        return rules


@dataclass
class EnforcementResult:
    text: str
    findings: list[Finding]
    blocked: bool = False
    blocked_reason: str = ""


class PolicyEngine:
    """Applies a Policy to text and actions."""

    def __init__(self, policy: Policy | None = None, redactor: Redactor | None = None) -> None:
        self.policy = policy or Policy()
        self.redactor = redactor or Redactor()

    def enforce_text(self, text: str) -> EnforcementResult:
        findings: list[Finding] = []
        out = text

        if self.policy.redact_pii or self.policy.redact_phi:
            out, findings = self.redactor.redact(out)

        for term in self.policy.blocked_terms:
            if term.lower() in out.lower():
                return EnforcementResult(
                    text=out,
                    findings=findings,
                    blocked=True,
                    blocked_reason=f"blocked term: {term!r}",
                )

        return EnforcementResult(text=out, findings=findings)

    def needs_approval(self, *, is_write: bool) -> bool:
        return is_write and self.policy.require_approval_for_writes
