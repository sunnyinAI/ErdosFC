"""Dashboard rollups — Insights Layer.

Assembles the headline numbers a cost/observability dashboard shows: spend,
latency, tokens, safety interventions, and ROI — from the tracer, cost
tracker, and audit log of a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Dashboard:
    """A point-in-time snapshot of a run, ready to render."""

    runs: int
    total_cost_usd: float
    total_tokens: int
    latency_ms: float
    safety_interventions: int
    approvals: int
    dollars_saved: float | None
    audit_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "safety_interventions": self.safety_interventions,
            "approvals": self.approvals,
            "dollars_saved": self.dollars_saved,
            "audit_ok": self.audit_ok,
        }

    def render(self) -> str:
        lines = [
            "┌─ Erdos Insights ─────────────────────────────",
            f"│ runs                 {self.runs}",
            f"│ total cost           ${self.total_cost_usd:.4f}",
            f"│ tokens               {self.total_tokens:,}",
            f"│ latency              {self.latency_ms:.0f} ms",
            f"│ safety interventions {self.safety_interventions}",
            f"│ approvals            {self.approvals}",
        ]
        if self.dollars_saved is not None:
            lines.append(f"│ dollars saved        ${self.dollars_saved:.2f}")
        lines.append(f"│ audit chain          {'verified ✓' if self.audit_ok else 'BROKEN ✗'}")
        lines.append("└──────────────────────────────────────────────")
        return "\n".join(lines)
