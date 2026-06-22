"""Real-time cost & ROI tracking — Insights Layer.

Converts token usage into dollars using a per-model price table, and rolls
the result up by agent/step so the dashboard can show total cost, cost
trend, and dollars/hours saved.

Prices are USD per 1M tokens and reflect the Claude model line-up. They are
intentionally editable — override ``PRICING`` for your own contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1,000,000 tokens: model -> (input, output)
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Offline development provider — free.
    "echo": (0.0, 0.0),
}

# What a comparable unit of human work costs, used for ROI math.
DEFAULT_HUMAN_RATE_PER_HOUR = 75.0


def price_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of a single call. Unknown models fall back to Opus pricing."""
    in_rate, out_rate = PRICING.get(model, PRICING["claude-opus-4-8"])
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates per-call cost and exposes dashboard-ready rollups."""

    human_rate_per_hour: float = DEFAULT_HUMAN_RATE_PER_HOUR
    entries: list[CostEntry] = field(default_factory=list)

    def record(self, *, label: str, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
        entry = CostEntry(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=price_for(model, input_tokens, output_tokens),
        )
        self.entries.append(entry)
        return entry

    @property
    def total_cost(self) -> float:
        return round(sum(e.cost_usd for e in self.entries), 6)

    @property
    def total_tokens(self) -> int:
        return sum(e.input_tokens + e.output_tokens for e in self.entries)

    def by_label(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.label] = round(out.get(e.label, 0.0) + e.cost_usd, 6)
        return out

    def dollars_saved(self, human_minutes: float) -> float:
        """ROI: human cost of the equivalent manual work minus AI spend."""
        human_cost = (human_minutes / 60) * self.human_rate_per_hour
        return round(human_cost - self.total_cost, 2)

    def summary(self, human_minutes: float = 0.0) -> dict[str, object]:
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "calls": len(self.entries),
            "cost_by_step": self.by_label(),
            "dollars_saved": self.dollars_saved(human_minutes) if human_minutes else None,
        }
