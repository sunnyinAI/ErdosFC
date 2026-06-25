"""Risk & trend scoring — Execution Layer.

Turns connected event data into a single, explainable risk score against a
configured threshold — the "risk evaluation" step that decides whether a
workflow escalates. Each :class:`Signal` contributes a normalized, weighted
amount; :class:`RiskModel` rolls them into a :class:`RiskScore` with a band
(low / medium / high / critical), a breach flag, and the per-signal
contributors so the decision is auditable rather than a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Score bands, low → high.
BANDS = ("low", "medium", "high", "critical")
_BAND_CUTOFFS = ((0.35, "low"), (0.6, "medium"), (0.85, "high"), (1.01, "critical"))


@dataclass
class Signal:
    """One observed input to a risk score.

    ``value`` is normalized to 0..1 against ``[floor, ceiling]`` before being
    weighted, so heterogeneous inputs (heart rate, lab deltas, wait minutes)
    combine on a common scale.
    """

    name: str
    value: float
    weight: float = 1.0
    floor: float = 0.0
    ceiling: float = 1.0

    def normalized(self) -> float:
        # ``ceiling`` may be below ``floor`` for inverted signals where a lower
        # raw value means higher risk (e.g. SpO2: floor=100, ceiling=85).
        span = self.ceiling - self.floor
        if span == 0:
            return max(0.0, min(1.0, self.value))
        return max(0.0, min(1.0, (self.value - self.floor) / span))


@dataclass
class RiskScore:
    score: float                       # 0..1
    band: str
    breached: bool
    threshold: float
    contributors: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        top = sorted(self.contributors.items(), key=lambda kv: kv[1], reverse=True)
        drivers = ", ".join(f"{k} {v:.2f}" for k, v in top[:3])
        mark = "⚠ breach" if self.breached else "ok"
        return f"risk {self.score:.2f} [{self.band}] {mark} (≥{self.threshold:.2f}) — {drivers}"


def band_for(score: float) -> str:
    for cutoff, name in _BAND_CUTOFFS:
        if score < cutoff:
            return name
    return "critical"


class RiskModel:
    """A weighted, threshold-based scorer over named signals."""

    def __init__(self, *, threshold: float = 0.6, weights: dict[str, float] | None = None) -> None:
        self.threshold = threshold
        self.weights = weights or {}

    def evaluate(self, signals: list[Signal]) -> RiskScore:
        if not signals:
            return RiskScore(score=0.0, band="low", breached=False, threshold=self.threshold)

        contributors: dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        for s in signals:
            weight = self.weights.get(s.name, s.weight)
            contribution = s.normalized() * weight
            contributors[s.name] = round(contribution, 4)
            weighted_sum += contribution
            total_weight += weight

        score = round(weighted_sum / total_weight, 4) if total_weight else 0.0
        return RiskScore(
            score=score,
            band=band_for(score),
            breached=score >= self.threshold,
            threshold=self.threshold,
            contributors=contributors,
        )
