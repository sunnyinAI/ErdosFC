"""Insights Layer: cost pricing, tamper-evident audit, and tracing."""

from erdos_fai import AuditLog, CostTracker, Tracer
from erdos_fai.insights import price_for


def test_cost_pricing_matches_table():
    # Opus 4.8: $5 / $25 per 1M tokens.
    cost = price_for("claude-opus-4-8", 1_000_000, 1_000_000)
    assert round(cost, 2) == 30.0


def test_cost_tracker_rollup_and_roi():
    tracker = CostTracker(human_rate_per_hour=60.0)
    tracker.record(label="step1", model="claude-opus-4-8", input_tokens=1000, output_tokens=500)
    tracker.record(label="step2", model="claude-haiku-4-5", input_tokens=1000, output_tokens=500)
    assert tracker.total_tokens == 3000
    assert "step1" in tracker.by_label()
    # 30 minutes of human work at $60/hr = $30, minus tiny AI spend.
    assert tracker.dollars_saved(30) > 29


def test_audit_chain_verifies_and_detects_tampering():
    log = AuditLog()
    log.record("start", actor="orchestrator")
    log.record("agent.run", actor="Intake")
    assert log.verify() is True

    # Tamper with a committed record.
    log.records[0].detail["injected"] = "evil"
    assert log.verify() is False


def test_tracer_nesting_and_export():
    tracer = Tracer()
    with tracer.span("outer"):
        with tracer.span("inner"):
            pass
    events = tracer.export()
    names = [e["name"] for e in events]
    assert names == ["outer", "inner"]
    inner = next(e for e in events if e["name"] == "inner")
    outer = next(e for e in events if e["name"] == "outer")
    assert inner["parent_id"] == outer["span_id"]
