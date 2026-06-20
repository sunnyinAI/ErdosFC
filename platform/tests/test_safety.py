"""Safety Layer: redaction, policy enforcement, and write gates."""

import pytest

from erdos_platform import AutoApprover, Policy, PolicyEngine, Tool, WriteGate
from erdos_platform.safety import GateDenied, Redactor


def test_redactor_finds_and_scrubs_phi():
    text = "MRN: 4821990, email john@example.com, phone 415-555-0199, SSN 123-45-6789"
    redacted, findings = Redactor().redact(text)
    labels = {f.label for f in findings}
    assert {"mrn", "email", "phone", "ssn"} <= labels
    assert "john@example.com" not in redacted
    assert "[REDACTED:SSN]" in redacted


def test_policy_blocks_terms():
    engine = PolicyEngine(Policy(blocked_terms=["wire transfer"]))
    result = engine.enforce_text("Please initiate a wire transfer now")
    assert result.blocked is True


def test_write_gate_denied_raises():
    engine = PolicyEngine(Policy(require_approval_for_writes=True))
    gate = WriteGate(engine, AutoApprover(approve=False))
    tool = Tool("delete_record", "Delete a record", handler=lambda **k: "deleted", write=True)
    with pytest.raises(GateDenied):
        gate.guard(tool, record_id=7)


def test_write_gate_approved_executes():
    engine = PolicyEngine(Policy(require_approval_for_writes=True))
    gate = WriteGate(engine, AutoApprover(approve=True))
    tool = Tool("save", "Save a record", handler=lambda **k: "saved", write=True)
    outcome = gate.guard(tool, value=1)
    assert outcome.allowed and outcome.result == "saved"


def test_read_tool_not_gated():
    engine = PolicyEngine(Policy(require_approval_for_writes=True))
    gate = WriteGate(engine, AutoApprover(approve=False))  # would deny if asked
    read_tool = Tool("lookup", "Look something up", handler=lambda **k: "value", write=False)
    outcome = gate.guard(read_tool)  # not a write → no approval needed
    assert outcome.result == "value"
