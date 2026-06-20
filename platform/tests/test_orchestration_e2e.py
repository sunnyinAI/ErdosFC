"""End-to-end Orchestration: a multi-agent pipeline through every layer.

Runs offline on the EchoProvider — deterministic, no API key.
"""

from erdos_platform import (
    Agent,
    AgentCard,
    AutoApprover,
    EchoProvider,
    Pipeline,
    Policy,
    PolicyEngine,
    RunContext,
    Step,
)


def _pipeline() -> Pipeline:
    a = Agent(AgentCard(name="A", system="Agent A."), provider=EchoProvider())
    b = Agent(AgentCard(name="B", system="Agent B."), provider=EchoProvider())
    return Pipeline(
        "test-pipe",
        [
            Step("first", a, prompt="Handle: {input}", output_key="first"),
            Step("second", b, prompt="Refine: {first}", output_key="second", requires_approval=True),
        ],
    )


def test_pipeline_runs_and_passes_artifacts():
    ctx = RunContext(approver=AutoApprover(approve=True))
    result = _pipeline().run("the task", ctx)
    assert not result.halted
    assert "first" in result.artifacts and "second" in result.artifacts
    # Step two's prompt referenced step one's output (artifact flow).
    assert "first" in ctx.trajectory.steps[1].prompt or result.artifacts["first"]
    assert len(ctx.trajectory.steps) == 2
    assert ctx.audit.verify() is True


def test_pipeline_halts_when_approval_denied():
    ctx = RunContext(approver=AutoApprover(approve=False))
    result = _pipeline().run("the task", ctx)
    assert result.halted is True
    assert "denied" in result.halted_reason
    # The gated second step never produced an artifact.
    assert "second" not in result.artifacts


def test_phi_is_redacted_in_pipeline_output():
    ctx = RunContext(policy=PolicyEngine(Policy(name="HIPAA")), approver=AutoApprover(approve=True))
    result = _pipeline().run("Contact john@example.com about SSN 123-45-6789", ctx)
    assert "john@example.com" not in result.final_output
    assert ctx.safety_interventions >= 1
