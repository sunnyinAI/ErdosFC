"""Agent engine — Runtime.

An ``Agent`` binds an ``AgentCard`` (Intelligence) to an ``LLMProvider`` and
executes one turn end-to-end through every layer: redact the input (Safety),
trace the call (Insights), invoke the model, record cost (Insights), redact
the output (Safety), and emit a trajectory step (Learning).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..intelligence.agent_card import AgentCard
from ..learning.trajectory import TrajectoryStep
from .context import RunContext
from .llm import LLMProvider, Message, default_provider


@dataclass
class AgentResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Agent:
    """A single agent: an agent card plus the provider that runs it."""

    def __init__(self, card: AgentCard, provider: LLMProvider | None = None) -> None:
        self.card = card
        self.provider = provider or default_provider()

    def run(self, task: str, context: RunContext, *, step_name: str | None = None) -> AgentResult:
        step_name = step_name or self.card.name
        with context.tracer.span("agent.run", agent=self.card.name, model=self.card.model):
            # 1. Safety: scrub sensitive data out of the input before it reaches the model.
            with context.tracer.span("safety.input"):
                enforced_in = context.policy.enforce_text(task)
            if enforced_in.findings:
                context.note_intervention(len(enforced_in.findings))
                context.audit.record(
                    "redaction.input",
                    actor="policy",
                    agent=self.card.name,
                    findings=[f.label for f in enforced_in.findings],
                )

            # 2. Runtime: call the model.
            system = self.card.render_system()
            with context.tracer.span("llm.complete", provider=self.provider.name):
                resp = self.provider.complete(
                    system=system,
                    messages=[Message(role="user", content=enforced_in.text)],
                    model=self.card.model,
                )

            # 3. Insights: record cost.
            context.cost.record(
                label=step_name,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )

            # 4. Safety: scrub the output too (defense in depth).
            with context.tracer.span("safety.output"):
                enforced_out = context.policy.enforce_text(resp.text)
            if enforced_out.findings:
                context.note_intervention(len(enforced_out.findings))
                context.audit.record(
                    "redaction.output", actor="policy", agent=self.card.name,
                    findings=[f.label for f in enforced_out.findings],
                )

            context.audit.record(
                "agent.run", actor=self.card.name,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            )

            # 5. Learning: append a trajectory step.
            if context.trajectory is not None:
                context.trajectory.add(
                    TrajectoryStep(
                        name=step_name,
                        agent=self.card.name,
                        prompt=enforced_in.text,
                        output=enforced_out.text,
                        model=resp.model,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                    )
                )

            return AgentResult(
                text=enforced_out.text,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
