"""Pipeline steps — Orchestration Layer.

A ``Step`` binds an agent to a prompt template and optional approval gate.
The prompt template can interpolate ``{input}`` (the pipeline task) and any
prior step's output by its ``output_key`` (e.g. ``{intake}``), which is how
data flows from one agent to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..runtime.agent import Agent


@dataclass
class Step:
    name: str
    agent: Agent
    prompt: str = "{input}"
    output_key: str | None = None
    requires_approval: bool = False
    approval_risk: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, task: str, artifacts: dict[str, Any]) -> str:
        return self.prompt.format(input=task, **artifacts)
