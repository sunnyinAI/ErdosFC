"""Skills — Intelligence Layer.

A skill is a named, reusable bundle of instructions that gets folded into an
agent's system prompt only when the agent carries it. Skills turn a
general-purpose agent into a specialist without rewriting its base prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    instructions: str
    description: str = ""

    def render(self) -> str:
        header = f"## Skill: {self.name}"
        if self.description:
            header += f"\n_{self.description}_"
        return f"{header}\n{self.instructions.strip()}"
