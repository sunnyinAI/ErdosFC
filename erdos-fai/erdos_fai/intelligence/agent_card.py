"""Composable agent cards — the heart of the Intelligence Layer.

An ``AgentCard`` wires a persona (system prompt), skills, a memory, a tool
library, and MCP servers into one versioned, reusable object. ``render_system``
composes them into a single system prompt for the model. This is the unit the
Orchestration Layer references and the Learning Layer upgrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import Memory
from .mcp import MCPServer
from .skill import Skill
from .tools import Tool, ToolLibrary

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class AgentCard:
    name: str
    system: str = ""
    model: str = DEFAULT_MODEL
    description: str = ""
    skills: list[Skill] = field(default_factory=list)
    tools: ToolLibrary = field(default_factory=ToolLibrary)
    memory: Memory = field(default_factory=Memory)
    mcp_servers: list[MCPServer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Convenience: accept a plain list of Tools for `tools`.
        if isinstance(self.tools, list):
            self.tools = ToolLibrary(self.tools)

    def add_skill(self, skill: Skill) -> "AgentCard":
        self.skills.append(skill)
        return self

    def add_tool(self, tool: Tool) -> "AgentCard":
        self.tools.register(tool)
        return self

    def render_system(self) -> str:
        """Compose persona + skills + tool catalog + MCP + memory into one prompt."""
        parts: list[str] = []
        if self.system:
            parts.append(self.system.strip())

        for skill in self.skills:
            parts.append(skill.render())

        catalog = self.tools.catalog()
        if catalog:
            parts.append("## Available tools\n" + catalog)

        if self.mcp_servers:
            parts.append(
                "## Connected MCP servers\n"
                + "\n".join(s.card() for s in self.mcp_servers)
            )

        mem = self.memory.as_context()
        if mem:
            parts.append("## Memory (learned context)\n" + mem)

        return "\n\n".join(parts)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "skills": [s.name for s in self.skills],
            "tools": [t.name for t in self.tools],
            "mcp_servers": [s.name for s in self.mcp_servers],
            "memory_entries": len(self.memory.entries),
        }
