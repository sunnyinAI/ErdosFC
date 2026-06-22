"""MCP server references — Intelligence Layer.

A declaration that an agent may use a Model Context Protocol server. Mirrors
the agent-card model: the card declares *which* servers (name, url); auth
lives elsewhere (a vault/secret store), never on the reusable card.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCPServer:
    name: str
    url: str
    description: str = ""

    def card(self) -> str:
        suffix = f" — {self.description}" if self.description else ""
        return f"- {self.name} ({self.url}){suffix}"
