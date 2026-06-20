"""Tools & tool library — Intelligence Layer.

A tool is a typed, callable capability an agent can invoke. The ``write``
flag marks side-effecting tools (send, persist, delete) so the Safety Layer
can gate them behind human approval — see ``erdos_platform.safety``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    write: bool = False  # True => side-effecting; gated by the Safety Layer

    def __call__(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)

    def card(self) -> str:
        flag = " (write)" if self.write else ""
        return f"- `{self.name}`{flag}: {self.description}"


class ToolLibrary:
    """A small registry of tools, addressable by name."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def catalog(self) -> str:
        if not self._tools:
            return ""
        return "\n".join(t.card() for t in self._tools.values())
