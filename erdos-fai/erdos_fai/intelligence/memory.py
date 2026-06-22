"""Agent memory — Intelligence Layer.

A lightweight note store an agent can read into its prompt and write back to.
The Learning Layer feeds "refined memories" here so agents improve release
over release. Optionally mirrored to a JSON file for cross-session persistence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryEntry:
    content: str
    tag: str = "note"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "tag": self.tag, "created_at": self.created_at}


class Memory:
    """In-process note store with optional file backing."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[MemoryEntry] = []
        if self.path and self.path.exists():
            for row in json.loads(self.path.read_text()):
                self.entries.append(MemoryEntry(**row))

    def remember(self, content: str, tag: str = "note") -> MemoryEntry:
        entry = MemoryEntry(content=content.strip(), tag=tag)
        self.entries.append(entry)
        self._flush()
        return entry

    def recall(self, tag: str | None = None) -> list[MemoryEntry]:
        if tag is None:
            return list(self.entries)
        return [e for e in self.entries if e.tag == tag]

    def as_context(self, limit: int = 10) -> str:
        recent = self.entries[-limit:]
        if not recent:
            return ""
        return "\n".join(f"- {e.content}" for e in recent)

    def _flush(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([e.to_dict() for e in self.entries], indent=2))
