"""Event-replay tracing — Insights Layer.

A tiny, dependency-free span tracer. Every meaningful action in a run
(agent call, tool execution, approval request, redaction) opens a span;
spans nest, carry attributes, and are exported as a flat, replayable
event log. This is the substrate the Insights dashboards read from.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    """A single timed, attributed unit of work."""

    name: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    status: str = "ok"  # "ok" | "error"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return round((self.end - self.start) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": dict(self.attributes),
        }


class Tracer:
    """Collects nested spans for a single run and exports them for replay."""

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            name=name,
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._stack[-1] if self._stack else None,
            start=time.time(),
            attributes=dict(attributes),
        )
        self._spans.append(span)
        self._stack.append(span.span_id)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            span.status = "error"
            span.attributes["error"] = repr(exc)
            raise
        finally:
            span.end = time.time()
            self._stack.pop()

    def event(self, name: str, **attributes: Any) -> None:
        """Record a zero-duration marker span."""
        with self.span(name, **attributes):
            pass

    def export(self) -> list[dict[str, Any]]:
        """Flat, ordered event log — feed this to a UI for event replay."""
        return [s.to_dict() for s in self._spans]

    def total_ms(self) -> float:
        roots = [s for s in self._spans if s.parent_id is None]
        return round(sum(s.duration_ms for s in roots), 2)
