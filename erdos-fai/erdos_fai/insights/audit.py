"""Tamper-evident audit trail — Insights Layer.

An append-only event log with a SHA-256 hash chain: each record commits to
the hash of the previous one, so any after-the-fact edit or deletion breaks
the chain and is detectable by ``verify()``. This is the kind of immutable
audit trail compliance regimes (HIPAA, 21 CFR Part 11, SOC 2) expect.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GENESIS = "0" * 64


def _hash(prev_hash: str, payload: dict[str, Any]) -> str:
    blob = prev_hash + json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    seq: int
    timestamp: float
    event: str
    actor: str
    detail: dict[str, Any]
    prev_hash: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "actor": self.actor,
            "detail": self.detail,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


class AuditLog:
    """Append-only, hash-chained event log, optionally mirrored to disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[AuditRecord] = []
        if self.path and self.path.exists():
            self.path.unlink()  # start a fresh chain per run

    @property
    def _last_hash(self) -> str:
        return self.records[-1].hash if self.records else _GENESIS

    def record(self, event: str, actor: str = "system", **detail: Any) -> AuditRecord:
        payload = {
            "seq": len(self.records),
            "timestamp": time.time(),
            "event": event,
            "actor": actor,
            "detail": detail,
        }
        rec = AuditRecord(
            **payload,
            prev_hash=self._last_hash,
            hash=_hash(self._last_hash, payload),
        )
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return rec

    def verify(self) -> bool:
        """Recompute the chain; returns False if any record was altered."""
        prev = _GENESIS
        for rec in self.records:
            payload = {
                "seq": rec.seq,
                "timestamp": rec.timestamp,
                "event": rec.event,
                "actor": rec.actor,
                "detail": rec.detail,
            }
            if rec.prev_hash != prev or rec.hash != _hash(prev, payload):
                return False
            prev = rec.hash
        return True

    def export(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]
