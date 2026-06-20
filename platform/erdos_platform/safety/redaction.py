"""PHI / PII redaction — Safety Layer.

Regex detectors for the most common identifiers (SSN, MRN, email, phone,
DOB, credit card). ``detect`` returns findings; ``redact`` returns a scrubbed
copy plus findings. Healthcare-aware by default (MRN, DOB) but generic enough
for any PII regime. Extend ``DEFAULT_PATTERNS`` for your own entity types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (label, compiled pattern, redaction token)
DEFAULT_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:SSN]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED:EMAIL]"),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED:PHONE]"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED:CC]"),
    ("mrn", re.compile(r"\bMRN[:#]?\s?\d{5,10}\b", re.IGNORECASE), "[REDACTED:MRN]"),
    ("dob", re.compile(r"\b(?:DOB[:\s]*)?\d{1,2}/\d{1,2}/\d{4}\b", re.IGNORECASE), "[REDACTED:DOB]"),
]


@dataclass
class Finding:
    label: str
    match: str
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "match": self.match, "start": self.start, "end": self.end}


class Redactor:
    """Detects and removes sensitive spans from text."""

    def __init__(self, patterns: list[tuple[str, re.Pattern[str], str]] | None = None) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS

    def detect(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for label, pattern, _ in self.patterns:
            for m in pattern.finditer(text):
                findings.append(Finding(label=label, match=m.group(0), start=m.start(), end=m.end()))
        return findings

    def redact(self, text: str) -> tuple[str, list[Finding]]:
        findings = self.detect(text)
        redacted = text
        for label, pattern, token in self.patterns:
            redacted = pattern.sub(token, redacted)
        return redacted, findings
