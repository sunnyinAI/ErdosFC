"""Task routing, SLA & escalation — Execution Layer.

The "follow-through" half of the platform: once a workflow decides something
needs human action, the :class:`TaskRouter` creates a review :class:`Task` with
an owner, a due time, and suggested next steps, notifies the owner over a
channel, starts an SLA timer, and escalates up an :class:`EscalationPolicy` if
acknowledgement is late. Every step is written to the audit log, and each task
exposes the output variables an external system expects back — ``action_id``,
``assigned_owner``, ``escalation_status``, ``audit_event_id``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# Notification channels — backed by Messaging connectors at runtime.
IN_APP = "in_app"
SMS = "sms"
WHATSAPP = "whatsapp"
EMAIL = "email"
SLACK = "slack"
CHANNELS = (IN_APP, SMS, WHATSAPP, EMAIL, SLACK)

# Task lifecycle.
OPEN = "open"
ACKNOWLEDGED = "acknowledged"
ESCALATED = "escalated"
RESOLVED = "resolved"


@dataclass
class SLA:
    """Service-level targets, in minutes."""

    ack_minutes: float = 15.0
    resolve_minutes: float = 120.0


@dataclass
class EscalationLevel:
    after_minutes: float
    role: str
    channel: str = IN_APP


@dataclass
class EscalationPolicy:
    """An ordered ladder of who to escalate to, and when."""

    levels: list[EscalationLevel] = field(default_factory=list)

    def due_levels(self, elapsed_minutes: float) -> list[EscalationLevel]:
        return [lvl for lvl in self.levels if elapsed_minutes >= lvl.after_minutes]


@dataclass
class Notification:
    channel: str
    to: str
    message: str


@dataclass
class Task:
    """A review task routed to a human, with full output-variable surface."""

    title: str
    assigned_owner: str
    action_id: str = field(default_factory=lambda: "act_" + uuid.uuid4().hex[:10])
    status: str = OPEN
    due_minutes: float = 120.0
    risk: str = "medium"
    source: str = ""
    next_steps: list[str] = field(default_factory=list)
    escalation_status: str = "none"
    audit_event_id: str | None = None
    notifications: list[Notification] = field(default_factory=list)

    def outputs(self) -> dict[str, object]:
        """The variables an external workflow reads back after routing."""
        return {
            "action_id": self.action_id,
            "assigned_owner": self.assigned_owner,
            "status": self.status,
            "escalation_status": self.escalation_status,
            "audit_event_id": self.audit_event_id,
        }


class TaskRouter:
    """Creates, notifies, escalates, and resolves review tasks."""

    def __init__(self, audit=None, default_channel: str = IN_APP) -> None:
        self.audit = audit
        self.default_channel = default_channel
        self.tasks: list[Task] = []

    def create_task(
        self,
        *,
        title: str,
        owner: str,
        sla: SLA | None = None,
        risk: str = "medium",
        source: str = "",
        next_steps: list[str] | None = None,
        channel: str | None = None,
    ) -> Task:
        sla = sla or SLA()
        task = Task(
            title=title,
            assigned_owner=owner,
            due_minutes=sla.resolve_minutes,
            risk=risk,
            source=source,
            next_steps=list(next_steps or []),
        )
        rec = self._audit("task.create", actor="router", action_id=task.action_id,
                          owner=owner, title=title, risk=risk)
        task.audit_event_id = rec
        self.notify(task, channel or self.default_channel,
                    f"New {risk}-risk task: {title}. Ack within {sla.ack_minutes:.0f} min.")
        self.tasks.append(task)
        return task

    def notify(self, task: Task, channel: str, message: str) -> Notification:
        note = Notification(channel=channel, to=task.assigned_owner, message=message)
        task.notifications.append(note)
        self._audit("notify.send", actor="router", action_id=task.action_id,
                    channel=channel, to=task.assigned_owner)
        return note

    def acknowledge(self, task: Task, by: str) -> Task:
        task.status = ACKNOWLEDGED
        self._audit("task.ack", actor=by, action_id=task.action_id)
        return task

    def resolve(self, task: Task, by: str) -> Task:
        task.status = RESOLVED
        self._audit("task.resolve", actor=by, action_id=task.action_id)
        return task

    def escalate(self, task: Task, policy: EscalationPolicy, elapsed_minutes: float) -> list[EscalationLevel]:
        """Escalate a still-unacknowledged task per the policy ladder."""
        if task.status in (ACKNOWLEDGED, RESOLVED):
            return []
        fired = policy.due_levels(elapsed_minutes)
        for lvl in fired:
            task.status = ESCALATED
            task.escalation_status = f"L{policy.levels.index(lvl) + 1}:{lvl.role}"
            self.notify(task, lvl.channel,
                        f"ESCALATION → {lvl.role}: '{task.title}' unacknowledged after {lvl.after_minutes:.0f} min.")
            self._audit("task.escalate", actor="router", action_id=task.action_id,
                        to_role=lvl.role, after_minutes=lvl.after_minutes)
        return fired

    def _audit(self, event: str, **detail) -> str | None:
        if not self.audit:
            return None
        rec = self.audit.record(event, **detail)
        return getattr(rec, "hash", None)
