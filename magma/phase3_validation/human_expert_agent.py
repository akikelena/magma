"""Human Expert Agent.

Modality: human-in-the-loop; this agent realizes the human control
prescribed by trust, risk, and security management for agentic systems.

Input (pre): an escalated case with diagnostic context.
Output (post): an approval or rejection decision with an annotated
rationale.

Review is triggered under four conditions
(`magma.core.governance.escalation.EscalationPolicy`): low documentation-
interpretation confidence, validation failures exceeding the retry limit,
conflicting specifications, and security-critical changes. Because a real
human cannot be blocked on synchronously inside an automated pipeline,
this agent works as a queue: `process` enqueues a `ReviewTicket` and
returns immediately (unresolved); a human later calls `submit_decision`
(e.g. via `magma.scripts.review_queue`, the CLI dashboard the
Explainability Agent's escalations surface through), which resolves the
ticket. This agent is the terminal authority for escalated cases.

`ReviewTicketStore` is storage-agnostic like `GraphStore`/`VectorStore`:
the default `InMemoryTicketStore` matches this module's original
behavior (fine for library use and tests), but a CLI-driven process is
short-lived, and an escalation a human can never see again after the
process exits isn't actually human-in-the-loop. `JsonFileTicketStore`
is the real backend `run_workflow`/`run_doc_watch` use so a ticket
enqueued by one process invocation survives to be decided by another.
"""
from __future__ import annotations

import abc
import enum
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from magma.core.governance.audit_log import AuditLogger
from magma.core.governance.escalation import EscalationReason
from magma.core.interfaces.agent import Agent, AgentResult, Modality
from magma.core.utils.metrics import get_metrics


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ReviewDecision:
    approved: bool
    rationale: str
    pattern_annotation: str | None = None
    decided_by: str | None = None
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReviewTicket:
    episode_id: str
    reason: EscalationReason
    context: dict
    ticket_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ReviewStatus = ReviewStatus.PENDING
    decision: ReviewDecision | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _ticket_to_jsonable(ticket: ReviewTicket) -> dict:
    payload = asdict(ticket)
    payload["reason"] = ticket.reason.value
    payload["status"] = ticket.status.value
    payload["created_at"] = ticket.created_at.isoformat()
    if ticket.decision is not None:
        payload["decision"]["decided_at"] = ticket.decision.decided_at.isoformat()
    return payload


def _ticket_from_jsonable(payload: dict) -> ReviewTicket:
    decision = None
    if payload.get("decision") is not None:
        raw_decision = dict(payload["decision"])
        raw_decision["decided_at"] = datetime.fromisoformat(raw_decision["decided_at"])
        decision = ReviewDecision(**raw_decision)
    return ReviewTicket(
        episode_id=payload["episode_id"],
        reason=EscalationReason(payload["reason"]),
        context=payload["context"],
        ticket_id=payload["ticket_id"],
        status=ReviewStatus(payload["status"]),
        decision=decision,
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


class ReviewTicketStore(abc.ABC):
    """Storage-agnostic contract for persisting the review queue."""

    @abc.abstractmethod
    def load_all(self) -> dict[str, ReviewTicket]: ...

    @abc.abstractmethod
    def save_all(self, tickets: dict[str, ReviewTicket]) -> None: ...


class InMemoryTicketStore(ReviewTicketStore):
    """Default store: nothing survives past the current process, matching
    this module's original behavior. Fine for library use and tests;
    a CLI-driven deployment should use `JsonFileTicketStore` instead."""

    def load_all(self) -> dict[str, ReviewTicket]:
        return {}

    def save_all(self, tickets: dict[str, ReviewTicket]) -> None:
        return None


class JsonFileTicketStore(ReviewTicketStore):
    """Whole-file JSON snapshot of the ticket queue, rewritten atomically
    (write-to-temp then rename) on every mutation. Ticket volume is
    escalations, not high-throughput events, so a full rewrite per
    mutation is simpler than an append log and avoids having to replay
    updates to reconstruct a ticket's current state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_all(self) -> dict[str, ReviewTicket]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return {ticket_id: _ticket_from_jsonable(payload) for ticket_id, payload in raw.items()}

    def save_all(self, tickets: dict[str, ReviewTicket]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump({ticket_id: _ticket_to_jsonable(t) for ticket_id, t in tickets.items()}, f, indent=2)
        os.replace(tmp_path, self.path)


class HumanExpertAgent(Agent[dict, ReviewTicket]):
    name = "Human Expert Agent"
    modality = Modality.HUMAN

    def __init__(
        self, audit_logger: AuditLogger | None = None, store: ReviewTicketStore | None = None,
        metrics: object | None = None,
    ) -> None:
        super().__init__(blackboard=None)
        self.audit = audit_logger or AuditLogger()
        self.store = store or InMemoryTicketStore()
        self.metrics = metrics if metrics is not None else get_metrics()
        self._tickets: dict[str, ReviewTicket] = self.store.load_all()

    def enqueue(self, episode_id: str, reason: EscalationReason, context: dict) -> ReviewTicket:
        ticket = ReviewTicket(episode_id=episode_id, reason=reason, context=context)
        self._tickets[ticket.ticket_id] = ticket
        self.store.save_all(self._tickets)
        self.metrics.record_escalation(reason.value)
        self.audit.log(self.name, "escalate", episode_id=episode_id, reason=reason.value, ticket_id=ticket.ticket_id)
        return ticket

    def process(self, data: dict) -> AgentResult[ReviewTicket]:
        """Convenience wrapper over `enqueue` conforming to the `Agent`
        interface; expects `data` = {"episode_id", "reason", "context"}."""
        ticket = self.enqueue(data["episode_id"], data["reason"], data.get("context", {}))
        return AgentResult(agent_name=self.name, modality=self.modality, output=ticket, success=True)

    def submit_decision(
        self, ticket_id: str, *, approved: bool, rationale: str,
        pattern_annotation: str | None = None, decided_by: str | None = None,
    ) -> ReviewTicket:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"no such review ticket: {ticket_id!r}")
        ticket.decision = ReviewDecision(
            approved=approved, rationale=rationale, pattern_annotation=pattern_annotation, decided_by=decided_by
        )
        ticket.status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        self.store.save_all(self._tickets)
        self.audit.log(
            self.name, "decide", episode_id=ticket.episode_id, ticket_id=ticket_id,
            approved=approved, rationale=rationale,
        )
        return ticket

    def pending_tickets(self) -> list[ReviewTicket]:
        return [t for t in self._tickets.values() if t.status == ReviewStatus.PENDING]

    def get_ticket(self, ticket_id: str) -> ReviewTicket | None:
        return self._tickets.get(ticket_id)
