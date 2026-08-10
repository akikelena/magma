from magma.core.governance.escalation import EscalationReason
from magma.core.utils.metrics import _NullMetrics
from magma.phase3_validation import HumanExpertAgent, JsonFileTicketStore
from magma.phase3_validation.human_expert_agent import ReviewStatus


def test_enqueue_creates_pending_ticket():
    agent = HumanExpertAgent()
    ticket = agent.enqueue("ep1", EscalationReason.SECURITY_CRITICAL, {"detail": "x"})
    assert ticket.status == ReviewStatus.PENDING
    assert ticket in agent.pending_tickets()


def test_submit_decision_resolves_ticket():
    agent = HumanExpertAgent()
    ticket = agent.enqueue("ep1", EscalationReason.LOW_CONFIDENCE, {})
    agent.submit_decision(ticket.ticket_id, approved=True, rationale="looks fine", decided_by="elena")
    assert ticket.status == ReviewStatus.APPROVED
    assert ticket.decision.approved is True
    assert ticket not in agent.pending_tickets()


def test_submit_decision_unknown_ticket_raises():
    agent = HumanExpertAgent()
    try:
        agent.submit_decision("nonexistent", approved=True, rationale="x")
        assert False, "expected KeyError"
    except KeyError:
        pass


class _RecordingMetrics(_NullMetrics):
    def __init__(self) -> None:
        self.escalations: list[str] = []

    def record_escalation(self, reason: str) -> None:
        self.escalations.append(reason)


def test_enqueue_records_escalation_metric():
    metrics = _RecordingMetrics()
    agent = HumanExpertAgent(metrics=metrics)
    agent.enqueue("ep1", EscalationReason.CONFLICTING_SPECIFICATION, {})
    assert metrics.escalations == ["conflicting_specification"]


def test_json_file_ticket_store_persists_across_agent_instances(tmp_path):
    """The whole point of JsonFileTicketStore: a ticket enqueued by one
    process (or, here, one HumanExpertAgent instance) must be visible to
    -- and decidable by -- a second instance reading the same file, since
    a real deployment's escalating run and its reviewing CLI invocation
    are two separate processes."""
    store_path = tmp_path / "reviews.json"

    writer = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    ticket = writer.enqueue("ep1", EscalationReason.LOW_CONFIDENCE, {"rationale": "renamed field"})

    reader = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    reloaded = reader.get_ticket(ticket.ticket_id)
    assert reloaded is not None
    assert reloaded.episode_id == "ep1"
    assert reloaded.reason == EscalationReason.LOW_CONFIDENCE
    assert reloaded.context == {"rationale": "renamed field"}
    assert reloaded.status == ReviewStatus.PENDING


def test_json_file_ticket_store_persists_a_decision_across_instances(tmp_path):
    store_path = tmp_path / "reviews.json"

    writer = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    ticket = writer.enqueue("ep1", EscalationReason.SECURITY_CRITICAL, {})

    decider = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    decider.submit_decision(ticket.ticket_id, approved=False, rationale="touches auth", decided_by="elena")

    reader = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    reloaded = reader.get_ticket(ticket.ticket_id)
    assert reloaded.status == ReviewStatus.REJECTED
    assert reloaded.decision.approved is False
    assert reloaded.decision.rationale == "touches auth"
    assert reloaded.decision.decided_by == "elena"
    assert reloaded not in reader.pending_tickets()


def test_json_file_ticket_store_missing_file_starts_empty(tmp_path):
    agent = HumanExpertAgent(store=JsonFileTicketStore(tmp_path / "does-not-exist-yet.json"))
    assert agent.pending_tickets() == []
