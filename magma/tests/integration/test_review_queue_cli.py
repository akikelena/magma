"""Locks in `review_queue.py` as the other end of a persisted review
queue: a ticket written by one HumanExpertAgent instance (standing in
for a pipeline run in a separate process) must be listable and
decidable through this CLI alone, reading only the shared JSON file."""
from __future__ import annotations

import json

from magma.core.governance.escalation import EscalationReason
from magma.phase3_validation import HumanExpertAgent, JsonFileTicketStore
from magma.scripts.review_queue import main


def _seed_ticket(store_path, reason=EscalationReason.LOW_CONFIDENCE, context=None):
    agent = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    return agent.enqueue("ep1", reason, context or {"rationale": "renamed field"})


def test_list_shows_pending_ticket_from_a_separate_writer(tmp_path, capsys):
    store_path = tmp_path / "reviews.json"
    ticket = _seed_ticket(store_path)

    exit_code = main(["--queue-path", str(store_path), "--list", "--json"])
    captured = capsys.readouterr()
    tickets = json.loads(captured.out)

    assert exit_code == 0
    assert len(tickets) == 1
    assert tickets[0]["ticket_id"] == ticket.ticket_id
    assert tickets[0]["status"] == "pending"


def test_list_against_missing_file_reports_no_pending_tickets(tmp_path, capsys):
    exit_code = main(["--queue-path", str(tmp_path / "nope.json"), "--list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no pending tickets" in captured.out


def test_approve_persists_decision_visible_to_a_third_reader(tmp_path, capsys):
    store_path = tmp_path / "reviews.json"
    ticket = _seed_ticket(store_path)

    exit_code = main([
        "--queue-path", str(store_path), "--approve", ticket.ticket_id,
        "--rationale", "safe rename", "--decided-by", "elena",
    ])
    assert exit_code == 0

    reader = HumanExpertAgent(store=JsonFileTicketStore(store_path))
    reloaded = reader.get_ticket(ticket.ticket_id)
    assert reloaded.status.value == "approved"
    assert reloaded.decision.decided_by == "elena"
    assert reloaded not in reader.pending_tickets()


def test_reject_requires_rationale(tmp_path, capsys):
    store_path = tmp_path / "reviews.json"
    ticket = _seed_ticket(store_path)

    exit_code = main(["--queue-path", str(store_path), "--reject", ticket.ticket_id])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--rationale is required" in captured.err


def test_show_unknown_ticket_reports_an_error(tmp_path, capsys):
    store_path = tmp_path / "reviews.json"
    _seed_ticket(store_path)

    exit_code = main(["--queue-path", str(store_path), "--show", "not-a-real-id"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no such ticket" in captured.err


def test_show_prints_full_ticket_detail(tmp_path, capsys):
    store_path = tmp_path / "reviews.json"
    ticket = _seed_ticket(store_path, context={"rationale": "renamed top_k to limit"})

    exit_code = main(["--queue-path", str(store_path), "--show", ticket.ticket_id])
    captured = capsys.readouterr()
    detail = json.loads(captured.out)

    assert exit_code == 0
    assert detail["context"] == {"rationale": "renamed top_k to limit"}
    assert detail["reason"] == "low_confidence"
