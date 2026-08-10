#!/usr/bin/env python3
"""Command-line dashboard for the Human Expert Agent's review queue.

An escalated `ReviewTicket` is created by whichever process ran the
pipeline (`run_workflow.py`, `run_doc_watch.py`, or a library caller);
it can only be seen and decided by a human if that process persisted it
to a `JsonFileTicketStore`. This CLI is the other end of that queue: it
opens the same file and lets a human list pending tickets and record a
decision, closing the loop that makes escalation to a human meaningful
rather than a dead end.

    python -m magma.scripts.review_queue --queue-path reviews.json --list
    python -m magma.scripts.review_queue --queue-path reviews.json --show <ticket-id>
    python -m magma.scripts.review_queue --queue-path reviews.json \\
        --approve <ticket-id> --rationale "renamed param, safe" --decided-by alice
    python -m magma.scripts.review_queue --queue-path reviews.json \\
        --reject <ticket-id> --rationale "breaks backward compatibility"
"""
from __future__ import annotations

import argparse
import json
import sys

from magma.core.governance.audit_log import AuditLogger
from magma.phase3_validation.human_expert_agent import HumanExpertAgent, JsonFileTicketStore, ReviewTicket


def _summarize(ticket: ReviewTicket) -> dict:
    summary = {
        "ticket_id": ticket.ticket_id,
        "episode_id": ticket.episode_id,
        "reason": ticket.reason.value,
        "status": ticket.status.value,
        "created_at": ticket.created_at.isoformat(),
        "context": ticket.context,
    }
    if ticket.decision is not None:
        summary["decision"] = {
            "approved": ticket.decision.approved,
            "rationale": ticket.decision.rationale,
            "pattern_annotation": ticket.decision.pattern_annotation,
            "decided_by": ticket.decision.decided_by,
            "decided_at": ticket.decision.decided_at.isoformat(),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue-path", required=True, help="path to the JSON ticket store shared with the pipeline run(s) that escalated to it")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list pending tickets")
    action.add_argument("--show", metavar="TICKET_ID", help="show one ticket (pending or decided) in full")
    action.add_argument("--approve", metavar="TICKET_ID", help="approve a pending ticket")
    action.add_argument("--reject", metavar="TICKET_ID", help="reject a pending ticket")
    parser.add_argument("--rationale", help="required with --approve/--reject")
    parser.add_argument("--decided-by", help="reviewer identity, recorded on the decision and in the audit trail")
    parser.add_argument("--pattern-annotation", help="optional note the Adaptive Learning Agent can mine from decided tickets")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of human-readable text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if (args.approve or args.reject) and not args.rationale:
        print("error: --rationale is required with --approve/--reject", file=sys.stderr)
        return 2

    store = JsonFileTicketStore(args.queue_path)
    agent = HumanExpertAgent(audit_logger=AuditLogger(), store=store)

    if args.list:
        tickets = agent.pending_tickets()
        if args.json:
            print(json.dumps([_summarize(t) for t in tickets], indent=2))
        else:
            if not tickets:
                print("no pending tickets")
            for ticket in tickets:
                print(f"{ticket.ticket_id}  [{ticket.reason.value}]  episode={ticket.episode_id}  created={ticket.created_at.isoformat()}")
        return 0

    if args.show:
        ticket = agent.get_ticket(args.show)
        if ticket is None:
            print(f"error: no such ticket: {args.show!r}", file=sys.stderr)
            return 1
        print(json.dumps(_summarize(ticket), indent=2))
        return 0

    ticket_id = args.approve or args.reject
    try:
        ticket = agent.submit_decision(
            ticket_id, approved=bool(args.approve), rationale=args.rationale,
            pattern_annotation=args.pattern_annotation, decided_by=args.decided_by,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_summarize(ticket), indent=2))
    else:
        print(f"{ticket.ticket_id}: {ticket.status.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
