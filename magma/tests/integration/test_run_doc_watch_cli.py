"""Locks in that `run_doc_watch.py` is a real, directly runnable CLI:
vendor filtering, unknown-vendor validation, JSON output, and a real
end-to-end cycle (fetch via the real `requests` library against a local
HTTP server, classify via a scripted mock LLM) that proves change
detection and classification actually run, not just parse.

Every test points the vendor registry at a local server rather than the
real vendor URLs -- the sandbox this suite runs in has no route to the
public internet, and a test suite shouldn't depend on one anyway."""
from __future__ import annotations

import http.server
import json
import os
import signal
import threading

import pytest

from magma.core.utils.config import MagmaConfig
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.scripts.run_doc_watch import main


@pytest.fixture
def local_http_server():
    body_holder = {"body": b"# Search Reference\n\nUses top_k. " + b"Filler content. " * 20}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.end_headers()
            self.wfile.write(body_holder["body"])

        def log_message(self, *args) -> None:  # silence default request logging
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, body_holder
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_unknown_vendor_exits_with_usage_error(capsys):
    exit_code = main(["--once", "--vendors", "not-a-real-vendor"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown vendor" in captured.err


def test_once_checks_every_registered_vendor_when_unfiltered(capsys, local_http_server):
    pytest.importorskip("requests")
    server, _ = local_http_server
    port = server.server_address[1]
    # Redirect all 5 default vendors at the local server -- proves every
    # registered vendor is attempted with no --vendors filter, without
    # depending on real internet access.
    config = MagmaConfig(vendor_sources={
        vendor: {"url": f"http://127.0.0.1:{port}/docs"}
        for vendor in ("milvus", "weaviate", "pinecone", "qdrant", "chroma")
    })

    exit_code = main(["--once", "--json"], config=config, knowledge_repository=KnowledgeRepositoryAgent())
    captured = capsys.readouterr()
    cycle = json.loads(captured.out)

    assert exit_code == 0
    assert {entry["vendor"] for entry in cycle["vendors"]} == {"milvus", "weaviate", "pinecone", "qdrant", "chroma"}
    assert all(entry["changed"] for entry in cycle["vendors"])


def test_real_end_to_end_cycle_detects_and_classifies_a_change(capsys, local_http_server):
    pytest.importorskip("requests")
    server, body_holder = local_http_server
    port = server.server_address[1]
    config = MagmaConfig(vendor_sources={"local": {"url": f"http://127.0.0.1:{port}/docs", "description": "test fixture"}})
    kr = KnowledgeRepositoryAgent()

    # First cycle establishes the baseline snapshot -- nothing to compare
    # against yet, so it's reported as "changed" (there's no prior version).
    exit_code = main(["--once", "--vendors", "local", "--json"], config=config, knowledge_repository=kr)
    captured = capsys.readouterr()
    first_cycle = json.loads(captured.out)
    assert exit_code == 0
    assert first_cycle["vendors"][0]["changed"] is True

    # Second cycle against identical content: no change, no classification.
    exit_code = main(["--once", "--vendors", "local", "--json"], config=config, knowledge_repository=kr)
    captured = capsys.readouterr()
    second_cycle = json.loads(captured.out)
    assert second_cycle["vendors"][0]["changed"] is False
    assert "classification" not in second_cycle["vendors"][0]

    # Third cycle against genuinely different content: change detected and classified.
    body_holder["body"] = b"# Search Reference\n\nUses limit. " + b"Filler content. " * 20
    exit_code = main(["--once", "--vendors", "local", "--json"], config=config, knowledge_repository=kr)
    captured = capsys.readouterr()
    third_cycle = json.loads(captured.out)
    entry = third_cycle["vendors"][0]
    assert entry["changed"] is True
    assert "classification" in entry
    assert "error" not in entry["classification"]
    assert entry["classification"]["change_count"] >= 1


def test_audit_log_and_review_queue_paths_persist_a_real_escalation(tmp_path, capsys, local_http_server):
    """--audit-log/--review-queue-path exist for the same reason as
    run_workflow.py's: without them, an escalation from a --once cron
    invocation is unreachable the moment the process exits. The
    unscripted default MockLLMProvider returns unparseable text, so a
    real content change here classifies as low-confidence and escalates
    -- proving the persisted files end up with real content, not just
    that the flags are accepted."""
    pytest.importorskip("requests")
    from magma.phase3_validation import HumanExpertAgent, JsonFileTicketStore

    server, body_holder = local_http_server
    port = server.server_address[1]
    config = MagmaConfig(vendor_sources={"local": {"url": f"http://127.0.0.1:{port}/docs"}})
    kr = KnowledgeRepositoryAgent()
    audit_log_path = tmp_path / "audit.jsonl"
    review_queue_path = tmp_path / "reviews.json"

    # Baseline snapshot.
    main(["--once", "--vendors", "local", "--json"], config=config, knowledge_repository=kr)
    capsys.readouterr()

    # Real content change -> classification runs -> unparseable mock
    # output -> low-confidence escalation.
    body_holder["body"] = b"# Search Reference\n\nUses limit. " + b"Filler content. " * 20
    exit_code = main(
        ["--once", "--vendors", "local", "--json", "--audit-log", str(audit_log_path), "--review-queue-path", str(review_queue_path)],
        config=config, knowledge_repository=kr,
    )
    captured = capsys.readouterr()
    cycle = json.loads(captured.out)
    assert exit_code == 0
    assert cycle["vendors"][0]["classification"]["requires_human_review"] is True

    audit_lines = audit_log_path.read_text().strip().splitlines()
    assert any(json.loads(line)["action"] == "escalate" for line in audit_lines)

    reader = HumanExpertAgent(store=JsonFileTicketStore(review_queue_path))
    assert len(reader.pending_tickets()) == 1


def test_sigterm_during_interval_polling_exits_cleanly(capsys):
    """`docker stop` sends SIGTERM, not SIGINT -- without a handler, the
    process would ignore it and be hard-killed once Docker's grace period
    expires, losing whatever the in-flight cycle hadn't flushed yet. A
    background timer delivers a real SIGTERM to this process partway
    through the sleep between cycles; main() must unwind and return
    instead of hanging until the 30s --interval-seconds elapses."""
    timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        # --vendors "" -> an empty vendor set, so each cycle is instant and
        # makes no network calls; the test only exercises signal timing.
        exit_code = main(["--interval-seconds", "30", "--vendors", ""])
    finally:
        timer.cancel()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stopped." in captured.err


def test_metrics_port_flag_starts_exporter_without_crashing(capsys):
    exit_code = main(["--once", "--vendors", "", "--json", "--metrics-port", "9198"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "9198" in captured.err
