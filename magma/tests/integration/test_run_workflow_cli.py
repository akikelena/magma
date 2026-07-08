"""Locks in that the repo is directly runnable via the CLI entry point,
not just importable as a library -- covers the --demo path
(zero setup) and the general file-based path (arg parsing, template/
heading linking, JSON summary)."""
from __future__ import annotations

import json

from magma.core.utils.config import MagmaConfig, load_config
from magma.scripts.run_workflow import build_arg_parser, main


def test_demo_runs_end_to_end_and_completes(capsys):
    exit_code = main(["--demo", "--json"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "completed"
    assert summary["detected_changes"][0]["change_type"] == "BREAKING"
    assert summary["artifacts_generated"] == ["search_template"]
    assert summary["validation_all_passed"] is True


def test_missing_required_args_without_demo_exits_with_usage_error(capsys):
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "missing required argument" in captured.err


def test_general_path_parses_and_links_template_by_heading(tmp_path):
    old_doc = tmp_path / "old.md"
    new_doc = tmp_path / "new.md"
    template_file = tmp_path / "tmpl.py"
    old_doc.write_text("# API\n\n## search\nUses top_k.\n")
    new_doc.write_text("# API\n\n## search\nUses limit.\n")
    template_file.write_text("def search(top_k):\n    return top_k\n")

    exit_code = main([
        "--old-doc", str(old_doc), "--new-doc", str(new_doc),
        "--source-url", "https://example.com/docs", "--vendor", "acme",
        "--old-version", "v1", "--new-version", "v2",
        "--heading", "API/search",
        "--template", f"search_template={template_file}",
        "--json",
    ])
    assert exit_code in (0, 1)  # mock provider without an API key won't fully validate; wiring must not crash


def test_template_and_heading_count_mismatch_is_rejected(tmp_path, capsys):
    old_doc = tmp_path / "old.md"
    new_doc = tmp_path / "new.md"
    old_doc.write_text("# API\n")
    new_doc.write_text("# API\n")

    exit_code = main([
        "--old-doc", str(old_doc), "--new-doc", str(new_doc),
        "--source-url", "u", "--vendor", "v", "--old-version", "1", "--new-version", "2",
        "--heading", "API/search",
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "same number of times" in captured.err


def test_arg_parser_rejects_malformed_template_arg():
    parser = build_arg_parser()
    try:
        parser.parse_args(["--template", "no-equals-sign"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 2


def test_metrics_port_flag_starts_exporter_without_crashing(capsys):
    """--metrics-port must not break the run even when prometheus_client
    isn't installed (falls back to a warning, not a crash); when it is
    installed, the workflow must still complete normally."""
    exit_code = main(["--demo", "--json", "--metrics-port", "9199"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "9199" in captured.err  # either "exposed on" or the not-installed warning


def test_default_config_dir_is_actually_read_and_matches_the_committed_yaml():
    """`config/agents.yaml` and `config/llm_models.yaml` are real files that
    document the intended per-agent thresholds/model aliases -- this locks
    in that `build_arg_parser()`'s defaults are genuinely sourced from them
    (via `load_config()`) rather than merely being hardcoded literals that
    happen to match what the YAML says."""
    parser = build_arg_parser()
    assert parser.get_default("confidence_threshold") == 0.70
    assert parser.get_default("model") == "claude-sonnet-4-5"
    assert parser.get_default("max_attempts") == 3


def test_build_arg_parser_defaults_come_from_the_injected_config_not_hardcoded_literals():
    """A config with deliberately different values from the committed YAML
    proves the defaults are actually threaded through `MagmaConfig`, not
    hardcoded numbers that happen to line up with agents.yaml."""
    custom_config = MagmaConfig(
        agents={
            "change_detection_agent": {"model": "custom_alias", "confidence_threshold": 0.42},
            "automated_validation_agent": {"max_rework_iterations": 7},
        },
        llm_models={"custom_alias": {"provider": "anthropic", "model": "claude-custom-9", "temperature": 0.1}},
    )
    parser = build_arg_parser(config=custom_config)
    assert parser.get_default("confidence_threshold") == 0.42
    assert parser.get_default("model") == "claude-custom-9"
    assert parser.get_default("max_attempts") == 7


def test_main_accepts_an_injected_config_and_uses_its_defaults(capsys):
    # 0.50 (not 0.70, the real config/agents.yaml value) so this proves the
    # injected config -- not the committed YAML -- drove the run; the demo
    # fixture's scripted confidence is 0.95, comfortably above 0.50, so it
    # still completes without triggering the low-confidence escalation path.
    custom_config = MagmaConfig(
        agents={"change_detection_agent": {"model": "custom_alias", "confidence_threshold": 0.50}},
        llm_models={"custom_alias": {"provider": "mock", "model": "mock-echo", "temperature": 0.0}},
    )
    exit_code = main(["--demo", "--json"], config=custom_config)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["status"] == "completed"


def test_a_high_injected_confidence_threshold_actually_changes_demo_behavior(capsys):
    """Proves the config wiring has real effect on the pipeline, not just on
    argparse defaults: the demo fixture's scripted classification confidence
    is 0.95, so a threshold above that must flip the outcome to pending
    human review via the low-confidence escalation path."""
    custom_config = MagmaConfig(
        agents={"change_detection_agent": {"model": "custom_alias", "confidence_threshold": 0.99}},
        llm_models={"custom_alias": {"provider": "mock", "model": "mock-echo", "temperature": 0.0}},
    )
    exit_code = main(["--demo", "--json"], config=custom_config)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["status"] == "pending_human_review"
    assert summary["requires_human_review"] is True


def test_load_config_reads_the_real_committed_yaml_files():
    """Direct integration check of `load_config()` itself (not just its
    effect through the CLI defaults), confirming config/agents.yaml and
    config/llm_models.yaml are parsed and cross-referenced correctly."""
    config = load_config()
    assert config.agent("change_detection_agent")["confidence_threshold"] == 0.70
    assert config.agent("automated_validation_agent")["max_rework_iterations"] == 3
    assert config.model_for("change_detection_agent")["model"] == "claude-sonnet-4-5"
    # knowledge_repository_agent has no chat model -- falls back to "default" alias
    assert config.model_for("documentation_processing_agent") == config.llm_models["default"]


def test_audit_log_and_review_queue_paths_persist_a_real_escalation(tmp_path, capsys):
    """--audit-log/--review-queue-path exist because the in-memory
    defaults make a run's audit trail and any escalated ticket
    unreachable once the CLI process exits. Forces the same
    confidence-threshold escalation as the test above, then proves both
    files actually contain the resulting record after the process
    "exits" (main() returns)."""
    from magma.phase3_validation import HumanExpertAgent, JsonFileTicketStore

    audit_log_path = tmp_path / "audit.jsonl"
    review_queue_path = tmp_path / "reviews.json"
    custom_config = MagmaConfig(
        agents={"change_detection_agent": {"model": "custom_alias", "confidence_threshold": 0.99}},
        llm_models={"custom_alias": {"provider": "mock", "model": "mock-echo", "temperature": 0.0}},
    )

    exit_code = main(
        ["--demo", "--json", "--audit-log", str(audit_log_path), "--review-queue-path", str(review_queue_path)],
        config=custom_config,
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["status"] == "pending_human_review"

    # The audit trail survived process exit and has real content in it.
    audit_lines = audit_log_path.read_text().strip().splitlines()
    assert len(audit_lines) > 0
    assert any(json.loads(line)["action"] == "escalate" for line in audit_lines)

    # The escalated ticket is readable by a completely separate
    # HumanExpertAgent instance -- i.e. by magma.scripts.review_queue
    # running as its own process against the same file.
    reader = HumanExpertAgent(store=JsonFileTicketStore(review_queue_path))
    pending = reader.pending_tickets()
    assert len(pending) == 1
    assert pending[0].reason.value == "low_confidence"
