#!/usr/bin/env python3
"""Command-line entry point for MAGMA: the tool must be directly
runnable, not just importable as a library.

Runs `MagmaWorkflow` end to end against two documentation versions: builds
the Knowledge Repository blackboard, links the affected template(s) to the
documentation fragment(s) that generate them, and prints the resulting
`WorkflowResult` (status, detected changes, validation outcome,
Phase 4 explanation).

Two ways to run it:

  # Zero-setup demo using the bundled Milvus fixture pair and a scripted
  # mock LLM provider -- no API key required, deterministic output.
  python -m magma.scripts.run_workflow --demo

  # Real run against your own documents and templates.
  python -m magma.scripts.run_workflow \\
      --old-doc old.md --new-doc new.md \\
      --source-url https://example.com/docs/api --vendor acme \\
      --old-version v1 --new-version v2 \\
      --heading "API Reference/search" \\
      --template search_template=path/to/current_search_template.py \\
      --anthropic-api-key sk-...   # or set ANTHROPIC_API_KEY

Without --anthropic-api-key/$ANTHROPIC_API_KEY and without --demo, the
script falls back to a mock provider that echoes back deterministic
placeholder completions -- useful for exercising the pipeline wiring
without spending API calls, but not representative of real LLM output.

Both the audit trail and the human review queue are in-memory by default,
so a run's escalations are unreachable once this process exits. Pass
--audit-log/--review-queue-path to persist them across separate
invocations; magma.scripts.review_queue reads --review-queue-path's file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.documentation import stable_fragment_id
from magma.core.utils.config import MagmaConfig, load_config
from magma.core.utils.metrics import start_metrics_server
from magma.core.models.graph import EdgeType
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import MagmaWorkflow, WorkflowResult
from magma.phase1_documentation_detection.documentation_processing_agent import RawDocument
from magma.phase3_validation.human_expert_agent import HumanExpertAgent, JsonFileTicketStore

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "integration" / "fixtures"
_DEMO_SOURCE_URL = "https://milvus.io/docs/search"
_DEMO_HEADING = ["Milvus Collection.search Reference", "Parameters"]
_DEMO_TEMPLATE_ID = "search_template"
_DEMO_TEMPLATE_CONTENT = "def search(limit):\n    return limit\n"


def _demo_responder(prompt: str) -> str:
    """Scripted, deterministic responses so `--demo` never needs a live
    LLM call, mirroring the fixture used by the integration test suite."""
    if "Reasoning from the previous stage" in prompt:
        return json.dumps({
            "template_content": "def search(limit):\n    return limit\n",
            "documentation_comments": "renamed top_k to limit",
            "test_cases": "def test_search():\n    assert search(5) == 5\n",
        })
    if "Audit trail:" in prompt:
        return "This episode renamed top_k to limit; validation passed on the first attempt."
    if prompt.strip().startswith("Fragment:"):
        return json.dumps({
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        })
    return "reasoning: replace top_k with limit"


def _build_llm_provider(args: argparse.Namespace) -> LLMProvider:
    if args.demo:
        return MockLLMProvider(responder=_demo_responder)
    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from magma.llm_providers import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=args.model)
    print(
        "warning: no Anthropic API key found (--anthropic-api-key or $ANTHROPIC_API_KEY); "
        "falling back to a mock provider that returns placeholder completions.",
        file=sys.stderr,
    )
    return MockLLMProvider()


def _parse_template_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--template must be TEMPLATE_ID=FILE_PATH, got: {raw!r}")
    template_id, _, path = raw.partition("=")
    return template_id, path


def _summarize(result: WorkflowResult) -> dict:
    summary: dict = {
        "status": result.status.value,
        "changeset_id": result.changeset_id,
        "detected_changes": [
            {"fragment_id": c.fragment_id, "change_type": c.change_type.name, "confidence": c.confidence}
            for c in result.phase1.change_set.changes
        ],
        "requires_human_review": result.phase1.requires_human_review,
    }
    if result.phase2 is not None:
        summary["artifacts_generated"] = [a.template_id for a in result.phase2.artifacts]
        summary["artifacts_failed"] = result.phase2.failed_template_ids
    if result.phase3 is not None:
        summary["validation_all_passed"] = result.phase3.all_passed
        summary["validation_escalated"] = [e.template_id for e in result.phase3.escalated]
    if result.phase4 is not None and result.phase4.explanation is not None:
        summary["explanation"] = result.phase4.explanation.narrative
    return summary


def build_arg_parser(config: MagmaConfig | None = None) -> argparse.ArgumentParser:
    """`config` defaults to `config/agents.yaml` + `config/llm_models.yaml`
    -- per-agent configuration is externalized, not hardcoded. The Change
    Detection Agent's confidence threshold, its model alias
    (resolved through llm_models.yaml), and the Automated Validation
    Agent's rework-attempt budget all come from that file; CLI flags
    still override them per-run."""
    config = config or load_config()
    change_detection_config = config.agent("change_detection_agent")
    validation_config = config.agent("automated_validation_agent")
    default_model = config.model_for("change_detection_agent").get("model", "claude-sonnet-4-5")
    default_confidence_threshold = change_detection_config.get("confidence_threshold", 0.70)
    default_max_attempts = validation_config.get("max_rework_iterations", 3)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="run the bundled zero-setup demo and ignore other inputs")
    parser.add_argument("--old-doc", type=Path, help="path to the old documentation version")
    parser.add_argument("--new-doc", type=Path, help="path to the new documentation version")
    parser.add_argument("--source-url", help="canonical URL identifying the documentation source")
    parser.add_argument("--vendor", help="vendor/product name, e.g. 'Milvus'")
    parser.add_argument("--old-version", help="version label of --old-doc")
    parser.add_argument("--new-version", help="version label of --new-doc")
    parser.add_argument(
        "--heading", action="append", default=[],
        help="'/'-separated heading path identifying the changed fragment (repeatable, pairs positionally with --template)",
    )
    parser.add_argument(
        "--template", action="append", default=[], type=_parse_template_arg, metavar="TEMPLATE_ID=FILE_PATH",
        help="a template generated from the matching --heading fragment, and the file holding its current content (repeatable)",
    )
    parser.add_argument("--anthropic-api-key", help="Anthropic API key (defaults to $ANTHROPIC_API_KEY)")
    parser.add_argument(
        "--model", default=default_model,
        help=f"Anthropic model id (defaults to config/llm_models.yaml via config/agents.yaml: {default_model!r})",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=default_confidence_threshold,
        help=f"defaults to config/agents.yaml change_detection_agent.confidence_threshold ({default_confidence_threshold})",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=default_max_attempts,
        help=f"defaults to config/agents.yaml automated_validation_agent.max_rework_iterations ({default_max_attempts})",
    )
    parser.add_argument("--json", action="store_true", help="print the full machine-readable summary as JSON")
    parser.add_argument(
        "--metrics-port", type=int, default=None,
        help="expose Prometheus /metrics on this port before running (requires prometheus_client; "
             "defaults to $PROMETHEUS_PORT if that's set and this flag isn't)",
    )
    parser.add_argument(
        "--audit-log", default=os.environ.get("AUDIT_LOG_PATH"),
        help="append this run's audit trail to this JSONL file (default: $AUDIT_LOG_PATH, or in-memory only)",
    )
    parser.add_argument(
        "--review-queue-path", default=os.environ.get("REVIEW_QUEUE_PATH"),
        help="persist escalated review tickets to this JSON file so magma.scripts.review_queue can act on them "
             "later (default: $REVIEW_QUEUE_PATH, or in-memory only -- an escalation from this run is otherwise "
             "unreachable once it exits)",
    )
    return parser


def main(argv: list[str] | None = None, config: MagmaConfig | None = None) -> int:
    args = build_arg_parser(config=config).parse_args(argv)

    kr = KnowledgeRepositoryAgent()

    if args.demo:
        old_doc = RawDocument(
            content=(_FIXTURES_DIR / "milvus_search_v2.3.md").read_text(),
            source_url=_DEMO_SOURCE_URL, vendor="Milvus", version="v2.3.x",
        )
        new_doc = RawDocument(
            content=(_FIXTURES_DIR / "milvus_search_v2.4.md").read_text(),
            source_url=_DEMO_SOURCE_URL, vendor="Milvus", version="v2.4.x",
        )
        kr.add_template_node(_DEMO_TEMPLATE_ID, {"path": "search.xtend"})
        kr.link(_DEMO_TEMPLATE_ID, stable_fragment_id(_DEMO_SOURCE_URL, _DEMO_HEADING, 0), EdgeType.GENERATION)
        template_contents = {_DEMO_TEMPLATE_ID: _DEMO_TEMPLATE_CONTENT}
    else:
        missing = [
            name for name, value in [
                ("--old-doc", args.old_doc), ("--new-doc", args.new_doc), ("--source-url", args.source_url),
                ("--vendor", args.vendor), ("--old-version", args.old_version), ("--new-version", args.new_version),
            ] if value is None
        ]
        if missing:
            print(f"error: missing required argument(s): {', '.join(missing)} (or pass --demo)", file=sys.stderr)
            return 2
        if len(args.template) != len(args.heading):
            print("error: --template and --heading must be given the same number of times, paired positionally", file=sys.stderr)
            return 2

        old_doc = RawDocument(content=args.old_doc.read_text(), source_url=args.source_url, vendor=args.vendor, version=args.old_version)
        new_doc = RawDocument(content=args.new_doc.read_text(), source_url=args.source_url, vendor=args.vendor, version=args.new_version)

        template_contents: dict[str, str] = {}
        for (template_id, content_path), heading in zip(args.template, args.heading):
            kr.add_template_node(template_id, {})
            fragment_id = stable_fragment_id(args.source_url, heading.split("/"), 0)
            kr.link(template_id, fragment_id, EdgeType.GENERATION)
            template_contents[template_id] = Path(content_path).read_text()

    metrics_port = args.metrics_port or (int(os.environ["PROMETHEUS_PORT"]) if os.environ.get("PROMETHEUS_PORT") else None)
    if metrics_port is not None:
        if start_metrics_server(metrics_port):
            print(f"Prometheus metrics exposed on :{metrics_port}/metrics", file=sys.stderr)
        else:
            print("warning: --metrics-port given but prometheus_client is not installed; skipping metrics export", file=sys.stderr)

    llm_provider = _build_llm_provider(args)
    audit = AuditLogger(sink_path=args.audit_log)
    human_expert_agent = HumanExpertAgent(
        audit_logger=audit, store=JsonFileTicketStore(args.review_queue_path) if args.review_queue_path else None
    )
    workflow = MagmaWorkflow(
        knowledge_repository=kr, llm_provider=llm_provider,
        confidence_threshold=args.confidence_threshold, max_validation_attempts=args.max_attempts,
        audit_logger=audit, human_expert_agent=human_expert_agent,
    )
    result = workflow.run(old_doc, new_doc, template_contents)
    summary = _summarize(result)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"status: {summary['status']}")
        print(f"changeset: {summary['changeset_id']}")
        print(f"detected changes: {len(summary['detected_changes'])}")
        for change in summary["detected_changes"]:
            print(f"  - {change['fragment_id']}: {change['change_type']} (confidence {change['confidence']:.2f})")
        if "artifacts_generated" in summary:
            print(f"artifacts generated: {summary['artifacts_generated']}")
        if "validation_all_passed" in summary:
            print(f"validation all passed: {summary['validation_all_passed']}")
        if "explanation" in summary:
            print(f"explanation: {summary['explanation']}")

    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
