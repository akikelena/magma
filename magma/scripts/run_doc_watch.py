#!/usr/bin/env python3
"""Command-line entry point for scheduled/on-demand vendor documentation
watching: the "full pipeline for detecting changes in docs" production
entry point.

Each cycle: fetch every registered vendor's documentation (`config/vendor_sources.yaml`),
compare it against the last known snapshot stored in the Knowledge Repository, and run
Phase 1 classification on whatever actually changed. Placeholder fetches (pages that need
JS rendering, e.g. Qdrant today) are reported but never classified. A vendor whose fetch
fails outright is recorded as a cycle error and skipped, rather than aborting the whole run.

A vendor with no --templates-config entry only gets classified -- useful on its own for
visibility into a vendor you haven't wired anything to yet. A vendor WITH template mappings
gets the full MagmaWorkflow: Phase 2 regenerates each mapped template from the new
documentation, Phase 3 validates the regenerated content, and -- only if validation passes --
the template's file on disk is overwritten with the new content. This is what makes the
pipeline actually maintain the generators rather than just describe what changed about them.

Three ways to run it:

  # One pass over every registered vendor, human-readable summary. Classification only --
  # no templates are mapped to anything, so nothing gets regenerated.
  python -m magma.scripts.run_doc_watch --once

  # Continuous polling, checking every hour, only these two vendors, regenerating and
  # validating whatever templates.yaml maps to milvus/weaviate.
  python -m magma.scripts.run_doc_watch --interval-seconds 3600 --vendors milvus,weaviate \\
      --templates-config templates.yaml

  # Route the scan through the CrewAI-orchestrated Documentation Change Detector agent
  # instead of calling DocumentWatcher/Phase1Pipeline directly. Classification only --
  # --templates-config's regeneration path is deliberately not exposed as a crewai tool,
  # consistent with keeping autonomous decision points to the four components the paper
  # names (see crewai_agents/phase1_crew.py's module docstring); regeneration is a
  # value-affecting production action, unlike re-checking a vendor.
  python -m magma.scripts.run_doc_watch --once --use-crewai

--templates-config points at a small YAML file mapping vendor -> list of
{template_id, path, heading}:

    milvus:
      - template_id: search_template
        path: /templates/milvus_search.xtend
        heading: "Milvus Collection.search Reference/Parameters"

`path` is read at startup for Phase 2's "current template content" input and overwritten
in place after a validated regeneration; `heading` is the '/'-separated heading path
identifying which documentation fragment generates this template (same convention as
run_workflow.py's --heading), used to link the template into the Knowledge Repository graph.

By default the Knowledge Repository is in-memory, which means "last known"
state is lost every time the process exits -- fine for `--interval-seconds`
polling within one long-lived process, useless for `--once` runs from cron.
Pass --neo4j-uri/--neo4j-password (and, optionally, --weaviate-url) to persist
snapshots in the real backends across separate invocations instead.

The audit trail and human review queue are likewise in-memory by default;
pass --audit-log/--review-queue-path (or $AUDIT_LOG_PATH/$REVIEW_QUEUE_PATH)
to persist them across separate --once invocations (--interval-seconds
polling keeps them for the life of the process either way).

In --interval-seconds mode this process is meant to run as a long-lived
container: SIGTERM (what `docker stop` sends) is handled the same as
Ctrl-C, finishing the in-flight cycle's prints and exiting cleanly rather
than being hard-killed once Docker's grace period expires.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from magma.core.governance.audit_log import AuditLogger
from magma.core.interfaces.llm_provider import LLMProvider
from magma.core.models.documentation import stable_fragment_id
from magma.core.models.graph import EdgeType
from magma.core.utils.config import MagmaConfig, load_config
from magma.core.utils.metrics import start_metrics_server
from magma.knowledge_repository.graph_store import Neo4jGraphStore
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.knowledge_repository.vector_store import WeaviateVectorStore
from magma.llm_providers import MockLLMProvider
from magma.orchestration import MagmaWorkflow
from magma.orchestration.phase1_pipeline import Phase1Pipeline
from magma.phase1_documentation_detection.document_source import vendor_doc_sources
from magma.phase1_documentation_detection.document_watcher import DocumentWatcher, DocumentWatcherResult
from magma.phase1_documentation_detection.documentation_processing_agent import RawDocument
from magma.phase3_validation.human_expert_agent import HumanExpertAgent, JsonFileTicketStore


class _ShutdownRequested(Exception):
    """Raised from the SIGTERM handler so a signal delivered mid-sleep
    (the common case during --interval-seconds polling) unwinds the loop
    the same way Ctrl-C/KeyboardInterrupt already does, instead of Docker
    having to wait out its stop grace period and SIGKILL the process."""


def _handle_sigterm(signum, frame) -> None:
    raise _ShutdownRequested()


@dataclass(frozen=True)
class TemplateMapping:
    template_id: str
    path: Path
    heading: str


def load_template_mappings(config_path: str | Path) -> dict[str, list[TemplateMapping]]:
    """Parses --templates-config's vendor -> [{template_id, path, heading}] file."""
    with Path(config_path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    mappings: dict[str, list[TemplateMapping]] = {}
    for vendor, entries in raw.items():
        mappings[vendor] = [
            TemplateMapping(template_id=entry["template_id"], path=Path(entry["path"]), heading=entry["heading"])
            for entry in entries
        ]
    return mappings


def _register_template_mappings(
    kr: KnowledgeRepositoryAgent, sources, mappings: dict[str, list[TemplateMapping]]
) -> dict[str, dict[str, str]]:
    """Links every mapped template into the Knowledge Repository graph
    (GENERATION edge from the fragment its heading identifies) and reads
    its current on-disk content, once, at startup -- mirrors
    run_workflow.py's general (non---demo) path. Returns vendor ->
    {template_id: current content}, the per-vendor template_contents
    Phase 2 needs as its "what does this template currently say" input."""
    template_contents: dict[str, dict[str, str]] = {}
    for vendor, entries in mappings.items():
        if vendor not in sources:
            print(f"warning: --templates-config maps vendor {vendor!r}, which isn't a registered documentation source; ignoring", file=sys.stderr)
            continue
        source_url = sources[vendor].url
        vendor_contents: dict[str, str] = {}
        for entry in entries:
            kr.add_template_node(entry.template_id, {})
            fragment_id = stable_fragment_id(source_url, entry.heading.split("/"), 0)
            kr.link(entry.template_id, fragment_id, EdgeType.GENERATION)
            vendor_contents[entry.template_id] = entry.path.read_text() if entry.path.exists() else ""
        template_contents[vendor] = vendor_contents
    return template_contents


def _build_llm_provider(args: argparse.Namespace) -> LLMProvider:
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


def _build_knowledge_repository(args: argparse.Namespace) -> KnowledgeRepositoryAgent:
    if not args.neo4j_password:
        print(
            "warning: no --neo4j-password/$NEO4J_PASSWORD given; using an in-memory Knowledge "
            "Repository. 'Last known' snapshots will NOT persist across separate process runs -- "
            "fine for --interval-seconds polling, not for repeated --once invocations from cron.",
            file=sys.stderr,
        )
        return KnowledgeRepositoryAgent()

    graph_store = Neo4jGraphStore(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    vector_store = None
    if not args.skip_weaviate:
        vector_store = WeaviateVectorStore(url=args.weaviate_url, api_key=args.weaviate_api_key)
    return KnowledgeRepositoryAgent(graph_store=graph_store, vector_store=vector_store)


def _classify_change(pipeline: Phase1Pipeline, check: DocumentWatcherResult) -> dict:
    old_content = check.previous_content if check.previous_content is not None else ""
    old_version = check.previous_version_id or "unknown"
    old_doc = RawDocument(content=old_content, source_url=check.source_url, vendor=check.vendor, version=old_version)
    new_doc = RawDocument(content=check.fetch.content, source_url=check.source_url, vendor=check.vendor, version=check.new_version_id)
    try:
        result = pipeline.run(old_doc, new_doc)
    except Exception as exc:  # noqa: BLE001 - one vendor's classification failure must not abort the cycle
        return {"error": f"classification failed for vendor {check.vendor!r}: {exc!r}"}
    return {
        "changeset_id": result.change_set.changeset_id,
        "change_count": len(result.change_set.changes),
        "requires_human_review": result.requires_human_review,
        "conflicted": result.coherence.conflicted,
    }


def _regenerate_templates(
    workflow: MagmaWorkflow, check: DocumentWatcherResult,
    mappings: list[TemplateMapping], template_contents: dict[str, str],
) -> dict:
    """The full-pipeline path for a vendor with --templates-config
    entries: classify, then (unless escalated) regenerate and validate
    every mapped template, writing each one's file back to disk -- and
    updating `template_contents` in place so the next cycle's Phase 2
    call sees the just-regenerated content as "current" -- only if the
    regenerated content actually passed Phase 3 validation. A template
    whose regeneration or validation failed keeps its existing on-disk
    content: MAGMA never overwrites a working generator with an artifact
    it isn't confident in."""
    old_content = check.previous_content if check.previous_content is not None else ""
    old_version = check.previous_version_id or "unknown"
    old_doc = RawDocument(content=old_content, source_url=check.source_url, vendor=check.vendor, version=old_version)
    new_doc = RawDocument(content=check.fetch.content, source_url=check.source_url, vendor=check.vendor, version=check.new_version_id)
    try:
        result = workflow.run(old_doc, new_doc, template_contents)
    except Exception as exc:  # noqa: BLE001 - one vendor's regeneration failure must not abort the cycle
        return {"error": f"full-pipeline run failed for vendor {check.vendor!r}: {exc!r}"}

    report: dict = {
        "changeset_id": result.changeset_id,
        "status": result.status.value,
        "requires_human_review": result.phase1.requires_human_review,
        "change_count": len(result.phase1.change_set.changes),
    }
    if result.phase2 is None or result.phase3 is None:
        return report

    paths_by_template = {m.template_id: m.path for m in mappings}
    validated_by_template = {v.template_id: v for v in result.phase3.validated}
    written, kept_existing = [], []
    for artifact in result.phase2.artifacts:
        validated = validated_by_template.get(artifact.template_id)
        if validated is not None and validated.outcome and validated.outcome.passed:
            path = paths_by_template.get(artifact.template_id)
            if path is not None:
                path.write_text(artifact.content)
                template_contents[artifact.template_id] = artifact.content
            written.append(artifact.template_id)
        else:
            kept_existing.append(artifact.template_id)
    report["templates_validated_and_written"] = written
    report["templates_kept_existing_content"] = kept_existing
    report["templates_generation_failed"] = result.phase2.failed_template_ids
    return report


def _run_cycle_direct(
    watcher: DocumentWatcher, pipeline: Phase1Pipeline, workflow: MagmaWorkflow | None,
    template_mappings: dict[str, list[TemplateMapping]], template_contents: dict[str, dict[str, str]],
) -> dict:
    """Deterministic path: `DocumentWatcher.check_all()` (which already
    isolates one vendor's fetch failure from the rest of the batch), then
    per changed vendor either `_classify_change` (no template mappings)
    or `_regenerate_templates` (mapped -- runs the full MagmaWorkflow and
    writes validated artifacts back to disk). This is the default because
    the CLI's baseline job -- "tell me what changed" -- shouldn't require
    an LLM call or the optional crewai dependency to work at all."""
    checks = watcher.check_all()
    vendor_results = []
    for check in checks:
        entry: dict = {"vendor": check.vendor, "changed": check.changed, "is_placeholder": check.fetch.is_placeholder}
        if check.changed and not check.fetch.is_placeholder:
            mappings = template_mappings.get(check.vendor)
            if mappings and workflow is not None:
                entry["regeneration"] = _regenerate_templates(
                    workflow, check, mappings, template_contents.setdefault(check.vendor, {})
                )
            else:
                entry["classification"] = _classify_change(pipeline, check)
        vendor_results.append(entry)
    for vendor, error in watcher.last_errors.items():
        vendor_results.append({"vendor": vendor, "error": error})
    return {"vendors": vendor_results, "errors": dict(watcher.last_errors)}


def _run_cycle_crewai(pipeline: Phase1Pipeline, watcher: DocumentWatcher) -> dict:
    """Routes the same scan through the CrewAI-orchestrated Documentation
    Change Detector agent (`build_phase1_crew`) instead of calling
    DocumentWatcher/Phase1Pipeline directly, so its multi-tool autonomy is
    exercised for real rather than only in tests. Requires crewai and a
    live/configured LLM. Classification only -- see the module docstring
    for why template regeneration isn't exposed as a crewai tool."""
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    crew = build_phase1_crew(pipeline, watcher=watcher)
    crew.kickoff()
    # The crew's own tools already wrote DocumentWatcher snapshots and ran
    # classification; check_all() here is cheap (snapshots are already
    # fresh) and gives the same structured per-vendor summary as the
    # direct path, so callers get one consistent report shape either way.
    return _run_cycle_direct(watcher, pipeline, None, {}, {})


def _print_human_readable(cycle: dict) -> None:
    for entry in cycle["vendors"]:
        if "error" in entry:
            print(f"  {entry['vendor']}: ERROR - {entry['error']}")
            continue
        status = "CHANGED" if entry["changed"] else "unchanged"
        if entry.get("is_placeholder"):
            status += " (placeholder fetch, not classified)"
        print(f"  {entry['vendor']}: {status}")
        classification = entry.get("classification")
        if classification:
            if "error" in classification:
                print(f"    classification error: {classification['error']}")
            else:
                print(
                    f"    changeset {classification['changeset_id']}: "
                    f"{classification['change_count']} change(s), "
                    f"human review required: {classification['requires_human_review']}"
                )
        regeneration = entry.get("regeneration")
        if regeneration:
            if "error" in regeneration:
                print(f"    regeneration error: {regeneration['error']}")
            else:
                print(f"    changeset {regeneration['changeset_id']}: status={regeneration['status']}")
                if regeneration.get("templates_validated_and_written"):
                    print(f"      written (validated): {regeneration['templates_validated_and_written']}")
                if regeneration.get("templates_kept_existing_content"):
                    print(
                        "      kept existing content (validation failed): "
                        f"{regeneration['templates_kept_existing_content']}"
                    )
                if regeneration.get("templates_generation_failed"):
                    print(f"      generation failed: {regeneration['templates_generation_failed']}")


def build_arg_parser(config: MagmaConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_config()
    default_model = config.model_for("change_detection_agent").get("model", "claude-sonnet-4-5")
    change_detection_config = config.agent("change_detection_agent")
    default_confidence_threshold = change_detection_config.get("confidence_threshold", 0.70)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run a single cycle and exit (default if neither flag is given)")
    mode.add_argument("--interval-seconds", type=int, default=None, help="poll continuously, sleeping this many seconds between cycles")
    parser.add_argument(
        "--vendors", help="comma-separated subset of vendors to check (defaults to every vendor in config/vendor_sources.yaml)"
    )
    parser.add_argument(
        "--templates-config", default=os.environ.get("TEMPLATES_CONFIG_PATH"),
        help="YAML file mapping vendor -> [{template_id, path, heading}] (default: $TEMPLATES_CONFIG_PATH); "
             "a vendor with an entry here gets the full MagmaWorkflow (regenerate + validate + write) instead "
             "of classification only",
    )
    parser.add_argument(
        "--use-crewai", action="store_true",
        help="route the scan through the CrewAI Documentation Change Detector agent instead of calling the "
             "pipeline directly (classification only -- ignores --templates-config)",
    )
    parser.add_argument("--anthropic-api-key", help="Anthropic API key (defaults to $ANTHROPIC_API_KEY); only used with --use-crewai or when a change is classified")
    parser.add_argument("--model", default=default_model, help=f"Anthropic model id (defaults to config/llm_models.yaml: {default_model!r})")
    parser.add_argument("--confidence-threshold", type=float, default=default_confidence_threshold, help=f"defaults to config/agents.yaml ({default_confidence_threshold})")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"), help="if given (or $NEO4J_PASSWORD is set), snapshots persist in Neo4j/Weaviate instead of in-memory")
    parser.add_argument("--weaviate-url", default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"))
    parser.add_argument("--weaviate-api-key", default=os.environ.get("WEAVIATE_API_KEY") or None)
    parser.add_argument("--skip-weaviate", action="store_true", help="persist graph state in Neo4j but keep vector storage in-memory")
    parser.add_argument(
        "--audit-log", default=os.environ.get("AUDIT_LOG_PATH"),
        help="append this run's audit trail to this JSONL file (default: $AUDIT_LOG_PATH, or in-memory only)",
    )
    parser.add_argument(
        "--review-queue-path", default=os.environ.get("REVIEW_QUEUE_PATH"),
        help="persist escalated review tickets to this JSON file so magma.scripts.review_queue can act on them "
             "later (default: $REVIEW_QUEUE_PATH, or in-memory only)",
    )
    parser.add_argument(
        "--metrics-port", type=int, default=None,
        help="expose Prometheus /metrics on this port before the first cycle (requires prometheus_client; "
             "defaults to $PROMETHEUS_PORT if that's set and this flag isn't) -- also what a container "
             "healthcheck can probe to confirm the polling loop is alive",
    )
    parser.add_argument("--json", action="store_true", help="print each cycle's summary as JSON instead of human-readable text")
    return parser


def main(
    argv: list[str] | None = None,
    config: MagmaConfig | None = None,
    knowledge_repository: KnowledgeRepositoryAgent | None = None,
    llm_provider: LLMProvider | None = None,
) -> int:
    """`knowledge_repository` lets a caller share one Knowledge Repository
    instance across multiple `main()` calls without a real Neo4j backend
    -- used by the CLI's own tests to prove snapshot persistence, and
    available to any embedding caller that manages the repository's
    lifecycle itself. `llm_provider` similarly lets a caller supply a
    scripted MockLLMProvider (e.g. to exercise --templates-config's
    regenerate-and-validate path end to end) instead of the real
    Anthropic-or-placeholder-mock resolution --build_llm_provider does."""
    args = build_arg_parser(config=config).parse_args(argv)
    config = config or load_config()

    all_sources = vendor_doc_sources(config)
    vendors = sorted(all_sources) if args.vendors is None else [v.strip() for v in args.vendors.split(",") if v.strip()]
    unknown = [v for v in vendors if v not in all_sources]
    if unknown:
        print(f"error: unknown vendor(s): {', '.join(unknown)}; known vendors: {', '.join(sorted(all_sources))}", file=sys.stderr)
        return 2
    sources = {v: all_sources[v] for v in vendors}

    kr = knowledge_repository if knowledge_repository is not None else _build_knowledge_repository(args)
    llm_provider = llm_provider if llm_provider is not None else _build_llm_provider(args)
    audit = AuditLogger(sink_path=args.audit_log)
    human_expert_agent = HumanExpertAgent(
        audit_logger=audit, store=JsonFileTicketStore(args.review_queue_path) if args.review_queue_path else None
    )

    template_mappings: dict[str, list[TemplateMapping]] = {}
    template_contents: dict[str, dict[str, str]] = {}
    if args.templates_config:
        template_mappings = load_template_mappings(args.templates_config)
        template_contents = _register_template_mappings(kr, sources, template_mappings)

    workflow = MagmaWorkflow(
        knowledge_repository=kr, llm_provider=llm_provider, confidence_threshold=args.confidence_threshold,
        audit_logger=audit, human_expert_agent=human_expert_agent,
    )
    pipeline = workflow.phase1
    watcher = DocumentWatcher(kr, sources=sources, audit_logger=audit)

    metrics_port = args.metrics_port or (int(os.environ["PROMETHEUS_PORT"]) if os.environ.get("PROMETHEUS_PORT") else None)
    if metrics_port is not None:
        if start_metrics_server(metrics_port):
            print(f"Prometheus metrics exposed on :{metrics_port}/metrics", file=sys.stderr)
        else:
            print("warning: --metrics-port given but prometheus_client is not installed; skipping metrics export", file=sys.stderr)

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    exit_code = 0
    cycle_number = 0
    try:
        while True:
            cycle_number += 1
            if args.use_crewai:
                cycle = _run_cycle_crewai(pipeline, watcher)
            else:
                cycle = _run_cycle_direct(watcher, pipeline, workflow, template_mappings, template_contents)

            if args.json:
                print(json.dumps({"cycle": cycle_number, **cycle}, indent=2))
            else:
                print(f"cycle {cycle_number}: checked {len(vendors)} vendor(s)")
                _print_human_readable(cycle)

            if cycle["errors"]:
                exit_code = 1

            if args.interval_seconds is None:
                break
            time.sleep(args.interval_seconds)
    except (KeyboardInterrupt, _ShutdownRequested):
        print("stopped.", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
