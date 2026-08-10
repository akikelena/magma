"""CrewAI wiring for the Documentation Change Detection Module.

`crewai` is imported lazily: Phase1Pipeline already runs the four Phase 1
agents correctly on its own, so this is a thin adapter letting the same
agents run under CrewAI's Crew/Task/Process scheduling instead of a direct
Python call. CrewAI owns scheduling/retries; MAGMA's agents keep the logic.

Unlike the phase2/3/4 crews (exactly one tool each, because their real
input is already-computed Python objects an LLM shouldn't be asked to
reconstruct), the Documentation Change Detector agent here gets four
tools and genuine autonomy over how to use them: which vendors to check,
whether a vendor's documentation actually changed, and whether that
change warrants running full classification. This is a deliberate,
narrow exception to keeping autonomous decision points to a minimum --
the four MAGMA components with real LLM autonomy are Prompt Generation,
Code Generation, Explainability, and Change Detection's own classification
stage; deciding "should I re-check this vendor" is new orchestration
surface this module adds, not a route through any of those, and a wrong
choice here just means a vendor gets checked again next run rather than a
value-affecting production decision -- the boundedness of that failure
mode is what makes delegating it to the agent an acceptable trade rather
than the kind of unconstrained autonomy that the same reliability
argument warns against elsewhere in the pipeline.
"""
from __future__ import annotations

from typing import Any

from magma.crewai_agents.settings import CrewAgentSettings, CrewSettings, load_agent_settings, load_crew_settings
from magma.orchestration.phase1_pipeline import Phase1Pipeline
from magma.phase1_documentation_detection.document_watcher import DocumentWatcher, DocumentWatcherResult
from magma.phase1_documentation_detection.documentation_processing_agent import RawDocument


def build_phase1_crew(
    pipeline: Phase1Pipeline,
    watcher: DocumentWatcher | None = None,
    agent_settings: CrewAgentSettings | None = None,
    crew_settings: CrewSettings | None = None,
) -> Any:
    """Build a CrewAI `Crew` whose task delegates to `pipeline`'s agents,
    with an agent that can autonomously scan registered vendors for
    documentation changes before deciding whether to classify them.

    `watcher` defaults to a `DocumentWatcher` over `pipeline`'s own
    Knowledge Repository. `agent_settings`/`crew_settings` default to
    config/crewai.yaml (phase1 / crew sections). Raises `ImportError` if
    crewai isn't installed, so callers can fall back to `pipeline.run(...)`
    directly.
    """
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover - exercised only when crewai is absent
        raise ImportError(
            "crewai is not installed; run `pip install crewai` to use the CrewAI-orchestrated "
            "Phase 1 crew, or call magma.orchestration.Phase1Pipeline.run(...) directly instead."
        ) from exc

    watcher = watcher or DocumentWatcher(pipeline.knowledge_repository)
    agent_settings = agent_settings or load_agent_settings("phase1")
    crew_settings = crew_settings or load_crew_settings()
    last_checks: dict[str, DocumentWatcherResult] = {}

    def _list_supported_vendors_tool() -> dict:
        return {
            "vendors": [
                {"vendor": v, "url": s.url, "description": s.description, "requires_js_rendering": s.requires_js_rendering}
                for v, s in sorted(watcher.sources.items())
            ]
        }

    def _check_vendor_documentation_tool(vendor: str) -> dict:
        try:
            result = watcher.check(vendor)
        except KeyError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a fetch failure must not crash the crew run
            return {"error": f"fetch failed for vendor {vendor!r}: {exc!r}"}
        last_checks[vendor] = result
        return {
            "vendor": vendor,
            "changed": result.changed,
            "is_placeholder": result.fetch.is_placeholder,
            "content_hash": result.fetch.content_hash[:16],
            "char_count": len(result.fetch.content),
            "had_previous_snapshot": result.previous_content is not None,
        }

    def _run_change_detection_for_vendor_tool(vendor: str) -> dict:
        """Classifies the change captured by the most recent
        `check_vendor_documentation` call for `vendor` in this run --
        call that tool for this vendor first."""
        if vendor not in last_checks:
            return {"error": f"no check_vendor_documentation call recorded for vendor {vendor!r} in this run yet"}
        check = last_checks[vendor]
        if check.fetch.is_placeholder:
            return {
                "error": f"fetch for vendor {vendor!r} was flagged as a placeholder "
                "(likely a client-rendered page with no real content) -- classification skipped",
            }
        old_content = check.previous_content if check.previous_content is not None else ""
        old_version = check.previous_version_id or "unknown"
        new_doc = RawDocument(content=check.fetch.content, source_url=check.source_url, vendor=vendor, version=check.new_version_id)
        old_doc = RawDocument(content=old_content, source_url=check.source_url, vendor=vendor, version=old_version)
        try:
            result = pipeline.run(old_doc, new_doc)
        except Exception as exc:  # noqa: BLE001 - a classification failure must not crash the crew run
            return {"error": f"classification failed for vendor {vendor!r}: {exc!r}"}
        return {
            "vendor": vendor,
            "changeset_id": result.change_set.changeset_id,
            "change_count": len(result.change_set.changes),
            "requires_human_review": result.requires_human_review,
            "conflicted": result.coherence.conflicted,
            "sequenced_template_ids": result.coherence.sequenced_template_ids,
        }

    def _classify_document_versions_tool(
        old_content: str, new_content: str, vendor: str, old_version: str, new_version: str,
        source_url: str = "",
    ) -> dict:
        """Direct classification when the caller already has both document
        versions in hand (no vendor registration or fetch needed). Fragment
        identity is source_url + heading_path + ordinal, so both versions
        must share one source_url or the graph can't match this fragment
        against an existing GENERATION edge; a synthetic fallback is used
        only if the caller omits it."""
        resolved_source_url = source_url or f"crewai://{vendor}"
        old_doc = RawDocument(content=old_content, source_url=resolved_source_url, vendor=vendor, version=old_version)
        new_doc = RawDocument(content=new_content, source_url=resolved_source_url, vendor=vendor, version=new_version)
        result = pipeline.run(old_doc, new_doc)
        return {
            "changeset_id": result.change_set.changeset_id,
            "change_count": len(result.change_set.changes),
            "requires_human_review": result.requires_human_review,
            "conflicted": result.coherence.conflicted,
            "sequenced_template_ids": result.coherence.sequenced_template_ids,
        }

    documentation_change_detector = crewai.Agent(
        role="Documentation Change Detector",
        goal=(
            "Monitor registered vendor documentation for changes and classify, sequence, and "
            "flag for review whatever documentation-driven changes are found"
        ),
        backstory=(
            "Coordinates the Documentation Processing, Knowledge Repository, Change Detection, "
            "and Version Coherence agents that make up MAGMA's Documentation Change Detection "
            "Module. Given a set of vendors to monitor, decides which ones need checking, "
            "recognizes when a fetch didn't return real content, and only spends classification "
            "effort on vendors whose documentation actually changed."
        ),
        tools=[
            _list_supported_vendors_tool,
            _check_vendor_documentation_tool,
            _run_change_detection_for_vendor_tool,
            _classify_document_versions_tool,
        ],
        **agent_settings.as_kwargs(),
    )

    detect_task = crewai.Task(
        description=(
            "Check every supported vendor's documentation for changes since it was last checked. "
            "For each vendor: check its documentation, skip it if the fetch was flagged as a "
            "placeholder or nothing changed, and otherwise classify the change. Report which "
            "vendors changed and their classification outcomes."
        ),
        agent=documentation_change_detector,
        expected_output="A JSON summary of which vendors were checked, which changed, and their classification outcomes.",
    )

    return crewai.Crew(
        agents=[documentation_change_detector], tasks=[detect_task], process=crewai.Process.sequential,
        **crew_settings.as_kwargs(),
    )
