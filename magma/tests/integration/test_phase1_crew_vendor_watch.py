"""Tests for phase1_crew.py's multi-tool vendor-scanning autonomy: the
Documentation Change Detector agent gets list_supported_vendors,
check_vendor_documentation, run_change_detection_for_vendor, and
classify_document_versions, and is expected to decide for itself which
vendors need checking and which changes are worth classifying.

fake_crewai's kickoff(planner=...) hook stands in for real LLM tool
selection: `_scan_all_vendors_planner` below scripts the sequence a
competent LLM would follow (list vendors, check each, classify only the
ones that changed and returned real content) so the end-to-end result can
be locked in deterministically, without a live LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import Phase1Pipeline
from magma.phase1_documentation_detection.document_source import DocumentFetcher, FetchResult, VendorDocSource
from magma.phase1_documentation_detection.document_watcher import DocumentWatcher

FIXTURES = Path(__file__).parent / "fixtures"


def _breaking_rename_responder(prompt: str) -> str:
    if "Reasoning from the previous stage" in prompt:
        return json.dumps({
            "template_content": "def search(limit):\n    return limit",
            "documentation_comments": "renamed top_k to limit",
            "test_cases": "def test_search():\n    assert search(5) == 5",
        })
    if "Audit trail:" in prompt:
        return "renamed top_k to limit; validation passed on the first attempt."
    if prompt.strip().startswith("Fragment:"):
        return json.dumps({
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        })
    return "reasoning: replace top_k with limit"


class _TwoVendorFetcher(DocumentFetcher):
    """One vendor ("milvus") returns the real changed-Milvus fixture pair
    across two fetches; the other ("stable_vendor") returns identical
    content both times, so a run against it should report no change."""

    def __init__(self) -> None:
        self._milvus_versions = [
            (FIXTURES / "milvus_search_v2.3.md").read_text(),
            (FIXTURES / "milvus_search_v2.4.md").read_text(),
        ]
        self._milvus_calls = 0
        self._stable_content = "# Stable\n\nNothing here ever changes across checks.\n"

    def fetch(self, source: VendorDocSource) -> FetchResult:
        if source.vendor == "milvus":
            content = self._milvus_versions[min(self._milvus_calls, 1)]
            self._milvus_calls += 1
            return FetchResult(vendor="milvus", source_url=source.url, content=content)
        return FetchResult(vendor=source.vendor, source_url=source.url, content=self._stable_content)


def _sources() -> dict[str, VendorDocSource]:
    return {
        "milvus": VendorDocSource(vendor="milvus", url="https://milvus.io/docs/search"),
        "stable_vendor": VendorDocSource(vendor="stable_vendor", url="https://example.com/stable/docs"),
    }


def _scan_all_vendors_planner(agent, task):
    """Stand-in for LLM tool selection: list vendors, check each one, and
    classify only the ones that changed with real (non-placeholder)
    content -- exactly what the task description asks for."""
    tools = agent.tools_by_name
    vendors = [v["vendor"] for v in tools["_list_supported_vendors_tool"]()["vendors"]]
    outcomes = {}
    for vendor in vendors:
        check = tools["_check_vendor_documentation_tool"](vendor=vendor)
        if not check["changed"] or check["is_placeholder"]:
            outcomes[vendor] = {"skipped": True, **check}
            continue
        outcomes[vendor] = tools["_run_change_detection_for_vendor_tool"](vendor=vendor)
    return outcomes


def test_documentation_change_detector_has_four_autonomous_tools(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=KnowledgeRepositoryAgent(), llm_provider=MockLLMProvider())
    crew = build_phase1_crew(pipeline)

    tool_names = set(crew.agents[0].tools_by_name)
    assert tool_names == {
        "_list_supported_vendors_tool",
        "_check_vendor_documentation_tool",
        "_run_change_detection_for_vendor_tool",
        "_classify_document_versions_tool",
    }


def test_build_phase1_crew_defaults_to_a_watcher_over_the_real_vendor_registry(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=KnowledgeRepositoryAgent(), llm_provider=MockLLMProvider())
    crew = build_phase1_crew(pipeline)

    vendors = {v["vendor"] for v in crew.agents[0].tools_by_name["_list_supported_vendors_tool"]()["vendors"]}
    assert vendors == {"milvus", "weaviate", "pinecone", "qdrant", "chroma"}


def test_autonomous_scan_classifies_only_the_vendor_that_actually_changed(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    kr = KnowledgeRepositoryAgent()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_breaking_rename_responder))
    fetcher = _TwoVendorFetcher()
    watcher = DocumentWatcher(kr, fetcher=fetcher, sources=_sources())
    # Seed a "last known" snapshot for each vendor before the crew run, so
    # the crew's own check() call is a genuine second check rather than
    # an always-changed first one.
    watcher.check("milvus")
    watcher.check("stable_vendor")

    crew = build_phase1_crew(pipeline, watcher=watcher)
    outcomes = crew.kickoff(planner=_scan_all_vendors_planner)[0]

    assert outcomes["stable_vendor"]["skipped"] is True
    assert outcomes["stable_vendor"]["changed"] is False

    assert "skipped" not in outcomes["milvus"]
    assert outcomes["milvus"]["change_count"] == 1
    assert outcomes["milvus"]["sequenced_template_ids"] == []  # no template pre-linked to this source_url in this test


def test_run_change_detection_for_vendor_without_prior_check_returns_a_structured_error(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=KnowledgeRepositoryAgent(), llm_provider=MockLLMProvider())
    crew = build_phase1_crew(pipeline)

    result = crew.agents[0].tools_by_name["_run_change_detection_for_vendor_tool"](vendor="milvus")

    assert "error" in result


def test_run_change_detection_for_vendor_skips_placeholder_fetches(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    class _AlwaysPlaceholderFetcher(DocumentFetcher):
        def fetch(self, source: VendorDocSource) -> FetchResult:
            return FetchResult(vendor=source.vendor, source_url=source.url, content="short", is_placeholder=True)

    kr = KnowledgeRepositoryAgent()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider())
    watcher = DocumentWatcher(kr, fetcher=_AlwaysPlaceholderFetcher(), sources=_sources())
    crew = build_phase1_crew(pipeline, watcher=watcher)

    tools = crew.agents[0].tools_by_name
    tools["_check_vendor_documentation_tool"](vendor="milvus")
    result = tools["_run_change_detection_for_vendor_tool"](vendor="milvus")

    assert "error" in result
    assert "placeholder" in result["error"]


def test_check_vendor_documentation_tool_returns_structured_error_on_fetch_failure(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    class _AlwaysFailsFetcher(DocumentFetcher):
        def fetch(self, source: VendorDocSource) -> FetchResult:
            raise ConnectionError("simulated network failure")

    kr = KnowledgeRepositoryAgent()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider())
    watcher = DocumentWatcher(kr, fetcher=_AlwaysFailsFetcher(), sources=_sources())
    crew = build_phase1_crew(pipeline, watcher=watcher)

    result = crew.agents[0].tools_by_name["_check_vendor_documentation_tool"](vendor="milvus")

    assert "error" in result
    assert "simulated network failure" in result["error"]


def test_check_vendor_documentation_tool_returns_structured_error_for_unknown_vendor(fake_crewai):
    from magma.crewai_agents.phase1_crew import build_phase1_crew

    pipeline = Phase1Pipeline(knowledge_repository=KnowledgeRepositoryAgent(), llm_provider=MockLLMProvider())
    crew = build_phase1_crew(pipeline)

    result = crew.agents[0].tools_by_name["_check_vendor_documentation_tool"](vendor="not_a_real_vendor")

    assert "error" in result
