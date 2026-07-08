"""Integration test: Phase 1 (Documentation Change Detection) wired into
Phase 2 (Software Artifact Generation), against the real Milvus fixture.

Exercises the actual cross-module contract: Phase 2 reads Phase1Result off
the blackboard rather than being handed change data directly, and refuses
to run at all when Phase 1 flagged the changeset for human review.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from magma.core.models.documentation import stable_fragment_id
from magma.core.models.graph import EdgeType
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import Phase1Pipeline, Phase2Pipeline
from magma.phase1_documentation_detection import RawDocument

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE_URL = "https://milvus.io/docs/search"
HEADING_PATH = ["Milvus Collection.search Reference", "Parameters"]


def _detection_responder(prompt: str) -> str:
    if "top_k" in prompt:
        payload = {
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        }
    else:
        payload = {
            "change_type": "MINOR", "confidence": 0.85, "intent_shift": 0.2,
            "compatibility_preserved": True, "affected_entities": [], "rationale": "wording only",
        }
    return json.dumps(payload)


def _generation_responder(prompt: str) -> str:
    if "template_content" not in prompt and "Reasoning from" not in prompt:
        return "reasoning: replace top_k with limit"
    payload = {"template_content": "def search(limit): ...", "documentation_comments": "renamed top_k to limit",
               "test_cases": "def test_search(): pass"}
    return json.dumps(payload)


def test_phase1_output_flows_into_phase2_generation():
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("search_template", {"path": "search.xtend"})
    # Templates are registered against the stable fragment id ahead of
    # time, exactly as a real deployment would when it first indexes which
    # template generates from which documentation section.
    kr.link("search_template", stable_fragment_id(SOURCE_URL, HEADING_PATH, 0), EdgeType.GENERATION)

    phase1 = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_detection_responder))
    old_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.3.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.3.x")
    new_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.4.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.4.x")
    phase1_result = phase1.run(old_doc, new_doc)

    assert not phase1_result.requires_human_review
    assert "search_template" in phase1_result.coherence.sequenced_template_ids

    phase2 = Phase2Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_generation_responder))
    phase2_result = phase2.run(phase1_result, template_contents={"search_template": "def search(top_k): ..."})

    assert phase2_result.failed_template_ids == []
    assert len(phase2_result.artifacts) == 1
    artifact = phase2_result.artifacts[0]
    assert artifact.template_id == "search_template"
    assert "limit" in artifact.content


def test_phase2_refuses_to_run_when_phase1_flags_review():
    kr = KnowledgeRepositoryAgent()

    def low_confidence_responder(prompt: str) -> str:
        return json.dumps({
            "change_type": "MAJOR", "confidence": 0.2, "intent_shift": 0.3,
            "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
        })

    phase1 = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=low_confidence_responder))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")
    phase1_result = phase1.run(old_doc, new_doc)
    assert phase1_result.requires_human_review

    phase2 = Phase2Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider())
    with pytest.raises(RuntimeError, match="human review"):
        phase2.run(phase1_result, template_contents={})
