"""Integration test: the full Documentation Change Detection Module,
wired together end to end, against real vendor SDK documentation.

Fixtures are derived from the actual installed `pymilvus` and
`weaviate-client` packages (their docstrings mirror milvus-io/pymilvus and
weaviate/weaviate-python-client on GitHub) with one deliberate parameter
rename introduced per pair, mirroring a "real dataset" evaluation
(change units drawn from VDBMS vendor documentation).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import Phase1Pipeline
from magma.phase1_documentation_detection import RawDocument

FIXTURES = Path(__file__).parent / "fixtures"


def _breaking_rename_responder(prompt: str) -> str:
    """Scripted classification standing in for the LLM stage: treats a
    parameter rename as BREAKING (old callers using the removed name fail),
    everything else as MINOR. Deterministic so the test doesn't require API
    access, mirroring temperature=0.0 reproducibility."""
    if "top_k" in prompt or "index_inverted" in prompt:
        payload = {
            "change_type": "BREAKING",
            "confidence": 0.95,
            "intent_shift": 0.9,
            "compatibility_preserved": False,
            "affected_entities": ["top_k", "limit"] if "top_k" in prompt else ["index_inverted"],
            "rationale": "parameter renamed; prior name no longer accepted",
        }
    else:
        payload = {
            "change_type": "MINOR",
            "confidence": 0.85,
            "intent_shift": 0.2,
            "compatibility_preserved": True,
            "affected_entities": [],
            "rationale": "wording change only, prior calls remain valid",
        }
    return "reasoning...\n" + json.dumps(payload)


@pytest.mark.parametrize(
    "old_file,new_file,vendor",
    [
        ("milvus_search_v2.3.md", "milvus_search_v2.4.md", "Milvus"),
        ("weaviate_property_v3.md", "weaviate_property_v4.md", "Weaviate"),
    ],
)
def test_phase1_pipeline_detects_breaking_rename(old_file, new_file, vendor):
    kr = KnowledgeRepositoryAgent()
    llm = MockLLMProvider(responder=_breaking_rename_responder)
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=llm, confidence_threshold=0.70)

    old_doc = RawDocument(
        content=(FIXTURES / old_file).read_text(encoding="utf-8"),
        source_url=f"https://github.com/{vendor.lower()}/{vendor.lower()}",
        vendor=vendor, version="old",
    )
    new_doc = RawDocument(
        content=(FIXTURES / new_file).read_text(encoding="utf-8"),
        source_url=f"https://github.com/{vendor.lower()}/{vendor.lower()}",
        vendor=vendor, version="new",
    )

    result = pipeline.run(old_doc, new_doc)

    assert result.change_set.vendor == vendor
    assert len(result.change_set.changes) > 0, "expected at least one detected change"
    severities = {c.change_type.name for c in result.change_set.changes}
    assert "BREAKING" in severities, f"expected a BREAKING change among {severities}"
    # the rename is a real behavioral break, so the pipeline should not
    # silently sequence it without flagging it for review
    breaking = [c for c in result.change_set.changes if c.change_type.name == "BREAKING"]
    assert all(not c.compatibility_preserved for c in breaking)


def test_low_confidence_escalation_creates_a_review_ticket():
    """Locks in the fix: Phase 1 escalations must land in the Human Expert
    Agent's queue, not just set a flag on the result object."""
    kr = KnowledgeRepositoryAgent()

    def low_confidence_responder(prompt: str) -> str:
        return json.dumps({
            "change_type": "MAJOR", "confidence": 0.2, "intent_shift": 0.3,
            "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
        })

    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=low_confidence_responder))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")

    result = pipeline.run(old_doc, new_doc)

    assert result.requires_human_review
    assert len(result.review_tickets) == 1
    assert result.review_tickets[0] in pipeline.human_expert_agent.pending_tickets()
    assert result.review_tickets[0].reason.value == "low_confidence"


def test_unparseable_classification_escalation_creates_a_review_ticket():
    """Regression test for the same class of bug as the low-confidence
    case above, one level up the stack: unparseable LLM output sets
    escalate=True on the Change Detection Agent's result but used to leave
    escalation_reason unset, so it never actually reached
    Phase1Result.requires_human_review or the Human Expert Agent's queue."""
    kr = KnowledgeRepositoryAgent()
    pipeline = Phase1Pipeline(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=lambda p: "not json at all"))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")

    result = pipeline.run(old_doc, new_doc)

    assert result.requires_human_review
    assert len(result.review_tickets) == 1
    assert result.review_tickets[0] in pipeline.human_expert_agent.pending_tickets()
    assert result.review_tickets[0].reason.value == "low_confidence"
