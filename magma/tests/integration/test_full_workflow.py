"""End-to-end integration test: all four MAGMA modules wired together via
`MagmaWorkflow`, driven by the real Milvus fixture used throughout this
test suite, scaled down to one vendor/one change for a fast,
deterministic CI test.
"""
from __future__ import annotations

import json
from pathlib import Path

from magma.core.models.documentation import stable_fragment_id
from magma.core.models.graph import EdgeType
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import MockLLMProvider
from magma.orchestration import MagmaWorkflow, WorkflowStatus
from magma.phase1_documentation_detection import RawDocument

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE_URL = "https://milvus.io/docs/search"
HEADING_PATH = ["Milvus Collection.search Reference", "Parameters"]


def _responder(prompt: str) -> str:
    # Ordered from most to least structurally specific, since substrings
    # like "top_k" can legitimately appear embedded in later-stage prompts too.
    if "Reasoning from the previous stage" in prompt:
        return json.dumps({
            "template_content": "def search(limit):\n    return limit",
            "documentation_comments": "renamed top_k to limit",
            "test_cases": "def test_search():\n    assert search(5) == 5",
        })
    if "Audit trail:" in prompt:
        return "This episode renamed top_k to limit; validation passed on the first attempt."
    if prompt.strip().startswith("Fragment:"):
        return json.dumps({
            "change_type": "BREAKING", "confidence": 0.95, "intent_shift": 0.9,
            "compatibility_preserved": False, "affected_entities": ["top_k"], "rationale": "renamed",
        })
    return "reasoning: replace top_k with limit"


def test_full_workflow_completes_for_a_clean_breaking_rename():
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("search_template", {"path": "search.xtend"})
    kr.link("search_template", stable_fragment_id(SOURCE_URL, HEADING_PATH, 0), EdgeType.GENERATION)

    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_responder))
    old_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.3.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.3.x")
    new_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.4.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.4.x")

    result = workflow.run(old_doc, new_doc, template_contents={"search_template": "def search(top_k): ..."})

    assert result.status == WorkflowStatus.COMPLETED
    assert result.phase1.change_set.max_severity.name == "BREAKING"
    assert not result.phase1.requires_human_review
    assert result.phase2 is not None and len(result.phase2.artifacts) == 1
    assert result.phase3 is not None and result.phase3.all_passed
    assert result.phase4 is not None
    assert result.phase4.explanation.is_complete
    assert result.phase4.adaptive_learning is not None


def test_workflow_stops_before_phase2_when_phase1_flags_review():
    kr = KnowledgeRepositoryAgent()

    def low_confidence(prompt: str) -> str:
        if prompt.strip().startswith("Fragment:"):
            return json.dumps({
                "change_type": "MAJOR", "confidence": 0.2, "intent_shift": 0.3,
                "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
            })
        return "explanation" if "Audit trail:" in prompt else "n/a"

    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=low_confidence))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")

    result = workflow.run(old_doc, new_doc, template_contents={})

    assert result.status == WorkflowStatus.PENDING_HUMAN_REVIEW
    assert result.phase2 is None
    assert result.phase3 is None
    # Phase 4 still runs (explainability over whatever the audit trail has so far)
    assert result.phase4 is not None


def test_workflow_shares_one_human_expert_queue_across_phase1_and_phase3():
    """Phase 1's own escalation and Phase 3's escalation (if it occurs)
    must both show up in the same `MagmaWorkflow.human_expert_agent` queue."""
    kr = KnowledgeRepositoryAgent()

    def low_confidence(prompt: str) -> str:
        if prompt.strip().startswith("Fragment:"):
            return json.dumps({
                "change_type": "MAJOR", "confidence": 0.2, "intent_shift": 0.3,
                "compatibility_preserved": True, "affected_entities": [], "rationale": "ambiguous",
            })
        return "explanation" if "Audit trail:" in prompt else "n/a"

    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=low_confidence))
    old_doc = RawDocument(content="# T\n\nold text\n", source_url="u", vendor="V", version="old")
    new_doc = RawDocument(content="# T\n\nnew text\n", source_url="u", vendor="V", version="new")

    result = workflow.run(old_doc, new_doc, template_contents={})

    assert result.status == WorkflowStatus.PENDING_HUMAN_REVIEW
    assert len(workflow.human_expert_agent.pending_tickets()) == 1
    assert workflow.phase1.human_expert_agent is workflow.human_expert_agent
    assert workflow.phase3.human_expert_agent is workflow.human_expert_agent


def test_workflow_exercises_adaptive_learning_supervised_half():
    """Locks in the fix: a completed workflow run must actually predict a
    prompt structure and collect the successful episode for fine-tuning,
    not just update the bandit policy."""
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("search_template", {"path": "search.xtend"})
    kr.link("search_template", stable_fragment_id(SOURCE_URL, HEADING_PATH, 0), EdgeType.GENERATION)

    workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=MockLLMProvider(responder=_responder))
    old_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.3.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.3.x")
    new_doc = RawDocument(content=(FIXTURES / "milvus_search_v2.4.md").read_text(), source_url=SOURCE_URL, vendor="Milvus", version="v2.4.x")

    result = workflow.run(old_doc, new_doc, template_contents={"search_template": "def search(top_k): ..."})

    assert result.status == WorkflowStatus.COMPLETED
    # the successful episode's (prompt, completion) pair should already
    # have been dispatched to the fine-tuning backend by the time run() returns
    backend = workflow.adaptive_learning_agent.fine_tuning_collector.backend
    assert len(backend.submitted_batches) == 1
    assert backend.submitted_batches[0][0].template_id == "search_template"
    assert workflow.adaptive_learning_agent.prompt_structure_predictor is not None
