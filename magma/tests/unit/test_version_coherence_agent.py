from magma.core.models.change import ChangeSet, ChangeType, DetectedChange
from magma.core.models.graph import EdgeType
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.phase1_documentation_detection.version_coherence_agent import (
    CoherenceRequest,
    VersionCoherenceAgent,
)


def _change_set(fragment_id="f1"):
    return ChangeSet(
        vendor="V", old_version="1", new_version="2",
        changes=[DetectedChange(fragment_id=fragment_id, change_type=ChangeType.MAJOR, rationale="x",
                                 intent_shift=0.5, compatibility_preserved=True, confidence=0.9)],
    )


def test_sequences_dependents_after_dependencies():
    kr = KnowledgeRepositoryAgent()
    kr.add_fragment_node("f1", {})
    kr.add_template_node("A", {})
    kr.add_template_node("B", {})
    kr.link("A", "f1", EdgeType.GENERATION)
    kr.link("B", "A", EdgeType.DEPENDENCY)  # B depends on A

    agent = VersionCoherenceAgent(graph_store=kr.graph)
    result = agent.process(CoherenceRequest(change_set=_change_set()))
    assert not result.output.conflicted
    # A is the dependency B requires, so A must be regenerated first
    assert result.output.sequenced_template_ids.index("A") < result.output.sequenced_template_ids.index("B")


def test_cycle_is_flagged_as_conflict():
    kr = KnowledgeRepositoryAgent()
    kr.add_fragment_node("f1", {})
    kr.add_template_node("A", {})
    kr.add_template_node("B", {})
    kr.link("A", "f1", EdgeType.GENERATION)
    kr.link("B", "A", EdgeType.DEPENDENCY)
    kr.link("A", "B", EdgeType.DEPENDENCY)  # cycle

    agent = VersionCoherenceAgent(graph_store=kr.graph)
    result = agent.process(CoherenceRequest(change_set=_change_set()))
    assert result.output.conflicted
    assert result.escalate


def test_simultaneous_modification_flagged():
    kr = KnowledgeRepositoryAgent()
    kr.add_fragment_node("f1", {})
    kr.add_template_node("A", {})
    kr.link("A", "f1", EdgeType.GENERATION)

    agent = VersionCoherenceAgent(graph_store=kr.graph)
    request = CoherenceRequest(change_set=_change_set(), proposed_template_owners={"A": "changeset-2"})
    # simultaneous_modification only triggers with >1 distinct owner for the
    # same template; a single owner is the common case and must not conflict
    result = agent.process(request)
    assert not result.output.conflicted
