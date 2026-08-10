from magma.core.models.change import ChangeSet, ChangeType, DetectedChange
from magma.core.models.execution import ExecutionMode, MaintenanceCategory
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.phase2_artifact_generation import OrchestrationRequest, TaskOrchestrationAgent


def _cs(severity=ChangeType.MINOR):
    return ChangeSet(vendor="V", old_version="1", new_version="2", changes=[
        DetectedChange(fragment_id="f1", change_type=severity, rationale="x",
                        intent_shift=0.5, compatibility_preserved=True, confidence=0.9)
    ])


def test_single_template_is_lightweight():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(change_set=_cs(), sequenced_template_ids=["a"]))
    assert result.output.execution_mode == ExecutionMode.LIGHTWEIGHT


def test_multiple_templates_is_comprehensive():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(change_set=_cs(), sequenced_template_ids=["a", "b"]))
    assert result.output.execution_mode == ExecutionMode.COMPREHENSIVE


def test_parallelizable_flag_wins():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(change_set=_cs(), sequenced_template_ids=["a"], is_parallelizable=True))
    assert result.output.execution_mode == ExecutionMode.PARALLEL


def test_breaking_change_maps_to_transformative():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(change_set=_cs(ChangeType.BREAKING), sequenced_template_ids=["a"]))
    assert result.output.ordered_tasks[0].maintenance_category == MaintenanceCategory.TRANSFORMATIVE


def test_line_fraction_selects_generation_mode():
    from magma.core.models.execution import GenerationMode
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(
        change_set=_cs(), sequenced_template_ids=["a"], template_line_stats={"a": (50, 100)}
    ))
    assert result.output.ordered_tasks[0].generation_mode == GenerationMode.COMPREHENSIVE_SYNTHESIS


def test_quality_gate_reworks_until_limit_then_escalates():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    assert agent.evaluate_quality_gate(attempt=1, passed=False, max_iterations=3) is True
    assert agent.evaluate_quality_gate(attempt=3, passed=False, max_iterations=3) is False
    assert agent.evaluate_quality_gate(attempt=1, passed=True, max_iterations=3) is False


def test_self_sufficient_affected_artifact_and_dependency_analysis():
    """When sequenced_template_ids is omitted, the agent must derive it
    itself: GENERATION edges from the changed fragment locate templates,
    then DEPENDENCY edges fix modification order -- this agent's own
    analysis, not borrowed from Version Coherence."""
    from magma.core.models.graph import EdgeType, GraphEdge, GraphNode, NodeType

    repo = KnowledgeRepositoryAgent()
    graph = repo.graph
    graph.add_node(GraphNode(node_type=NodeType.DOCUMENTATION_FRAGMENT, node_id="f1"))
    graph.add_node(GraphNode(
        node_type=NodeType.TEMPLATE_COMPONENT, node_id="base",
        properties={"generation_rules": ["snake_case"], "output_formats": ["python"]},
    ))
    graph.add_node(GraphNode(
        node_type=NodeType.TEMPLATE_COMPONENT, node_id="derived",
        properties={"generation_rules": ["camelCase"], "output_formats": ["java"]},
    ))
    # "base" is generated from f1; "derived" depends on "base".
    graph.add_edge(GraphEdge(source_id="base", target_id="f1", edge_type=EdgeType.GENERATION))
    graph.add_edge(GraphEdge(source_id="derived", target_id="base", edge_type=EdgeType.DEPENDENCY))

    agent = TaskOrchestrationAgent(graph_store=graph)
    result = agent.process(OrchestrationRequest(change_set=_cs()))

    ids = [t.template_id for t in result.output.ordered_tasks]
    assert ids == ["base", "derived"]  # dependency precedes dependent

    by_id = {t.template_id: t for t in result.output.ordered_tasks}
    assert by_id["base"].generation_rules == ["snake_case"]
    assert by_id["base"].output_formats == ["python"]
    assert by_id["derived"].generation_rules == ["camelCase"]
    assert by_id["derived"].output_formats == ["java"]


def test_self_sufficient_analysis_with_no_graph_matches_returns_empty_plan():
    agent = TaskOrchestrationAgent(graph_store=KnowledgeRepositoryAgent().graph)
    result = agent.process(OrchestrationRequest(change_set=_cs()))
    assert result.output.ordered_tasks == []
