from magma.core.models.graph import EdgeType
from magma.core.models.graph import NodeType
from magma.core.models.template import TemplateComponent
from magma.knowledge_repository import KnowledgeRepositoryAgent


def test_write_read_versioning():
    kr = KnowledgeRepositoryAgent()
    v1 = kr.write("k", {"a": 1})
    v2 = kr.write("k", {"a": 2})
    assert kr.read("k") == {"a": 2}
    assert kr.read("k", version=v1) == {"a": 1}
    assert kr.history("k") == [v1, v2]


def test_rollback_appends_restored_version():
    kr = KnowledgeRepositoryAgent()
    v1 = kr.write("k", {"a": 1})
    kr.write("k", {"a": 2})
    kr.rollback("k", v1)
    assert kr.read("k") == {"a": 1}
    assert len(kr.history("k")) == 3


def test_graph_link_and_query():
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("t1", {})
    kr.add_fragment_node("f1", {})
    kr.link("t1", "f1", EdgeType.GENERATION)
    edges = kr.graph.edges_from("t1", EdgeType.GENERATION)
    assert len(edges) == 1
    assert edges[0].target_id == "f1"


def test_transitive_closure_and_topological_sort():
    kr = KnowledgeRepositoryAgent()
    for node in ("a", "b", "c"):
        kr.add_template_node(node, {})
    kr.link("b", "a", EdgeType.DEPENDENCY)  # b depends on a
    kr.link("c", "b", EdgeType.DEPENDENCY)  # c depends on b
    assert kr.graph.reverse_transitive_closure("a", EdgeType.DEPENDENCY) == {"b", "c"}
    order = kr.graph.topological_sort(["a", "b", "c"], EdgeType.DEPENDENCY)
    assert order.index("a") < order.index("b") < order.index("c")


def test_cycle_raises_on_topological_sort():
    kr = KnowledgeRepositoryAgent()
    kr.add_template_node("a", {})
    kr.add_template_node("b", {})
    kr.link("a", "b", EdgeType.DEPENDENCY)
    kr.link("b", "a", EdgeType.DEPENDENCY)
    try:
        kr.graph.topological_sort(["a", "b"], EdgeType.DEPENDENCY)
        assert False, "expected ValueError for cyclic graph"
    except ValueError:
        pass


def test_add_template_component_stores_structured_fields_as_node_properties():
    """TemplateComponent is the structured counterpart to
    add_template_node's raw properties dict; its fields must land under
    the same property keys Task Orchestration Agent's affected-artifact
    analysis reads back out (_template_metadata in
    task_orchestration_agent.py), or a template registered this way
    would silently lose its generation_rules/output_formats."""
    kr = KnowledgeRepositoryAgent()
    component = TemplateComponent(
        path="search.xtend", language="xtend", content="def search(): ...",
        template_id="search_template",
        generation_rules=["snake_case"], output_formats=["python"],
    )
    kr.add_template_component(component, attribution="test")

    node = kr.graph.get_node("search_template")
    assert node is not None
    assert node.node_type == NodeType.TEMPLATE_COMPONENT
    assert node.attribution == "test"
    assert node.properties["path"] == "search.xtend"
    assert node.properties["language"] == "xtend"
    assert node.properties["generation_rules"] == ["snake_case"]
    assert node.properties["output_formats"] == ["python"]
