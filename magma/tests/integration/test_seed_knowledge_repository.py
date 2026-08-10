"""Integration test for the graph DB seed/insert script, using
InMemoryGraphStore/InMemoryVectorStore in place of the real
Neo4j/Weaviate backends -- same rationale as test_bootstrap_schema.py:
this repo's CI has no live instance of either, but the insert logic
itself (processing real vendor fixtures, writing fragment/template
nodes, GENERATION edges, and vector embeddings) is real and worth
locking in against fakes that implement the identical GraphStore/
VectorStore contracts.
"""
from __future__ import annotations

import magma.scripts.seed_knowledge_repository as seed_script
from magma.core.models.graph import EdgeType, NodeType
from magma.knowledge_repository.graph_store import InMemoryGraphStore
from magma.knowledge_repository.vector_store import InMemoryVectorStore


class _FakeNeo4jGraphStore(InMemoryGraphStore):
    instances: list["_FakeNeo4jGraphStore"] = []

    def __init__(self, uri, user, password):
        super().__init__()
        self.uri, self.user, self.password = uri, user, password
        self.closed = False
        _FakeNeo4jGraphStore.instances.append(self)

    def close(self):
        self.closed = True


class _FakeWeaviateVectorStore(InMemoryVectorStore):
    instances: list["_FakeWeaviateVectorStore"] = []

    def __init__(self, url, api_key=None):
        super().__init__()
        self.url, self.api_key = url, api_key
        _FakeWeaviateVectorStore.instances.append(self)


def _reset():
    _FakeNeo4jGraphStore.instances.clear()
    _FakeWeaviateVectorStore.instances.clear()


def test_seed_inserts_real_fragments_templates_and_edges(monkeypatch):
    _reset()
    monkeypatch.setattr(seed_script, "Neo4jGraphStore", _FakeNeo4jGraphStore)
    monkeypatch.setattr(seed_script, "WeaviateVectorStore", _FakeWeaviateVectorStore)

    exit_code = seed_script.main(["--neo4j-password", "secret"])
    assert exit_code == 0

    graph = _FakeNeo4jGraphStore.instances[0]
    vectors = _FakeWeaviateVectorStore.instances[0]
    assert graph.closed

    # Template nodes for both vendors, each linked via a GENERATION edge.
    milvus_template = graph.get_node("milvus_search_template")
    weaviate_template = graph.get_node("weaviate_property_template")
    assert milvus_template is not None and milvus_template.node_type == NodeType.TEMPLATE_COMPONENT
    assert weaviate_template is not None and weaviate_template.node_type == NodeType.TEMPLATE_COMPONENT

    milvus_generation_edges = graph.edges_from("milvus_search_template", EdgeType.GENERATION)
    weaviate_generation_edges = graph.edges_from("weaviate_property_template", EdgeType.GENERATION)
    assert len(milvus_generation_edges) == 1
    assert len(weaviate_generation_edges) == 1

    # The GENERATION edge must point at a real DOCUMENTATION_FRAGMENT node
    # that was actually inserted (not a dangling id).
    milvus_fragment = graph.get_node(milvus_generation_edges[0].target_id)
    assert milvus_fragment is not None
    assert milvus_fragment.node_type == NodeType.DOCUMENTATION_FRAGMENT

    # Real content from the bundled Milvus fixture reached the graph.
    assert "search" in milvus_template.properties["path"]
    # TemplateComponent's structured fields (Task Orchestration Agent's
    # affected-artifact analysis reads these back via _template_metadata).
    assert milvus_template.properties["generation_rules"] == ["snake_case parameter names", "keyword-only optional arguments"]
    assert milvus_template.properties["output_formats"] == ["python"]
    fragment_texts = " ".join(
        n.properties.get("content", "") for n in graph._nodes.values()
        if n.node_type == NodeType.DOCUMENTATION_FRAGMENT
    )
    assert "anns_field" in fragment_texts  # a real pymilvus parameter name from the fixture

    # Every inserted fragment got a vector too (offline deterministic
    # embedding backend, since no --openai-api-key was passed).
    fragment_node_count = sum(1 for n in graph._nodes.values() if n.node_type == NodeType.DOCUMENTATION_FRAGMENT)
    assert fragment_node_count > 0
    assert len(vectors._vectors) == fragment_node_count


def test_missing_neo4j_password_is_rejected(monkeypatch, capsys):
    _reset()
    monkeypatch.setattr(seed_script, "Neo4jGraphStore", _FakeNeo4jGraphStore)
    monkeypatch.setattr(seed_script, "WeaviateVectorStore", _FakeWeaviateVectorStore)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    exit_code = seed_script.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "neo4j-password" in captured.err
    assert not _FakeNeo4jGraphStore.instances
