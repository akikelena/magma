"""Unit tests for Neo4jGraphStore against the real neo4j driver package
(with a mocked session, not a live database) -- catches API drift or
typos in the Cypher/driver calls that a fully-stubbed module would let
through. Skipped if neo4j isn't installed; this repo's CI doesn't
require it, same as every other optional backend.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magma.knowledge_repository.graph_store import Neo4jGraphStore


def _store_with_mocked_driver():
    pytest.importorskip("neo4j")
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)  # bypass __init__'s live driver connection
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    store._database = "neo4j"
    return store, mock_session


def test_ensure_schema_creates_constraint_and_index():
    store, mock_session = _store_with_mocked_driver()
    store.ensure_schema()

    queries = [call.args[0] for call in mock_session.run.call_args_list]
    assert any("CREATE CONSTRAINT" in q and "IS UNIQUE" in q for q in queries)
    assert any("CREATE INDEX" in q for q in queries)
    assert all("IF NOT EXISTS" in q for q in queries)  # idempotent, safe to call every startup


def test_add_node_merges_by_node_id():
    store, mock_session = _store_with_mocked_driver()
    from magma.core.models.graph import GraphNode, NodeType

    node = GraphNode(node_type=NodeType.TEMPLATE_COMPONENT, node_id="t1", properties={"path": "x.xtend"})
    store.add_node(node)

    call = mock_session.run.call_args
    assert "MERGE (n:MagmaNode {node_id: $id})" in call.args[0]
    assert call.kwargs["id"] == "t1"
    assert call.kwargs["type"] == "template_component"


def test_get_node_round_trips_the_real_created_at_timestamp():
    """Regression test: get_node used to pop created_at out of the Neo4j
    record (to keep it out of `properties`) and then discard it, so the
    reconstructed GraphNode silently fell back to its created_at
    default -- the read time, not the node's real creation time. Every
    node loaded back from Neo4j had a corrupted provenance timestamp."""
    from datetime import datetime, timezone

    store, mock_session = _store_with_mocked_driver()
    mock_session.run.return_value.single.return_value = {
        "n": {
            "node_type": "template_component",
            "node_id": "t1",
            "created_at": "2020-01-01T00:00:00+00:00",
            "path": "x.xtend",
        }
    }

    node = store.get_node("t1")

    assert node.created_at == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert node.properties == {"path": "x.xtend"}
    assert node.node_id == "t1"
