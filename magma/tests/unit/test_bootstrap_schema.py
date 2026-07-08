"""Unit tests for the schema-bootstrap CLI, using fakes in
place of Neo4jGraphStore/WeaviateVectorStore so this runs without a live
Neo4j/Weaviate instance -- same pattern as the rest of the test suite,
which never requires that infrastructure."""
from __future__ import annotations

import magma.scripts.bootstrap_schema as bootstrap_schema


class _FakeGraphStore:
    instances: list["_FakeGraphStore"] = []

    def __init__(self, uri, user, password):
        self.uri, self.user, self.password = uri, user, password
        self.ensure_schema_called = False
        self.closed = False
        _FakeGraphStore.instances.append(self)

    def ensure_schema(self):
        self.ensure_schema_called = True

    def close(self):
        self.closed = True


class _FakeVectorStore:
    instances: list["_FakeVectorStore"] = []

    def __init__(self, url, api_key=None):
        self.url, self.api_key = url, api_key
        self.ensure_collection_dimensions = None
        _FakeVectorStore.instances.append(self)

    def ensure_collection(self, dimensions=1024):
        self.ensure_collection_dimensions = dimensions


def _reset():
    _FakeGraphStore.instances.clear()
    _FakeVectorStore.instances.clear()


def test_bootstraps_both_by_default(monkeypatch):
    _reset()
    monkeypatch.setattr(bootstrap_schema, "Neo4jGraphStore", _FakeGraphStore)
    monkeypatch.setattr(bootstrap_schema, "WeaviateVectorStore", _FakeVectorStore)

    exit_code = bootstrap_schema.main(["--neo4j-password", "secret", "--embedding-dimensions", "512"])

    assert exit_code == 0
    assert _FakeGraphStore.instances[0].ensure_schema_called
    assert _FakeGraphStore.instances[0].closed
    assert _FakeVectorStore.instances[0].ensure_collection_dimensions == 512


def test_missing_neo4j_password_is_rejected(monkeypatch, capsys):
    _reset()
    monkeypatch.setattr(bootstrap_schema, "Neo4jGraphStore", _FakeGraphStore)
    monkeypatch.setattr(bootstrap_schema, "WeaviateVectorStore", _FakeVectorStore)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    exit_code = bootstrap_schema.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "neo4j-password" in captured.err
    assert not _FakeGraphStore.instances


def test_skip_neo4j_only_bootstraps_weaviate(monkeypatch):
    _reset()
    monkeypatch.setattr(bootstrap_schema, "Neo4jGraphStore", _FakeGraphStore)
    monkeypatch.setattr(bootstrap_schema, "WeaviateVectorStore", _FakeVectorStore)

    exit_code = bootstrap_schema.main(["--skip-neo4j"])

    assert exit_code == 0
    assert not _FakeGraphStore.instances
    assert _FakeVectorStore.instances[0].ensure_collection_dimensions == 1024


def test_skip_weaviate_only_bootstraps_neo4j(monkeypatch):
    _reset()
    monkeypatch.setattr(bootstrap_schema, "Neo4jGraphStore", _FakeGraphStore)
    monkeypatch.setattr(bootstrap_schema, "WeaviateVectorStore", _FakeVectorStore)

    exit_code = bootstrap_schema.main(["--neo4j-password", "secret", "--skip-weaviate"])

    assert exit_code == 0
    assert _FakeGraphStore.instances[0].ensure_schema_called
    assert not _FakeVectorStore.instances
