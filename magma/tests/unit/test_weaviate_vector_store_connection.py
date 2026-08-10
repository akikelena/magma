"""Unit tests for WeaviateVectorStore's URL parsing.

WeaviateVectorStore is otherwise untested against a live instance (no
Weaviate in CI, same as Neo4jGraphStore), but the host/port/scheme
resolution from a WEAVIATE_URL-style string is pure logic and worth
locking in directly: a prior bug passed the full URL string as the
hostname and hardcoded port 443 + TLS regardless of the actual URL,
which silently produced an unreachable client against the local
docker-compose Weaviate (plain HTTP on :8080).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from magma.knowledge_repository.vector_store import WeaviateVectorStore


@pytest.fixture
def fake_weaviate(monkeypatch):
    """Installs a stand-in `weaviate` module so WeaviateVectorStore's
    lazy `import weaviate` resolves without the real package, and we can
    inspect exactly what connect_to_custom was called with."""
    fake = types.ModuleType("weaviate")
    fake.connect_to_custom = MagicMock(return_value=MagicMock())
    fake.auth = types.SimpleNamespace(AuthApiKey=lambda key: ("api_key", key))
    monkeypatch.setitem(sys.modules, "weaviate", fake)
    return fake


def test_http_localhost_url_resolves_to_plain_http_with_default_grpc_port(fake_weaviate):
    WeaviateVectorStore(url="http://localhost:8080")
    kwargs = fake_weaviate.connect_to_custom.call_args.kwargs
    assert kwargs["http_host"] == "localhost"
    assert kwargs["http_port"] == 8080
    assert kwargs["http_secure"] is False
    assert kwargs["grpc_host"] == "localhost"
    assert kwargs["grpc_port"] == 50051
    assert kwargs["grpc_secure"] is False


def test_https_url_resolves_to_secure_with_default_port_443(fake_weaviate):
    WeaviateVectorStore(url="https://my-cluster.weaviate.network")
    kwargs = fake_weaviate.connect_to_custom.call_args.kwargs
    assert kwargs["http_host"] == "my-cluster.weaviate.network"
    assert kwargs["http_port"] == 443
    assert kwargs["http_secure"] is True
    assert kwargs["grpc_secure"] is True


def test_custom_grpc_port_override_is_respected(fake_weaviate):
    WeaviateVectorStore(url="http://localhost:8080", grpc_port=51000)
    kwargs = fake_weaviate.connect_to_custom.call_args.kwargs
    assert kwargs["grpc_port"] == 51000


def test_api_key_is_passed_through_as_auth_credentials(fake_weaviate):
    WeaviateVectorStore(url="http://localhost:8080", api_key="secret-key")
    kwargs = fake_weaviate.connect_to_custom.call_args.kwargs
    assert kwargs["auth_credentials"] == ("api_key", "secret-key")


def test_ensure_collection_uses_real_weaviate_v4_config_api():
    """Runs against the real weaviate-client package (not the fake_weaviate
    stub above) with a mocked client object, to catch API drift or typos
    in the Configure/Property/DataType symbols that a stubbed module
    would happily let through. Skipped if weaviate-client isn't
    installed -- this repo's CI doesn't require it, same as every other
    optional backend."""
    pytest.importorskip("weaviate")
    from unittest.mock import MagicMock

    store = WeaviateVectorStore.__new__(WeaviateVectorStore)  # bypass __init__'s network setup
    store._client = MagicMock()
    store._class_name = "MagmaFragment"
    store._client.collections.exists.return_value = False

    store.ensure_collection(dimensions=1024)

    store.collections = store._client.collections
    assert store._client.collections.exists.call_args.args == ("MagmaFragment",)
    create_kwargs = store._client.collections.create.call_args.kwargs
    assert create_kwargs["name"] == "MagmaFragment"
    # These are real weaviate.classes.config objects, not strings -- if
    # any of Configure.Vectorizer.none / Configure.VectorIndex.hnsw /
    # VectorDistances.COSINE / Property / DataType had been renamed or
    # misspelled, constructing them above would already have raised.
    assert create_kwargs["vectorizer_config"] is not None
    assert create_kwargs["vector_index_config"] is not None
    assert len(create_kwargs["properties"]) == 3


def test_ensure_collection_is_a_noop_when_collection_already_exists():
    pytest.importorskip("weaviate")
    from unittest.mock import MagicMock

    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store._client = MagicMock()
    store._class_name = "MagmaFragment"
    store._client.collections.exists.return_value = True

    store.ensure_collection(dimensions=1024)

    store._client.collections.create.assert_not_called()
