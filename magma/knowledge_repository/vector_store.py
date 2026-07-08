"""Vector layer of the hybrid Knowledge Repository.

Stores embeddings of fragments and concepts and supports meaning-based
retrieval through cosine similarity, which remains effective when explicit
graph relationships are missing or evolving. `WeaviateVectorStore` is the
production backend (Weaviate 1.35); `InMemoryVectorStore` implements the
same contract for tests.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from magma.knowledge_repository.embeddings import cosine_similarity

DEFAULT_SIMILARITY_THRESHOLD = 0.78  # calibrated by grid search


@dataclass(frozen=True)
class SimilarityMatch:
    object_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(abc.ABC):
    def ensure_collection(self, dimensions: int = 1024) -> None:
        """Idempotently provision whatever the backend needs before
        vectors can be upserted (a collection/class/index definition). A
        no-op by default: `InMemoryVectorStore` needs no schema. Real
        backends (`WeaviateVectorStore`) override this."""
        return None

    @abc.abstractmethod
    def upsert(self, object_id: str, vector: list[float], metadata: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def similarity_search(
        self, query_vector: list[float], *, top_k: int = 10, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[SimilarityMatch]: ...

    @abc.abstractmethod
    def delete(self, object_id: str) -> None: ...

    def is_semantically_equivalent(
        self, vector_a: list[float], vector_b: list[float], *, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> bool:
        """True above `threshold`: the pair is treated as the same revision
        (no embedding regeneration needed). False below it: substantive
        change, regenerate. 0.78 was calibrated to bound the
        false-deletion rate."""
        return cosine_similarity(vector_a, vector_b) >= threshold


class InMemoryVectorStore(VectorStore):
    """Fake vector store for unit tests and offline development."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def upsert(self, object_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._vectors[object_id] = vector
        self._metadata[object_id] = metadata

    def similarity_search(
        self, query_vector: list[float], *, top_k: int = 10, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[SimilarityMatch]:
        scored = [
            SimilarityMatch(object_id=oid, score=cosine_similarity(query_vector, vec), metadata=self._metadata[oid])
            for oid, vec in self._vectors.items()
        ]
        scored = [m for m in scored if m.score >= threshold]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def delete(self, object_id: str) -> None:
        self._vectors.pop(object_id, None)
        self._metadata.pop(object_id, None)


class WeaviateVectorStore(VectorStore):
    """Weaviate-backed vector store (production configuration).

    The `weaviate` client is imported lazily so importing this module does
    not require a running Weaviate instance; only instantiation does.
    """

    def __init__(
        self, url: str, api_key: str | None = None, *, class_name: str = "MagmaFragment", grpc_port: int = 50051
    ) -> None:
        import weaviate  # noqa: F401
        from urllib.parse import urlsplit

        # `url` is a full URL (matching WEAVIATE_URL in .env.example, e.g.
        # "http://localhost:8080"), but connect_to_custom wants a bare
        # hostname plus separate scheme/port -- passing the raw URL string
        # as http_host silently produced an unreachable client (previous
        # bug: always assumed port 443 + TLS, which breaks the local
        # docker-compose setup on :8080 over plain HTTP). gRPC shares the
        # HTTP host and TLS setting but uses its own port (50051 by
        # default, matching docker-compose.yml's exposed port).
        parsed = urlsplit(url)
        host = parsed.hostname or url
        secure = parsed.scheme == "https"
        http_port = parsed.port or (443 if secure else 80)

        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self._client = weaviate.connect_to_custom(
            http_host=host, http_port=http_port, http_secure=secure,
            grpc_host=host, grpc_port=grpc_port, grpc_secure=secure,
            auth_credentials=auth,
        ) if hasattr(weaviate, "connect_to_custom") else weaviate.Client(url, auth_client_secret=auth)
        self._class_name = class_name

    def ensure_collection(self, dimensions: int = 1024) -> None:
        """Creates the `MagmaFragment` collection if it doesn't already
        exist, configured with no built-in vectorizer (MAGMA supplies its
        own MRL-truncated `text-embedding-3-large` vectors) and cosine
        distance (matching `DEFAULT_SIMILARITY_THRESHOLD`'s
        cosine-similarity semantics). Idempotent: safe to call on every
        startup. `dimensions` is accepted for interface symmetry with the
        Knowledge Repository's configured embedding size; Weaviate's HNSW
        index does not need the dimensionality declared up front."""
        import weaviate.classes as wvc

        if self._client.collections.exists(self._class_name):
            return
        self._client.collections.create(
            name=self._class_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
            properties=[
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="fragment_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source_url", data_type=wvc.config.DataType.TEXT),
            ],
        )

    def upsert(self, object_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        collection = self._client.collections.get(self._class_name)
        collection.data.insert(properties=metadata, vector=vector, uuid=object_id)

    def similarity_search(
        self, query_vector: list[float], *, top_k: int = 10, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[SimilarityMatch]:
        collection = self._client.collections.get(self._class_name)
        response = collection.query.near_vector(
            near_vector=query_vector, limit=top_k, return_metadata=["distance"]
        )
        matches = []
        for obj in response.objects:
            score = 1.0 - obj.metadata.distance  # cosine distance -> similarity
            if score >= threshold:
                matches.append(SimilarityMatch(object_id=str(obj.uuid), score=score, metadata=obj.properties))
        return matches

    def delete(self, object_id: str) -> None:
        collection = self._client.collections.get(self._class_name)
        collection.data.delete_by_id(object_id)
