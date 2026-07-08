"""Embedding client: text-embedding-3-large truncated via MRL to 1024 dims.

Matryoshka Representation Learning (Kusupati et al.) trains embeddings so
that any length-prefix of the full vector is itself a valid, near-optimal
embedding. Truncation is therefore just "take the first `dimensions`
components and re-normalize" -- no learned projection is needed at
inference time. This keeps retrieval quality high at reduced storage/search
cost.
"""
from __future__ import annotations

import math


def mrl_truncate(vector: list[float], dimensions: int) -> list[float]:
    """Truncate `vector` to its first `dimensions` components and L2-normalize."""
    if dimensions >= len(vector):
        truncated = list(vector)
    else:
        truncated = list(vector[:dimensions])
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm == 0:
        return truncated
    return [x / norm for x in truncated]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingClient:
    """Wraps an embedding backend and applies MRL truncation.

    `backend` is any callable `list[str] -> list[list[float]]` (e.g. the
    OpenAI SDK's `embeddings.create`), injected so tests can supply a
    deterministic fake instead of calling a real API.
    """

    def __init__(self, backend, *, model: str = "text-embedding-3-large", dimensions: int = 1024) -> None:
        self._backend = backend
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        raw_vectors = self._backend(texts)
        return [mrl_truncate(v, self.dimensions) for v in raw_vectors]


class OpenAIEmbeddingBackend:
    """Thin adapter over the OpenAI SDK, imported lazily."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        self._api_key = api_key
        self._model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import openai  # noqa: F401 (lazy import; only required if this backend is used)

        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
