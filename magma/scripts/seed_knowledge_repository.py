#!/usr/bin/env python3
"""Seeds the Knowledge Repository's live graph and vector stores with
real vendor documentation (change units drawn from VDBMS vendor
documentation), against actual Neo4j/Weaviate instances rather than only
the in-memory test doubles every other test/CLI path in this repo uses.

Runs DocumentationProcessingAgent over the bundled Milvus and Weaviate
vendor fixtures (the same real pymilvus/weaviate-client docstring-derived
files exercised throughout the test suite), which writes each resulting
fragment as a DOCUMENTATION_FRAGMENT graph node as a side effect
(KnowledgeRepositoryAgent is passed to the agent). This script then adds
the vector-layer embedding for each fragment, plus one TEMPLATE_COMPONENT
node per vendor with a GENERATION edge to the fragment it is generated
from -- so the resulting graph is immediately usable by
magma.scripts.run_workflow or TaskOrchestrationAgent's own
affected-artifact traversal.

Usage:
    docker compose up -d
    python -m magma.scripts.bootstrap_schema --neo4j-password ...
    python -m magma.scripts.seed_knowledge_repository --neo4j-password ...

Without --openai-api-key/$OPENAI_API_KEY, falls back to a deterministic
local embedding function (no network call, not semantically meaningful --
just enough for the vector layer to have something to search) so the
insert path can still be exercised end to end for free; pass a real key
for actual semantic retrieval quality.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from magma.core.models.documentation import stable_fragment_id
from magma.core.models.graph import EdgeType
from magma.core.models.template import TemplateComponent
from magma.knowledge_repository.embeddings import EmbeddingClient, OpenAIEmbeddingBackend
from magma.knowledge_repository.graph_store import Neo4jGraphStore
from magma.knowledge_repository.repository import KnowledgeRepositoryAgent
from magma.knowledge_repository.vector_store import WeaviateVectorStore
from magma.phase1_documentation_detection.documentation_processing_agent import (
    DocumentationProcessingAgent,
    RawDocument,
)

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "integration" / "fixtures"
_RAW_EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-large's native size, before MRL truncation

_DOCS = [
    {
        "vendor": "Milvus",
        "source_url": "https://github.com/milvus/milvus",
        "version": "v2.4.x",
        "fixture": "milvus_search_v2.4.md",
        "template_id": "milvus_search_template",
        "template_heading": ["Milvus Collection.search Reference", "Parameters"],
        "generation_rules": ["snake_case parameter names", "keyword-only optional arguments"],
        "output_formats": ["python"],
    },
    {
        "vendor": "Weaviate",
        "source_url": "https://github.com/weaviate/weaviate",
        "version": "v4.x",
        "fixture": "weaviate_property_v4.md",
        "template_id": "weaviate_property_template",
        "template_heading": ["Weaviate Property Configuration Reference", "Attributes"],
        "generation_rules": ["camelCase property names", "explicit DataType enum values"],
        "output_formats": ["python"],
    },
]


def _deterministic_embedding_backend(texts: list[str]) -> list[list[float]]:
    """A free, offline stand-in for a real embedding API: hashes each text
    to a reproducible pseudo-random vector. Not semantically meaningful
    (unrelated texts won't cluster), but lets the insert path -- and any
    later similarity_search call -- run end to end without network access
    or an API key."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        vectors.append([((seed * (i + 1)) % 9973) / 9973.0 for i in range(_RAW_EMBEDDING_DIMENSIONS)])
    return vectors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--weaviate-url", default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"))
    parser.add_argument("--weaviate-api-key", default=os.environ.get("WEAVIATE_API_KEY") or None)
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY") or None)
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.neo4j_password:
        print("error: --neo4j-password or $NEO4J_PASSWORD is required", file=sys.stderr)
        return 2

    graph_store = Neo4jGraphStore(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    vector_store = WeaviateVectorStore(url=args.weaviate_url, api_key=args.weaviate_api_key)

    if args.openai_api_key:
        backend = OpenAIEmbeddingBackend(api_key=args.openai_api_key)
    else:
        print("warning: no OpenAI API key found; using a deterministic offline embedding stand-in "
              "(not semantically meaningful, just enough to exercise the insert path).", file=sys.stderr)
        backend = _deterministic_embedding_backend
    embedding_client = EmbeddingClient(backend=backend, dimensions=args.embedding_dimensions)

    kr = KnowledgeRepositoryAgent(graph_store=graph_store, vector_store=vector_store, embedding_client=embedding_client)
    kr.graph.ensure_schema()
    kr.vectors.ensure_collection(dimensions=args.embedding_dimensions)

    processing_agent = DocumentationProcessingAgent(knowledge_repository=kr)

    fragment_count = 0
    template_count = 0
    for doc in _DOCS:
        raw_document = RawDocument(
            content=(_FIXTURES_DIR / doc["fixture"]).read_text(encoding="utf-8"),
            source_url=doc["source_url"], vendor=doc["vendor"], version=doc["version"],
        )
        result = processing_agent.process(raw_document)
        if not result.success or result.output is None:
            print(f"warning: failed to process {doc['fixture']}: {result.rationale}", file=sys.stderr)
            continue

        for fragment in result.output:
            kr.index_fragment(fragment.fragment_id, fragment.content, metadata={
                "content": fragment.content,
                "source_url": fragment.source_url,
                "fragment_id": fragment.fragment_id,
            })
            fragment_count += 1
        print(f"  inserted {len(result.output)} fragment(s) from {doc['fixture']} ({doc['vendor']})")

        template_fragment_id = stable_fragment_id(doc["source_url"], doc["template_heading"], 0)
        component = TemplateComponent(
            path=f"{doc['template_id']}.xtend", language="xtend", content="",
            template_id=doc["template_id"],
            generation_rules=doc["generation_rules"], output_formats=doc["output_formats"],
        )
        kr.add_template_component(component, attribution="seed_knowledge_repository")
        kr.link(doc["template_id"], template_fragment_id, EdgeType.GENERATION)
        template_count += 1
        print(f"  linked template '{doc['template_id']}' -> fragment {template_fragment_id[:12]}... via GENERATION")

    graph_store.close()
    print(f"seed complete: {fragment_count} fragment node(s) + embedding(s), {template_count} template node(s) inserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
