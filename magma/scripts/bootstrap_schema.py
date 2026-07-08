#!/usr/bin/env python3
"""Idempotent schema bootstrap for the Knowledge Repository's live
backends: creates the Neo4j uniqueness constraint/index and
the Weaviate collection that Neo4jGraphStore/WeaviateVectorStore assume
already exist. Safe to run every time (e.g. right after
`docker compose up`, or again after a fresh volume) since both stores'
`ensure_schema`/`ensure_collection` calls use CREATE ... IF NOT EXISTS /
existence checks.

Reads connection details from the environment (see .env.example):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  WEAVIATE_URL, WEAVIATE_API_KEY

Usage:
    docker compose up -d
    export $(grep -v '^#' .env | xargs)   # or otherwise load .env into the shell
    python -m magma.scripts.bootstrap_schema
"""
from __future__ import annotations

import argparse
import os
import sys

from magma.knowledge_repository.graph_store import Neo4jGraphStore
from magma.knowledge_repository.vector_store import DEFAULT_SIMILARITY_THRESHOLD, WeaviateVectorStore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--weaviate-url", default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"))
    parser.add_argument("--weaviate-api-key", default=os.environ.get("WEAVIATE_API_KEY") or None)
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--skip-neo4j", action="store_true", help="only bootstrap the Weaviate collection")
    parser.add_argument("--skip-weaviate", action="store_true", help="only bootstrap the Neo4j constraint/index")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.skip_neo4j:
        if not args.neo4j_password:
            print("error: --neo4j-password or $NEO4J_PASSWORD is required (skip with --skip-neo4j)", file=sys.stderr)
            return 2
        print(f"connecting to Neo4j at {args.neo4j_uri} ...")
        graph_store = Neo4jGraphStore(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
        try:
            graph_store.ensure_schema()
            print("  ok: magma_node_id_unique constraint + magma_node_type_idx index are in place")
        finally:
            graph_store.close()

    if not args.skip_weaviate:
        print(f"connecting to Weaviate at {args.weaviate_url} ...")
        vector_store = WeaviateVectorStore(url=args.weaviate_url, api_key=args.weaviate_api_key)
        vector_store.ensure_collection(dimensions=args.embedding_dimensions)
        print(
            f"  ok: MagmaFragment collection is in place "
            f"({args.embedding_dimensions}-dim vectors, cosine distance, similarity threshold {DEFAULT_SIMILARITY_THRESHOLD})"
        )

    print("schema bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
