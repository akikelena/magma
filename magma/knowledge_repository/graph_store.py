"""Graph layer of the hybrid Knowledge Repository.

Three node types (documentation fragment, template component, semantic
concept) connected by four directed edge types (dependency, generation,
supersession, conflict). This explicitness is what permits the
transitive-closure impact analysis realized by the Version Coherence Agent
(the second line of defense).

`Neo4jGraphStore` is the production backend (Neo4j 2025.11).
`InMemoryGraphStore` implements the identical `GraphStore` protocol so
agents, tests, and CI can run without a live Neo4j instance.
"""
from __future__ import annotations

import abc
from collections import defaultdict, deque
from datetime import datetime
from typing import Iterable

from magma.core.models.graph import EdgeType, GraphEdge, GraphNode, NodeType


class GraphStore(abc.ABC):
    """Storage-agnostic contract used by every agent that touches the graph."""

    def ensure_schema(self) -> None:
        """Idempotently provision whatever the backend needs before nodes
        and edges can be written (constraints, indexes, ...). A no-op by
        default: `InMemoryGraphStore` needs no schema. Real backends
        (`Neo4jGraphStore`) override this."""
        return None

    @abc.abstractmethod
    def add_node(self, node: GraphNode) -> str: ...

    @abc.abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None: ...

    @abc.abstractmethod
    def add_edge(self, edge: GraphEdge) -> None: ...

    @abc.abstractmethod
    def edges_from(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]: ...

    @abc.abstractmethod
    def edges_to(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]: ...

    def transitive_closure(self, node_id: str, edge_type: EdgeType) -> set[str]:
        """Nodes reachable from `node_id` by following `edge_type` edges.

        Used by the Version Coherence Agent to propagate impact through
        dependency edges and surface cascading effects (second line of
        defense), and reused for cycle detection (a cycle exists iff
        `node_id` is reachable from itself).
        """
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self.edges_from(current, edge_type):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        visited.discard(node_id)
        return visited

    def reverse_transitive_closure(self, node_id: str, edge_type: EdgeType) -> set[str]:
        """Nodes that reach `node_id` by following `edge_type` edges, i.e.
        the reverse-reachable set. Used for cascading-impact propagation:
        if `node_id` changes, every node in this set transitively depends
        on it and may need re-validation.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self.edges_to(current, edge_type):
                if edge.source_id not in visited:
                    visited.add(edge.source_id)
                    queue.append(edge.source_id)
        visited.discard(node_id)
        return visited

    def introduces_cycle(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        """Would adding source_id -> target_id create a cycle in edge_type?"""
        if source_id == target_id:
            return True
        return source_id in self.transitive_closure(target_id, edge_type)

    def topological_sort(self, node_ids: Iterable[str], edge_type: EdgeType) -> list[str]:
        """Order `node_ids` so that dependencies precede dependents.

        Edges are read as "source depends on target" (one component
        requires another), so the returned order places each
        node's targets -- the things it requires -- before the node itself.
        Concretely: nodes with no unresolved dependencies (in this node set)
        come first; a node becomes eligible once every node it depends on
        has already been placed.

        Raises ValueError if the induced subgraph has a cycle, which the
        Version Coherence Agent treats as a rule-based conflict.
        """
        nodes = list(node_ids)
        node_set = set(nodes)
        # in_degree[n] = number of not-yet-resolved dependencies of n (within node_set)
        in_degree = {n: 0 for n in nodes}
        # notify[target] = sources that depend on target, to unblock once target is resolved
        notify: dict[str, list[str]] = defaultdict(list)
        for n in nodes:
            for edge in self.edges_from(n, edge_type):
                if edge.target_id in node_set:
                    notify[edge.target_id].append(n)
                    in_degree[n] += 1
        queue = deque(sorted(n for n in nodes if in_degree[n] == 0))
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for dependent in sorted(notify[current]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        if len(ordered) != len(nodes):
            raise ValueError("cycle detected: cannot topologically sort node set")
        return ordered


class InMemoryGraphStore(GraphStore):
    """Fake graph store for unit tests and offline development."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> str:
        self._nodes[node.node_id] = node
        return node.node_id

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)

    def edges_from(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        return [
            e for e in self._edges
            if e.source_id == node_id and (edge_type is None or e.edge_type == edge_type)
        ]

    def edges_to(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        return [
            e for e in self._edges
            if e.target_id == node_id and (edge_type is None or e.edge_type == edge_type)
        ]

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store (production configuration).

    The `neo4j` driver is imported lazily so importing this module does not
    require a running database; only instantiating `Neo4jGraphStore` does.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        import neo4j  # noqa: F401  (import here to fail fast only when actually used)

        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        """Creates the uniqueness constraint and lookup index MagmaNode
        relies on, if they don't already exist. `IF NOT EXISTS` makes this
        safe to call on every startup, not just once during provisioning
        -- there is no separate "has this been bootstrapped" state to get
        out of sync. Called by magma/scripts/bootstrap_schema.py and safe
        to call before any add_node/add_edge call."""
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT magma_node_id_unique IF NOT EXISTS "
                "FOR (n:MagmaNode) REQUIRE n.node_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX magma_node_type_idx IF NOT EXISTS "
                "FOR (n:MagmaNode) ON (n.node_type)"
            )

    def add_node(self, node: GraphNode) -> str:
        with self._driver.session(database=self._database) as session:
            session.run(
                "MERGE (n:MagmaNode {node_id: $id}) "
                "SET n.node_type = $type, n += $props, n.created_at = $created_at",
                id=node.node_id,
                type=node.node_type.value,
                props=node.properties,
                created_at=node.created_at.isoformat(),
            )
        return node.node_id

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._driver.session(database=self._database) as session:
            record = session.run(
                "MATCH (n:MagmaNode {node_id: $id}) RETURN n", id=node_id
            ).single()
        if record is None:
            return None
        props = dict(record["n"])
        node_type = NodeType(props.pop("node_type"))
        props.pop("node_id", None)
        created_at = props.pop("created_at", None)
        # Bug fix: this was popped to keep it out of `properties` and then
        # discarded, so every node round-tripped through Neo4j silently
        # got GraphNode's created_at default (the read time, not the
        # original write time) instead of its real provenance timestamp --
        # corrupting whatever timeline the Explainability Agent
        # reconstructs from it. Parse the persisted ISO string back.
        kwargs = {"node_type": node_type, "properties": props, "node_id": node_id}
        if created_at is not None:
            kwargs["created_at"] = datetime.fromisoformat(created_at)
        return GraphNode(**kwargs)

    def add_edge(self, edge: GraphEdge) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (a:MagmaNode {node_id: $src}), (b:MagmaNode {node_id: $dst}) "
                "MERGE (a)-[r:%s]->(b) SET r += $props" % edge.edge_type.value.upper(),
                src=edge.source_id,
                dst=edge.target_id,
                props=edge.properties,
            )

    def edges_from(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        rel = f":{edge_type.value.upper()}" if edge_type else ""
        query = f"MATCH (a:MagmaNode {{node_id: $id}})-[r{rel}]->(b) RETURN type(r) AS t, b.node_id AS target, properties(r) AS props"
        with self._driver.session(database=self._database) as session:
            rows = session.run(query, id=node_id).data()
        return [
            GraphEdge(source_id=node_id, target_id=r["target"], edge_type=EdgeType(r["t"].lower()), properties=r["props"])
            for r in rows
        ]

    def edges_to(self, node_id: str, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        rel = f":{edge_type.value.upper()}" if edge_type else ""
        query = f"MATCH (a)-[r{rel}]->(b:MagmaNode {{node_id: $id}}) RETURN type(r) AS t, a.node_id AS source, properties(r) AS props"
        with self._driver.session(database=self._database) as session:
            rows = session.run(query, id=node_id).data()
        return [
            GraphEdge(source_id=r["source"], target_id=node_id, edge_type=EdgeType(r["t"].lower()), properties=r["props"])
            for r in rows
        ]
