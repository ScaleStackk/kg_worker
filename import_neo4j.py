#!/usr/bin/env python3
"""Import a generated legal graph into Neo4j through its transactional HTTP API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a QuestaCase graph into Neo4j")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--url", default=os.environ.get("NEO4J_URL", "http://kg-storage:7474"))
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def document_id(graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata", {})
    value = str(metadata.get("document_id", "")).strip()
    if value:
        return value
    source = str(metadata.get("source_pdf", "unknown.pdf"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def neo4j_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return [item for item in value if item is not None]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def properties(item: dict[str, Any], doc_id: str) -> dict[str, Any]:
    result = {
        key: neo4j_value(value)
        for key, value in item.items()
        if value is not None
    }
    result["document_id"] = doc_id
    return result


def execute(
    session: requests.Session,
    endpoint: str,
    statements: list[dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    response = session.post(endpoint, json={"statements": statements}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
    return payload


def wait_for_neo4j(session: requests.Session, endpoint: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            execute(session, endpoint, [{"statement": "RETURN 1"}], 10)
            return
        except Exception as error:  # Storage can still be starting.
            last_error = error
            time.sleep(2)
    raise TimeoutError(f"Neo4j did not become ready: {last_error}")


def main() -> int:
    args = parse_args()
    if not args.password:
        raise ValueError("NEO4J_PASSWORD is required")
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    doc_id = document_id(graph)
    nodes = []
    node_ids: dict[str, str] = {}
    for raw_node in graph.get("nodes", []):
        original_id = str(raw_node.get("id", "")).strip()
        if not original_id:
            raise ValueError("Every graph node must have an ID")
        scoped_id = f"{doc_id}:{original_id}"
        node_ids[original_id] = scoped_id
        node = properties(raw_node, doc_id)
        node["source_graph_id"] = original_id
        node["id"] = scoped_id
        nodes.append(node)

    edges = []
    for raw_edge in graph.get("edges", []):
        original_id = str(raw_edge.get("id", "")).strip()
        source_id = str(raw_edge.get("source", "")).strip()
        target_id = str(raw_edge.get("target", "")).strip()
        if not original_id or source_id not in node_ids or target_id not in node_ids:
            raise ValueError("Every graph edge must reference valid node IDs")
        edge = properties(raw_edge, doc_id)
        edge["source_graph_id"] = original_id
        edge["id"] = f"{doc_id}:{original_id}"
        edge["source"] = node_ids[source_id]
        edge["target"] = node_ids[target_id]
        edges.append(edge)
    if not nodes:
        raise ValueError("The graph contains no nodes")

    endpoint = f"{args.url.rstrip('/')}/db/{args.database}/tx/commit"
    session = requests.Session()
    session.auth = (args.username, args.password)
    session.headers.update({"Accept": "application/json"})
    wait_for_neo4j(session, endpoint, args.timeout)

    started = time.perf_counter()
    execute(
        session,
        endpoint,
        [
            {
                "statement": (
                    "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
                    "FOR (node:GraphNode) REQUIRE node.id IS UNIQUE"
                )
            }
        ],
        args.timeout,
    )
    statements = [
        {
            "statement": (
                "MATCH (node:GraphNode {document_id: $document_id}) "
                "DETACH DELETE node"
            ),
            "parameters": {"document_id": doc_id},
        },
        {
            "statement": (
                "UNWIND $nodes AS item "
                "MERGE (node:GraphNode {id: item.id}) "
                "SET node += item"
            ),
            "parameters": {"nodes": nodes},
        },
        {
            "statement": (
                "UNWIND $edges AS item "
                "MATCH (source:GraphNode {id: item.source}) "
                "MATCH (target:GraphNode {id: item.target}) "
                "MERGE (source)-[relation:LEGAL_RELATION {id: item.id}]->(target) "
                "SET relation += item"
            ),
            "parameters": {"edges": edges},
        },
        {
            "statement": (
                "MATCH (node:GraphNode {document_id: $document_id}) "
                "OPTIONAL MATCH (node)-[relation:LEGAL_RELATION {document_id: $document_id}]->() "
                "RETURN count(DISTINCT node) AS nodes, count(DISTINCT relation) AS edges"
            ),
            "parameters": {"document_id": doc_id},
            "resultDataContents": ["row"],
        },
    ]
    payload = execute(session, endpoint, statements, args.timeout)
    row = payload["results"][-1]["data"][0]["row"]
    elapsed = time.perf_counter() - started
    if row != [len(nodes), len(edges)]:
        raise RuntimeError(
            f"Neo4j verification mismatch: expected {[len(nodes), len(edges)]}, got {row}"
        )
    print(
        f"Imported and verified {row[0]} nodes and {row[1]} edges in Neo4j "
        f"for {doc_id} in {elapsed:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
