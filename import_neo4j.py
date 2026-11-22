#!/usr/bin/env python3
"""Import a generated legal graph into Neo4j through its transactional HTTP API.

Entities that exist independently of one judgment (cases, statutes, sections,
judges, courts) are shared across documents: their node id is global and they
carry a `document_ids` list. Everything a judgment says about itself (facts,
issues, holdings, rules, procedure, outcome, parties) stays scoped to that
document, as do all edges. Re-importing a document removes only its own
scoped nodes and edges and its entry in shared nodes' `document_ids`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


SHARED_TYPES = {"CASE", "STATUTE", "SECTION", "JUDGE", "COURT"}
# Properties of shared nodes that are merged by dedicated Cypher, not `+=`.
SHARED_MANAGED = {"id", "label", "match_keys", "is_current_case", "document_id", "document_ids"}


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


def shared_id(original_id: str) -> str:
    # kg_create derives node ids from the node type and canonical key, so the
    # same entity gets the same id in every document.
    return f"shared:{original_id}"


def prepare(graph: dict[str, Any], doc_id: str) -> dict[str, Any]:
    """Split a graph into scoped nodes, shared nodes and scoped edges."""
    scoped: list[dict[str, Any]] = []
    shared: dict[str, dict[str, Any]] = {}
    node_ids: dict[str, str] = {}

    for raw_node in graph.get("nodes", []):
        original_id = str(raw_node.get("id", "")).strip()
        if not original_id:
            raise ValueError("Every graph node must have an ID")
        if raw_node.get("type") in SHARED_TYPES:
            node_id = shared_id(original_id)
            values = {key: neo4j_value(value) for key, value in raw_node.items() if value is not None}
            shared[node_id] = {
                "id": node_id,
                "type": raw_node["type"],
                "label": values.get("label", ""),
                "match_keys": values.get("match_keys", []),
                "is_current_case": bool(values.get("is_current_case")),
                "props": {
                    key: value for key, value in values.items() if key not in SHARED_MANAGED
                },
            }
        else:
            node_id = f"{doc_id}:{original_id}"
            node = properties(raw_node, doc_id)
            node["source_graph_id"] = original_id
            node["id"] = node_id
            scoped.append(node)
        node_ids[original_id] = node_id

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

    if not scoped and not shared:
        raise ValueError("The graph contains no nodes")
    return {"scoped": scoped, "shared": list(shared.values()), "edges": edges}


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


FIND_CASE_MATCHES = (
    "UNWIND $cases AS c "
    "MATCH (n:GraphNode {type: 'CASE'}) "
    "WHERE n.id = c.id OR any(key IN coalesce(n.match_keys, []) WHERE key IN c.match_keys) "
    "RETURN c.id, n.id, n.judgment_document_id"
)


def resolve_cases(
    prepared: dict[str, Any], rows: list[list[Any]], doc_id: str
) -> list[dict[str, str]]:
    """Point incoming CASE nodes at the existing node for the same case.

    `rows` are (incoming id, existing id, existing judgment_document_id) from
    FIND_CASE_MATCHES. A node that represents an uploaded judgment wins over a
    cited-case node, so a citation resolves to the real judgment. Two different
    uploaded judgments are never merged. Returns the existing nodes that must
    be folded into the kept node.
    """
    candidates: dict[str, dict[str, str | None]] = {}
    for incoming, existing, judgment in rows:
        candidates.setdefault(incoming, {})[existing] = judgment

    merges: dict[str, str] = {}
    renamed: dict[str, str] = {}
    for node in prepared["shared"]:
        if node["type"] != "CASE" or node["id"] not in candidates:
            continue
        found = candidates[node["id"]]
        if node["is_current_case"]:
            # This document's own case: never absorb another uploaded judgment.
            found = {key: value for key, value in found.items() if value in (None, doc_id)}
        judgments = sorted(key for key, value in found.items() if value)
        if len(judgments) > 1:
            # Ambiguous: two uploaded judgments match. Leave the node alone.
            continue
        if judgments:
            keep = judgments[0]
        elif node["id"] in found:
            keep = node["id"]
        elif found:
            keep = sorted(found)[0]
        else:
            continue
        for other in found:
            if other != keep and not found[other]:
                merges[other] = keep
        if keep != node["id"]:
            renamed[node["id"]] = keep

    if renamed:
        apply_renames(prepared, renamed)
    return [{"old": old, "keep": keep} for old, keep in sorted(merges.items())]


def apply_renames(prepared: dict[str, Any], renamed: dict[str, str]) -> None:
    combined: dict[str, dict[str, Any]] = {}
    for node in prepared["shared"]:
        node["id"] = renamed.get(node["id"], node["id"])
        if node["id"] in combined:
            kept = combined[node["id"]]
            kept["match_keys"] = list(dict.fromkeys(kept["match_keys"] + node["match_keys"]))
            kept["is_current_case"] = kept["is_current_case"] or node["is_current_case"]
            if node["is_current_case"]:
                kept["label"] = node["label"]
            kept["props"] = {**kept["props"], **node["props"]}
        else:
            combined[node["id"]] = node
    prepared["shared"] = list(combined.values())
    for edge in prepared["edges"]:
        edge["source"] = renamed.get(edge["source"], edge["source"])
        edge["target"] = renamed.get(edge["target"], edge["target"])


def import_statements(
    prepared: dict[str, Any], merges: list[dict[str, str]], doc_id: str
) -> list[dict[str, Any]]:
    params = {"document_id": doc_id}
    return [
        # 1. Remove this document's previous import (and legacy scoped data).
        {
            "statement": (
                "MATCH ()-[relation:LEGAL_RELATION {document_id: $document_id}]->() "
                "DELETE relation"
            ),
            "parameters": params,
        },
        {
            "statement": "MATCH (node:GraphNode {document_id: $document_id}) DETACH DELETE node",
            "parameters": params,
        },
        {
            "statement": (
                "MATCH (node:GraphNode) WHERE $document_id IN coalesce(node.document_ids, []) "
                "SET node.document_ids = [d IN node.document_ids WHERE d <> $document_id]"
            ),
            "parameters": params,
        },
        {
            "statement": (
                "MATCH (node:GraphNode {judgment_document_id: $document_id}) "
                "SET node.is_current_case = false REMOVE node.judgment_document_id"
            ),
            "parameters": params,
        },
        # 2. Fold duplicate shared CASE nodes into the kept node.
        {
            "statement": (
                "UNWIND $merges AS m "
                "MATCH (old:GraphNode {id: m.old}) MATCH (keep:GraphNode {id: m.keep}) "
                "CALL { WITH old, keep "
                "  MATCH (old)-[r:LEGAL_RELATION]->(target) "
                "  CREATE (keep)-[copy:LEGAL_RELATION]->(target) SET copy = properties(r), "
                "    copy.source = keep.id "
                "  DELETE r } "
                "CALL { WITH old, keep "
                "  MATCH (source)-[r:LEGAL_RELATION]->(old) "
                "  CREATE (source)-[copy:LEGAL_RELATION]->(keep) SET copy = properties(r), "
                "    copy.target = keep.id "
                "  DELETE r } "
                "SET keep.document_ids = reduce(acc = coalesce(keep.document_ids, []), "
                "      d IN coalesce(old.document_ids, []) | "
                "      CASE WHEN d IN acc THEN acc ELSE acc + d END), "
                "    keep.match_keys = reduce(acc = coalesce(keep.match_keys, []), "
                "      k IN coalesce(old.match_keys, []) | "
                "      CASE WHEN k IN acc THEN acc ELSE acc + k END) "
                "DETACH DELETE old"
            ),
            "parameters": {"merges": merges},
        },
        # 3. Write nodes.
        {
            "statement": (
                "UNWIND $nodes AS item "
                "MERGE (node:GraphNode {id: item.id}) "
                "SET node += item"
            ),
            "parameters": {"nodes": prepared["scoped"]},
        },
        {
            "statement": (
                "UNWIND $nodes AS item "
                "MERGE (node:GraphNode {id: item.id}) "
                "ON CREATE SET node.label = item.label "
                "SET node += item.props, "
                "    node.type = item.type, "
                "    node.document_ids = CASE "
                "      WHEN $document_id IN coalesce(node.document_ids, []) THEN node.document_ids "
                "      ELSE coalesce(node.document_ids, []) + $document_id END, "
                "    node.match_keys = reduce(acc = coalesce(node.match_keys, []), "
                "      k IN item.match_keys | CASE WHEN k IN acc THEN acc ELSE acc + k END), "
                "    node.is_current_case = coalesce(node.is_current_case, false) OR item.is_current_case "
                "FOREACH (_ IN CASE WHEN item.is_current_case THEN [1] ELSE [] END | "
                "  SET node.judgment_document_id = $document_id, node.label = item.label)"
            ),
            "parameters": {"nodes": prepared["shared"], **params},
        },
        # 4. Drop shared nodes no document uses any more.
        {
            "statement": (
                "MATCH (node:GraphNode) "
                "WHERE node.document_id IS NULL AND node.document_ids = [] AND NOT (node)--() "
                "DELETE node"
            ),
        },
        # 5. Write this document's edges.
        {
            "statement": (
                "UNWIND $edges AS item "
                "MATCH (source:GraphNode {id: item.source}) "
                "MATCH (target:GraphNode {id: item.target}) "
                "MERGE (source)-[relation:LEGAL_RELATION {id: item.id}]->(target) "
                "SET relation += item"
            ),
            "parameters": {"edges": prepared["edges"]},
        },
        # 6. Verify.
        {
            "statement": (
                "MATCH (node:GraphNode) "
                "WHERE node.document_id = $document_id "
                "   OR $document_id IN coalesce(node.document_ids, []) "
                "WITH count(node) AS nodes "
                "OPTIONAL MATCH ()-[relation:LEGAL_RELATION {document_id: $document_id}]->() "
                "RETURN nodes, count(relation) AS edges"
            ),
            "parameters": params,
            "resultDataContents": ["row"],
        },
    ]


def import_graph(
    session: requests.Session,
    endpoint: str,
    graph: dict[str, Any],
    timeout: int,
) -> tuple[str, list[int]]:
    doc_id = document_id(graph)
    prepared = prepare(graph, doc_id)

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
        timeout,
    )
    cases = [
        {"id": node["id"], "match_keys": node["match_keys"]}
        for node in prepared["shared"]
        if node["type"] == "CASE"
    ]
    rows: list[list[Any]] = []
    if cases:
        payload = execute(
            session,
            endpoint,
            [
                {
                    "statement": FIND_CASE_MATCHES,
                    "parameters": {"cases": cases},
                    "resultDataContents": ["row"],
                }
            ],
            timeout,
        )
        rows = [item["row"] for item in payload["results"][0]["data"]]
    merges = resolve_cases(prepared, rows, doc_id)

    payload = execute(session, endpoint, import_statements(prepared, merges, doc_id), timeout)
    row = payload["results"][-1]["data"][0]["row"]
    expected = [len(prepared["scoped"]) + len(prepared["shared"]), len(prepared["edges"])]
    if row != expected:
        raise RuntimeError(f"Neo4j verification mismatch: expected {expected}, got {row}")
    return doc_id, row


def main() -> int:
    args = parse_args()
    if not args.password:
        raise ValueError("NEO4J_PASSWORD is required")
    graph = json.loads(args.graph.read_text(encoding="utf-8"))

    endpoint = f"{args.url.rstrip('/')}/db/{args.database}/tx/commit"
    session = requests.Session()
    session.auth = (args.username, args.password)
    session.headers.update({"Accept": "application/json"})
    wait_for_neo4j(session, endpoint, args.timeout)

    started = time.perf_counter()
    doc_id, row = import_graph(session, endpoint, graph, args.timeout)
    elapsed = time.perf_counter() - started
    print(
        f"Imported and verified {row[0]} nodes and {row[1]} edges in Neo4j "
        f"for {doc_id} in {elapsed:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
