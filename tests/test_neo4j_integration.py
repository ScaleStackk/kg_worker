"""Cross-document import against a real, empty Neo4j.

Skipped unless NEO4J_TEST_URL is set. It WIPES the database it points at:

    docker run -d --rm --name kg-test -p 127.0.0.1:17474:7474 \
        -e NEO4J_AUTH=neo4j/testpassword123 neo4j:2026.10.1
    NEO4J_TEST_URL=http://127.0.0.1:17474 NEO4J_TEST_PASSWORD=testpassword123 \
        .venv/bin/python -m pytest tests/test_neo4j_integration.py
"""

import os

import pytest
import requests

from import_neo4j import execute, import_graph
from kg_create import build_graph, empty_extraction

URL = os.environ.get("NEO4J_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="NEO4J_TEST_URL not set")


@pytest.fixture
def neo4j():
    session = requests.Session()
    session.auth = ("neo4j", os.environ.get("NEO4J_TEST_PASSWORD", "testpassword123"))
    endpoint = f"{URL.rstrip('/')}/db/neo4j/tx/commit"
    execute(session, endpoint, [{"statement": "MATCH (n) DETACH DELETE n"}], 30)

    def query(statement, **parameters):
        payload = execute(
            session,
            endpoint,
            [{"statement": statement, "parameters": parameters, "resultDataContents": ["row"]}],
            30,
        )
        return [item["row"] for item in payload["results"][0]["data"]]

    def load(graph):
        return import_graph(session, endpoint, graph, 30)

    yield load, query


def judgment(doc_id, name, citations=(), cited=(), statutes=()):
    extraction = empty_extraction()
    extraction["case"].update(
        {"name": name, "court": "Supreme Court of India", "citations": list(citations),
         "judges": ["Vikram Nath", "Sanjay Karol"]}
    )
    extraction["facts"] = [{"text": f"A fact in {name}", "evidence": "fact", "page": 1}]
    extraction["cited_cases"] = [
        {"name": n, "citation": c, "treatment": t, "evidence": f"{t} {n}", "page": 2}
        for n, c, t in cited
    ]
    extraction["statutes"] = [
        {"name": s, "sections": ["Section 302"], "treatment": "APPLIES", "evidence": f"applies {s}", "page": 3}
        for s in statutes
    ]
    nodes, edges = build_graph(extraction)
    return {"metadata": {"document_id": doc_id}, "nodes": nodes, "edges": edges}


DOC_A = judgment(
    "docA", "Ram v. Shyam", ["2025 INSC 1"],
    cited=[("State of Rajasthan v. Chatra", "2025 INSC 360", "FOLLOWS"),
           ("Old Case v. Union", "(1990) 2 SCC 10", "DISTINGUISHES")],
    statutes=["IPC"],
)
DOC_B = judgment(
    "docB", "State of Rajasthan v. Chatra", ["2025 INSC 360"],
    cited=[("Old Case v. Union of India", "(1990) 2 SCC 10", "RELIES_ON")],
    statutes=["Indian Penal Code, 1860"],
)


def counts(query):
    return query(
        "MATCH (n:GraphNode) RETURN n.type, count(*) ORDER BY n.type"
    )


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_citation_links_to_uploaded_judgment_in_either_order(neo4j, order):
    load, query = neo4j
    for name in order:
        load({"A": DOC_A, "B": DOC_B}[name])

    rows = query(
        "MATCH (a:GraphNode {judgment_document_id: 'docA'})"
        "-[r:LEGAL_RELATION]->(b:GraphNode {judgment_document_id: 'docB'}) "
        "RETURN r.relation, r.document_id, b.label"
    )
    assert rows == [["FOLLOWS", "docA", "State of Rajasthan v. Chatra"]]

    # One node per real-world entity, shared by both documents.
    shared = dict(
        query(
            "MATCH (n:GraphNode) WHERE n.type IN ['STATUTE', 'SECTION', 'JUDGE', 'COURT'] "
            "RETURN n.label, n.document_ids"
        )
    )
    assert len(shared) == 5  # IPC, s. 302, two judges, one court
    assert all(sorted(ids) == ["docA", "docB"] for ids in shared.values())

    old_case = query(
        "MATCH (n:GraphNode) WHERE 'scc:1990:2:10' IN n.match_keys "
        "MATCH (n)<-[r]-() RETURN count(DISTINCT n), collect(r.relation)"
    )
    assert old_case[0][0] == 1
    assert sorted(old_case[0][1]) == ["DISTINGUISHES", "RELIES_ON"]
    assert query("MATCH (n:GraphNode {type: 'CASE', is_current_case: true}) RETURN count(n)") == [[2]]


def test_reimport_is_idempotent_and_keeps_other_documents(neo4j):
    load, query = neo4j
    load(DOC_A)
    load(DOC_B)
    before = counts(query)
    edges_before = query("MATCH ()-[r]->() RETURN r.document_id, count(*) ORDER BY r.document_id")

    load(DOC_A)
    load(DOC_A)
    assert counts(query) == before
    assert query("MATCH ()-[r]->() RETURN r.document_id, count(*) ORDER BY r.document_id") == edges_before


def test_reimport_with_fewer_entities_removes_only_unused_shared_nodes(neo4j):
    load, query = neo4j
    load(DOC_A)
    load(DOC_B)
    # docA re-extracted without its statutes and citations.
    load(judgment("docA", "Ram v. Shyam", ["2025 INSC 1"]))

    ipc = query("MATCH (n:GraphNode {type: 'STATUTE'}) RETURN n.document_ids")
    assert ipc == [[["docB"]]]  # still used by docB
    assert query("MATCH (n:GraphNode {type: 'CASE'}) RETURN count(n)") == [[3]]
    assert query("MATCH (:GraphNode {judgment_document_id: 'docA'})-[r]->(:GraphNode {type: 'CASE'}) RETURN count(r)") == [[0]]
    assert query("MATCH (n:GraphNode) WHERE n.document_id IS NULL AND n.document_ids = [] RETURN count(n)") == [[0]]


def test_cited_only_case_removed_when_last_citing_document_is_reimported_without_it(neo4j):
    load, query = neo4j
    load(DOC_A)
    load(judgment("docA", "Ram v. Shyam", ["2025 INSC 1"]))
    assert query("MATCH (n:GraphNode {type: 'CASE'}) RETURN n.label") == [["Ram v. Shyam"]]


def test_legacy_scoped_nodes_are_replaced_on_reimport(neo4j):
    load, query = neo4j
    query(
        "CREATE (:GraphNode {id: 'docA:legacy', document_id: 'docA', type: 'CASE', label: 'old'})"
    )
    load(DOC_A)
    assert query("MATCH (n:GraphNode {id: 'docA:legacy'}) RETURN count(n)") == [[0]]
