"""Cross-document import logic, with Neo4j's HTTP endpoint mocked."""

import pytest

import import_neo4j
from import_neo4j import import_graph, prepare, resolve_cases
from kg_create import build_graph, empty_extraction


def record(name, citations=(), cited=(), statutes=(), judges=("S.R. Das",)):
    extraction = empty_extraction()
    extraction["case"].update(
        {"name": name, "court": "Supreme Court of India", "citations": list(citations),
         "judges": list(judges)}
    )
    extraction["facts"] = [{"text": f"fact of {name}", "evidence": "q", "page": 1}]
    extraction["cited_cases"] = [
        {"name": n, "citation": c, "treatment": "RELIES_ON", "evidence": f"cites {n}", "page": 2}
        for n, c in cited
    ]
    extraction["statutes"] = [
        {"name": s, "sections": ["s. 10"], "treatment": "APPLIES", "evidence": f"applies {s}", "page": 3}
        for s in statutes
    ]
    return extraction


def graph(doc_id, extraction):
    nodes, edges = build_graph(extraction)
    return {"metadata": {"document_id": doc_id}, "nodes": nodes, "edges": edges}


def test_prepare_splits_shared_and_scoped_nodes():
    prepared = prepare(graph("docA", record("A v. B", statutes=["Income-tax Act, 1961"])), "docA")
    shared_types = sorted(node["type"] for node in prepared["shared"])
    scoped_types = sorted(node["type"] for node in prepared["scoped"])
    assert shared_types == ["CASE", "COURT", "JUDGE", "SECTION", "STATUTE"]
    assert scoped_types == ["FACT"]
    assert all(node["id"].startswith("shared:") for node in prepared["shared"])
    assert all(node["id"].startswith("docA:") for node in prepared["scoped"])
    assert all(node["document_id"] == "docA" for node in prepared["scoped"])
    assert all(edge["document_id"] == "docA" and edge["id"].startswith("docA:") for edge in prepared["edges"])


def test_same_entity_gets_same_shared_id_in_two_documents():
    a = prepare(graph("docA", record("A v. B", statutes=["Income-tax Act, 1961"])), "docA")
    b = prepare(graph("docB", record("C v. D", statutes=["Income Tax Act, 1961"])), "docB")
    ids = lambda p, t: {n["id"] for n in p["shared"] if n["type"] == t}
    for node_type in ["STATUTE", "SECTION", "JUDGE", "COURT"]:
        assert ids(a, node_type) == ids(b, node_type)
    assert ids(a, "CASE") != ids(b, "CASE")
    assert not {n["id"] for n in a["scoped"]} & {n["id"] for n in b["scoped"]}


def test_cited_case_with_citation_matches_uploaded_judgment_by_neutral_citation():
    uploaded = prepare(graph("docB", record("Chatra v. State", citations=["2025 INSC 360"])), "docB")
    citing = prepare(
        graph("docA", record("A v. B", cited=[("State of Rajasthan v. Chatra", "2025 INSC 360")])),
        "docA",
    )
    current = next(n for n in uploaded["shared"] if n["is_current_case"])
    cited = next(n for n in citing["shared"] if n["type"] == "CASE" and not n["is_current_case"])
    assert "insc:2025:360" in current["match_keys"]
    assert "insc:2025:360" in cited["match_keys"]


def case_nodes(prepared):
    return [n for n in prepared["shared"] if n["type"] == "CASE"]


def test_resolve_cases_points_citation_at_uploaded_judgment():
    prepared = prepare(
        graph("docA", record("A v. B", cited=[("Chatra", "2025 INSC 360")])), "docA"
    )
    cited = next(n for n in case_nodes(prepared) if not n["is_current_case"])
    old_id = cited["id"]
    rows = [[old_id, "shared:judgmentB", "docB"]]
    merges = resolve_cases(prepared, rows, "docA")
    assert merges == []
    assert cited["id"] == "shared:judgmentB"
    assert any(edge["target"] == "shared:judgmentB" for edge in prepared["edges"])
    assert not any(edge["target"] == old_id for edge in prepared["edges"])


def test_resolve_cases_folds_earlier_cited_nodes_into_new_judgment():
    prepared = prepare(graph("docB", record("Chatra v. State", citations=["2025 INSC 360"])), "docB")
    current = next(n for n in case_nodes(prepared) if n["is_current_case"])
    rows = [
        [current["id"], "shared:citedByName", None],
        [current["id"], "shared:citedByCitation", None],
    ]
    merges = resolve_cases(prepared, rows, "docB")
    # One existing node is reused for the judgment; the other is folded into it.
    assert merges == [{"old": "shared:citedByName", "keep": "shared:citedByCitation"}]
    assert current["id"] == "shared:citedByCitation"


def test_resolve_cases_never_merges_two_uploaded_judgments():
    prepared = prepare(graph("docB", record("Ram v. Shyam")), "docB")
    current = next(n for n in case_nodes(prepared) if n["is_current_case"])
    original = current["id"]
    merges = resolve_cases(prepared, [[original, "shared:otherJudgment", "docA"]], "docB")
    assert merges == []
    assert current["id"] == original


def test_resolve_cases_reimport_keeps_own_node():
    prepared = prepare(graph("docB", record("Ram v. Shyam")), "docB")
    current = next(n for n in case_nodes(prepared) if n["is_current_case"])
    original = current["id"]
    merges = resolve_cases(prepared, [[original, original, "docB"]], "docB")
    assert merges == []
    assert current["id"] == original


def test_resolve_cases_ambiguous_citation_is_left_alone():
    prepared = prepare(graph("docA", record("A v. B", cited=[("X v. Y", "2025 INSC 9")])), "docA")
    cited = next(n for n in case_nodes(prepared) if not n["is_current_case"])
    original = cited["id"]
    rows = [[original, "shared:j1", "doc1"], [original, "shared:j2", "doc2"]]
    assert resolve_cases(prepared, rows, "docA") == []
    assert cited["id"] == original


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Records statements; answers the case lookup and the verification query."""

    def __init__(self, matches=()):
        self.matches = list(matches)
        self.requests = []

    def post(self, endpoint, json, timeout):
        statements = json["statements"]
        self.requests.append(statements)
        results = []
        for statement in statements:
            text = statement["statement"]
            if text == import_neo4j.FIND_CASE_MATCHES:
                rows = [{"row": row} for row in self.matches]
                results.append({"data": rows})
            elif text.startswith("MATCH (node:GraphNode) WHERE node.document_id = $document_id"):
                results.append({"data": [{"row": self.expected}]})
            else:
                results.append({"data": []})
        return FakeResponse({"results": results, "errors": []})


def test_import_graph_sends_scoped_delete_and_shared_upserts():
    data = graph("docA", record("A v. B", cited=[("C v. D", "")], statutes=["IPC"]))
    prepared = prepare(data, "docA")
    session = FakeSession()
    session.expected = [len(prepared["scoped"]) + len(prepared["shared"]), len(prepared["edges"])]

    doc_id, row = import_graph(session, "http://neo4j/db/neo4j/tx/commit", data, 5)

    assert doc_id == "docA" and row == session.expected
    constraint, lookup, main = session.requests
    assert "CREATE CONSTRAINT" in constraint[0]["statement"]
    assert lookup[0]["statement"] == import_neo4j.FIND_CASE_MATCHES
    texts = [s["statement"] for s in main]
    # Deletion is scoped to this document; shared nodes are only unlinked.
    assert texts[0].startswith("MATCH ()-[relation:LEGAL_RELATION {document_id: $document_id}]")
    assert texts[1].startswith("MATCH (node:GraphNode {document_id: $document_id}) DETACH DELETE")
    assert "SET node.document_ids = [d IN node.document_ids WHERE d <> $document_id]" in texts[2]
    assert not any("DETACH DELETE node" in t and "document_ids" in t for t in texts)
    shared_upsert = next(s for s in main if "node.document_ids = CASE" in s["statement"])
    assert {n["type"] for n in shared_upsert["parameters"]["nodes"]} >= {"CASE", "STATUTE", "JUDGE"}


def test_import_graph_raises_on_verification_mismatch():
    data = graph("docA", record("A v. B"))
    session = FakeSession()
    session.expected = [0, 0]
    with pytest.raises(RuntimeError, match="verification mismatch"):
        import_graph(session, "http://neo4j/db/neo4j/tx/commit", data, 5)
