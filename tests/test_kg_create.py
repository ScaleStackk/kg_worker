"""Unit tests for the pure parts of kg_create.py. No Ollama or Neo4j needed."""

import json
from pathlib import Path

import pymupdf
import pytest
import requests

import kg_create
from kg_create import (
    batch_pages,
    build_graph,
    clean_page_text,
    deduplicate,
    empty_extraction,
    find_source_quote,
    merge_extractions,
    normalize_key,
    parse_header_metadata,
    validate_extraction,
)


def page(number, text, raw_text=None):
    return {"page": number, "text": text, "raw_text": raw_text or text}


def raw_page(number, raw_text):
    return {"page": number, "text": clean_page_text(raw_text), "raw_text": raw_text}


# Old SCR headnote layout, as found on pre-2000 Indian Kanoon judgments.
OLD_HEADER = """Commissioner Of Income-Tax, Bombay vs Ahmedabad Mill Owners on 1 March, 1955
Equivalent citations: 1955 AIR 352, 1955 SCR (1) 829
Author: S.R. Das
Bench: S.R. Das, N.H. Bhagwati
PETITIONER:
COMMISSIONER OF INCOME-TAX, BOMBAY
Vs.
RESPONDENT:
AHMEDABAD MILL OWNERS
DATE OF JUDGMENT:
01/03/1955
BENCH:
DAS, SUDHI RANJAN
CITATION:
 1955 AIR 352            1955 SCR  (1) 829
 1955 SCC (2) 101
CITATOR INFO :
 R          1961 SC 1 (5)
ACT:
Income-tax Act, s. 10(2).
"""

OLD_BODY = (
    "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 12 of 1954. "
    "Appeals by special leave from the judgment and order of the Bombay High Court."
)


# --- normalize_key ----------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Income-Tax Act, 1961", "income tax act 1961"),
        ("  Section   10(2) ", "section 10"),
        ("Smith & Sons", "smith and sons"),
        ("A.B.C.", "a b c"),
        ("ÄBC", "bc"),
        ("", ""),
    ],
)
def test_normalize_key(value, expected):
    assert normalize_key(value) == expected


def test_normalize_key_drops_parenthesised_numbers():
    # Current behaviour: sub-section numbers are discarded, so "10(2)" and
    # "10(3)" collide. Kept here as a characterisation test.
    assert normalize_key("s. 10(2)") == normalize_key("s. 10(3)")


# --- clean_page_text --------------------------------------------------------


def test_clean_page_text_removes_indian_kanoon_footer():
    raw = (
        "1. Leave granted.\n"
        "   The appeal   is allowed.\n"
        "X vs Y on 10 February, 2025\n"
        "Indian Kanoon - http://indiankanoon.org/doc/1/\n"
        "1\n"
        "Next page text.\n"
    )
    assert clean_page_text(raw) == "1. Leave granted. The appeal is allowed. Next page text."


def test_clean_page_text_joins_hyphenated_words():
    assert clean_page_text("income-\ntax was paid") == "income-tax was paid"


def test_clean_page_text_keeps_digit_lines_not_after_footer():
    assert clean_page_text("Para\n12\nfollows") == "Para 12 follows"


# --- parse_header_metadata (old SCR layout) ---------------------------------


def test_parse_header_metadata_old_layout():
    pages = [raw_page(1, OLD_HEADER), page(2, OLD_BODY)]
    header = parse_header_metadata(pages)

    assert header["name"] == "Commissioner of Income-Tax, Bombay v. Ahmedabad Mill Owners"
    assert header["date"] == "1955-03-01"
    assert header["court"] == "Supreme Court of India"
    assert header["author"] == "S.R. Das"
    assert header["judges"] == ["S.R. Das", "N.H. Bhagwati"]
    assert header["citations"] == ["1955 AIR 352", "1955 SCR (1) 829", "1955 SCC (2) 101"]
    assert [(p["role"], p["name"]) for p in header["parties"]] == [
        ("PETITIONER", "Commissioner of Income-Tax, Bombay"),
        ("RESPONDENT", "Ahmedabad Mill Owners"),
    ]
    assert header["source_title"].startswith("Commissioner Of Income-Tax, Bombay vs")


def test_parse_header_metadata_strips_represented_by():
    raw = OLD_HEADER.replace(
        "COMMISSIONER OF INCOME-TAX, BOMBAY\nVs.",
        "COMMISSIONER OF INCOME-TAX, BOMBAYREPRESENTEDBY SOMEONE\nVs.",
    )
    header = parse_header_metadata([raw_page(1, raw)])
    assert header["parties"][0]["name"] == "Commissioner of Income-Tax, Bombay"


def test_parse_header_metadata_without_markers_never_uses_page_text_as_name():
    header = parse_header_metadata([raw_page(1, "Some judgment text without a header")])
    assert header["name"] == ""
    assert header["source_title"] == "Some judgment text without a header"
    assert header["court"] == ""
    assert header["date"] == ""
    assert header["citations"] == []
    assert header["parties"] == []


# --- batch_pages ------------------------------------------------------------


def test_batch_pages_groups_whole_pages():
    pages = [page(i, "x" * 40) for i in range(1, 6)]
    batches = batch_pages(pages, max_chars=100)
    assert [[p["page"] for p in batch] for batch in batches] == [[1, 2], [3, 4], [5]]


def test_batch_pages_keeps_oversized_page_alone():
    pages = [page(1, "x" * 10), page(2, "x" * 500), page(3, "x" * 10)]
    batches = batch_pages(pages, max_chars=100)
    assert [[p["page"] for p in batch] for batch in batches] == [[1], [2], [3]]


def test_batch_pages_empty():
    assert batch_pages([], max_chars=100) == []


# --- find_source_quote ------------------------------------------------------

SOURCE_PAGES = [
    page(1, "The appellant filed a return of income for the assessment year 1950-51."),
    page(
        2,
        "We are of the opinion that the roots ,of the tree were not part of the "
        "business. The appeal is accordingly allowed with costs.",
    ),
    page(3, "The appeal is accordingly allowed with costs."),
]


def test_find_source_quote_exact():
    assert find_source_quote("filed a return of income", 1, SOURCE_PAGES) == (
        "filed a return of income",
        1,
    )


def test_find_source_quote_case_insensitive_returns_source_casing():
    assert find_source_quote("THE APPELLANT FILED A RETURN", 1, SOURCE_PAGES) == (
        "The appellant filed a return",
        1,
    )


def test_find_source_quote_prefers_requested_page():
    quote = "The appeal is accordingly allowed with costs."
    assert find_source_quote(quote, 3, SOURCE_PAGES)[1] == 3
    assert find_source_quote(quote, 2, SOURCE_PAGES)[1] == 2


def test_find_source_quote_falls_back_to_other_page():
    assert find_source_quote("filed a return of income", 7, SOURCE_PAGES)[1] == 1


def test_find_source_quote_strips_wrapping_quotes():
    assert find_source_quote("“filed a return of income”", 1, SOURCE_PAGES) == (
        "filed a return of income",
        1,
    )


def test_find_source_quote_tolerates_punctuation_drift():
    assert find_source_quote("the roots of the tree were not", 2, SOURCE_PAGES) == (
        "the roots ,of the tree were not",
        2,
    )


def test_find_source_quote_ellipsis_validates_longest_segment():
    quote = "We are of the opinion that the roots ... allowed with costs"
    text, page_number = find_source_quote(quote, 2, SOURCE_PAGES)
    assert text == "We are of the opinion that the roots"
    assert page_number == 2


def test_find_source_quote_unicode_ellipsis():
    quote = "We are of the opinion that the roots … with costs"
    assert find_source_quote(quote, 2, SOURCE_PAGES)[0] == "We are of the opinion that the roots"


def test_find_source_quote_ellipsis_with_short_segments_rejected():
    assert find_source_quote("the roots ... the tree", 2, SOURCE_PAGES) is None


def test_find_source_quote_tolerates_injected_footnote():
    source = [
        page(
            1,
            "The arbitral tribunal held that both respondents 2 Bombay Stock Exchange "
            "were jointly and severally liable for the debit balance in the account.",
        )
    ]
    quote = (
        "The arbitral tribunal held that both respondents were jointly and "
        "severally liable for the debit balance in the account"
    )
    text, page_number = find_source_quote(quote, 1, source)
    assert text.startswith("The arbitral tribunal held")
    assert "Bombay Stock Exchange" in text
    assert text.endswith("account")
    assert page_number == 1


def test_find_source_quote_rejects_heavily_altered_quote():
    source = [
        page(
            1,
            "The arbitral tribunal held that both respondents were jointly and "
            "severally liable for the debit balance in the account.",
        )
    ]
    quote = (
        "The arbitral tribunal found that neither respondent was in any way "
        "responsible for any balance in the account"
    )
    assert find_source_quote(quote, 1, source) is None


@pytest.mark.parametrize("evidence", [None, 12, "", "short", "The court held that elephants fly"])
def test_find_source_quote_rejects(evidence):
    assert find_source_quote(evidence, 1, SOURCE_PAGES) is None


# --- deduplicate ------------------------------------------------------------


def test_deduplicate_by_normalized_fields_keeps_first():
    items = [
        {"name": "Income-Tax Act", "n": 1},
        {"name": "income tax act", "n": 2},
        {"name": "Wealth Tax Act", "n": 3},
    ]
    assert [item["n"] for item in deduplicate(items, ["name"])] == [1, 3]


def test_deduplicate_multiple_fields():
    items = [
        {"stage": "High Court", "decision": "Dismissed"},
        {"stage": "High Court", "decision": "Allowed"},
        {"stage": "high court", "decision": "dismissed"},
    ]
    assert len(deduplicate(items, ["stage", "decision"])) == 2


# --- merge_extractions ------------------------------------------------------


def test_merge_extractions_first_scalar_wins_and_arrays_concatenate():
    first = empty_extraction()
    first["case"]["court"] = "Supreme Court"
    first["facts"] = [{"text": "a"}]
    second = empty_extraction()
    second["case"]["court"] = "High Court"
    second["case"]["date"] = "2025-01-01"
    second["facts"] = [{"text": "b"}]
    second["outcome"] = {"disposition": "Allowed", "evidence": "x", "page": 2}

    merged = merge_extractions([first, second, {}])
    assert merged["case"]["court"] == "Supreme Court"
    assert merged["case"]["date"] == "2025-01-01"
    assert merged["facts"] == [{"text": "a"}, {"text": "b"}]
    assert merged["outcome"]["disposition"] == "Allowed"


# --- validate_extraction ----------------------------------------------------

VALIDATION_PAGES = [
    page(
        1,
        "The assessee claimed a deduction under section 10(2) of the Income-tax Act. "
        "Reliance was placed on Commissioner of Income-tax v. Maharajadhiraja Darbhanga "
        "reported in 1941 AIR 101.",
    ),
    page(2, "For these reasons the appeal is dismissed with costs."),
]

HEADER = {
    "name": "Header Case v. Union",
    "court": "Supreme Court of India",
    "date": "1955-03-01",
    "citations": ["1955 SCR (1) 829"],
    "judges": ["S.R. Das"],
    "author": "S.R. Das",
    "parties": [
        {"name": "Header Case", "role": "PETITIONER", "evidence": "Header Case", "page": 1}
    ],
}


def model_extraction():
    extraction = empty_extraction()
    extraction["case"].update(
        {
            "name": "Model Name",
            "court": "Model Court",
            "citations": ["1955 SCR (1) 829", "1955 AIR 352"],
            "judges": ["S.R. Das", "Bhagwati"],
            "parties": [
                {
                    "name": "  The   Assessee ",
                    "role": "APPELLANT",
                    "evidence": "The assessee claimed a deduction",
                    "page": 1,
                },
                {"name": "Ghost", "role": "OTHER", "evidence": "not in the text at all", "page": 1},
            ],
        }
    )
    extraction["facts"] = [
        {"text": "Deduction claimed", "evidence": "claimed a deduction under section", "page": 2},
        {"text": "Made up", "evidence": "the assessee was a shipping company", "page": 1},
        {"text": "", "evidence": "claimed a deduction", "page": 1},
        "not a dict",
    ]
    extraction["cited_cases"] = [
        {
            "name": "Commissioner of Income-tax v. Maharajadhiraja Darbhanga",
            "citation": "1941 AIR 101",
            "treatment": "REFERRED_TO",
            "proposition": "",
            "evidence": "Reliance was placed on Commissioner of Income-tax",
            "page": 1,
        },
        {
            "name": "Unrelated Shipping Case",
            "citation": "1941 AIR 101",
            "treatment": "CITES",
            "proposition": "",
            "evidence": "reported in 1941 AIR 101",
            "page": 1,
        },
    ]
    extraction["statutes"] = [
        {
            "name": "Income-tax Act",
            "sections": ["10(2)"],
            "treatment": "INTERPRETS",
            "evidence": "section 10(2) of the Income-tax Act",
            "page": 1,
        },
        {
            "name": "income tax act",
            "sections": [],
            "treatment": "MENTIONS",
            "evidence": "the Income-tax Act",
            "page": 1,
        },
    ]
    extraction["outcome"] = {
        "disposition": "Dismissed",
        "evidence": "the appeal is dismissed with costs",
        "page": 2,
    }
    return extraction


def test_validate_extraction_header_overrides_model_case_fields():
    record, _ = validate_extraction(model_extraction(), VALIDATION_PAGES, HEADER)
    case = record["case"]
    assert case["name"] == "Header Case v. Union"
    assert case["court"] == "Supreme Court of India"
    assert case["citations"] == ["1955 SCR (1) 829", "1955 AIR 352"]
    assert case["judges"] == ["S.R. Das", "Bhagwati"]


def test_validate_extraction_uses_model_fields_when_header_empty():
    header = dict(HEADER, name="", court="", date="", author="", parties=[])
    record, _ = validate_extraction(model_extraction(), VALIDATION_PAGES, header)
    assert record["case"]["name"] == "Model Name"
    assert record["case"]["court"] == "Model Court"


def test_validate_extraction_accepts_only_evidenced_items():
    record, stats = validate_extraction(model_extraction(), VALIDATION_PAGES, HEADER)

    assert [p["name"] for p in record["case"]["parties"]] == ["Header Case", "The Assessee"]
    assert [f["text"] for f in record["facts"]] == ["Deduction claimed"]
    # Evidence is replaced with the exact source text and the page it was found on.
    assert record["facts"][0]["page"] == 1
    assert record["outcome"] == {
        "disposition": "Dismissed",
        "evidence": "the appeal is dismissed with costs",
        "page": 2,
    }
    # header party 1 + model party 1 + fact 1 + cited 2 + statutes 2 + outcome 1
    assert stats == {"accepted_assertions": 8, "rejected_assertions": 4}


def test_validate_extraction_deduplicates_statutes():
    record, _ = validate_extraction(model_extraction(), VALIDATION_PAGES, HEADER)
    assert [s["name"] for s in record["statutes"]] == ["Income-tax Act"]


def test_validate_extraction_drops_citation_not_near_case_name():
    record, _ = validate_extraction(model_extraction(), VALIDATION_PAGES, HEADER)
    citations = {c["name"]: c["citation"] for c in record["cited_cases"]}
    assert citations["Commissioner of Income-tax v. Maharajadhiraja Darbhanga"] == "1941 AIR 101"
    assert citations["Unrelated Shipping Case"] == ""


def test_validate_extraction_rejects_outcome_without_evidence():
    extraction = model_extraction()
    extraction["outcome"]["evidence"] = "the appeal is allowed in part"
    record, stats = validate_extraction(extraction, VALIDATION_PAGES, HEADER)
    assert record["outcome"]["disposition"] == ""
    assert stats["rejected_assertions"] == 5


# --- build_graph ------------------------------------------------------------


def graph_record():
    record, _ = validate_extraction(model_extraction(), VALIDATION_PAGES, HEADER)
    return record


def by_type(nodes, node_type):
    return [node for node in nodes if node["type"] == node_type]


def test_build_graph_nodes():
    nodes, _ = build_graph(graph_record())
    current = [n for n in by_type(nodes, "CASE") if n.get("is_current_case")]
    assert len(current) == 1
    assert current[0]["label"] == "Header Case v. Union"
    assert current[0]["citations"] == ["1955 SCR (1) 829", "1955 AIR 352"]
    assert [n["label"] for n in by_type(nodes, "COURT")] == ["Supreme Court of India"]
    # Author and bench member with the same name collapse into one JUDGE node.
    assert sorted(n["label"] for n in by_type(nodes, "JUDGE")) == ["Bhagwati", "S.R. Das"]
    assert [n["label"] for n in by_type(nodes, "SECTION")] == ["10(2) of Income-tax Act"]
    assert len(by_type(nodes, "FACT")) == 1
    assert len(by_type(nodes, "OUTCOME")) == 1


def test_build_graph_edges():
    nodes, edges = build_graph(graph_record())
    labels = {node["id"]: (node["type"], node["label"]) for node in nodes}
    relations = {
        (labels[e["source"]][0], e["relation"], labels[e["target"]][0]) for e in edges
    }
    assert ("CASE", "DECIDED_AT", "COURT") in relations
    assert ("CASE", "DECIDED_BY", "JUDGE") in relations
    assert ("CASE", "AUTHORED_BY", "JUDGE") in relations
    assert ("CASE", "HAS_PARTY", "PARTY") in relations
    assert ("CASE", "HAS_MATERIAL_FACT", "FACT") in relations
    assert ("CASE", "INTERPRETS", "STATUTE") in relations
    assert ("CASE", "INTERPRETS", "SECTION") in relations
    assert ("SECTION", "PART_OF", "STATUTE") in relations
    assert ("CASE", "HAS_OUTCOME", "OUTCOME") in relations
    for edge in edges:
        assert edge["source"] in labels and edge["target"] in labels


def test_build_graph_is_deterministic():
    assert build_graph(graph_record()) == build_graph(graph_record())


def test_build_graph_coerces_unknown_treatments():
    record = graph_record()
    record["cited_cases"][0]["treatment"] = "LOVES"
    record["statutes"][0]["treatment"] = "IGNORES"
    _, edges = build_graph(record)
    relations = {e["relation"] for e in edges}
    assert "LOVES" not in relations and "IGNORES" not in relations
    assert "CITES" in relations and "MENTIONS" in relations


def test_build_graph_skips_empty_court_and_outcome():
    record = graph_record()
    record["case"]["court"] = ""
    record["outcome"] = {"disposition": "", "evidence": "", "page": 0}
    nodes, _ = build_graph(record)
    assert not by_type(nodes, "COURT")
    assert not by_type(nodes, "OUTCOME")


# --- call_ollama (mocked HTTP) ----------------------------------------------


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


def test_call_ollama_sends_schema_and_parses_response(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return FakeResponse({"response": '{"facts": []}'})

    monkeypatch.setattr(kg_create.requests, "post", fake_post)
    result = kg_create.call_ollama("prompt", "qwen3:4b", "http://ollama/api/generate", 4096)

    assert result == {"facts": []}
    url, body = calls[0]
    assert url == "http://ollama/api/generate"
    assert body["format"] == kg_create.EXTRACTION_SCHEMA
    assert body["think"] is False and body["stream"] is False
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["temperature"] == 0


def test_call_ollama_retries_then_succeeds(monkeypatch):
    responses = iter([FakeResponse({"response": "not json"}), FakeResponse({"response": "{}"})])
    monkeypatch.setattr(kg_create.requests, "post", lambda *a, **k: next(responses))
    assert kg_create.call_ollama("p", "m", "u", 1) == {}


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, status=500),
        FakeResponse({"no_response": ""}),
        FakeResponse({"response": "[1, 2]"}),
    ],
)
def test_call_ollama_raises_after_attempts(monkeypatch, response):
    monkeypatch.setattr(kg_create.requests, "post", lambda *a, **k: response)
    with pytest.raises(RuntimeError, match="Ollama extraction failed"):
        kg_create.call_ollama("p", "m", "u", 1, attempts=2)


# --- create_knowledge_graph end to end (mocked model) -----------------------


def write_pdf(path: Path, pages_text: list[str]) -> None:
    document = pymupdf.open()
    for text in pages_text:
        pdf_page = document.new_page()
        pdf_page.insert_text((72, 72), text, fontsize=9)
    document.save(path)
    document.close()


def test_create_knowledge_graph_with_mocked_model(tmp_path, monkeypatch):
    pdf_path = tmp_path / "judgment.pdf"
    write_pdf(pdf_path, [OLD_HEADER, OLD_BODY + "\nThe appeal is dismissed with costs."])

    prompts = []

    def fake_call(prompt, model, url, num_ctx):
        prompts.append(prompt)
        extraction = empty_extraction()
        extraction["outcome"] = {
            "disposition": "Dismissed",
            "evidence": "The appeal is dismissed with costs.",
            "page": 2,
        }
        return extraction

    monkeypatch.setattr(kg_create, "call_ollama", fake_call)
    output, raw = kg_create.create_knowledge_graph(pdf_path, "m", "u", 1024)

    assert len(prompts) == 1 and len(raw) == 1
    assert "CURRENT CASE: Commissioner of Income-Tax, Bombay v. Ahmedabad Mill Owners" in prompts[0]
    assert output["metadata"]["page_count"] == 2
    assert output["metadata"]["validation"]["accepted_assertions"] == 3
    assert output["case_record"]["outcome"]["page"] == 2
    assert {n["type"] for n in output["nodes"]} >= {"CASE", "COURT", "JUDGE", "PARTY", "OUTCOME"}
    json.dumps(output)  # must be serialisable


def test_create_knowledge_graph_rejects_textless_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    write_pdf(pdf_path, [""])
    with pytest.raises(ValueError, match="no extractable text"):
        kg_create.create_knowledge_graph(pdf_path, "m", "u", 1024)


# --- extraction schema ------------------------------------------------------


def walk_schema(schema, path="$"):
    yield path, schema
    for name, child in schema.get("properties", {}).items():
        yield from walk_schema(child, f"{path}.{name}")
    if "items" in schema:
        yield from walk_schema(schema["items"], f"{path}[]")


def test_every_free_text_string_and_array_in_schema_is_bounded():
    for path, node in walk_schema(kg_create.EXTRACTION_SCHEMA):
        if node.get("type") == "string" and "enum" not in node:
            assert node.get("maxLength"), path
        if node.get("type") == "array":
            assert node.get("maxItems"), path


def test_truncated_quote_with_junk_token_still_validates():
    # What a maxLength cut can produce: the quote stops mid-word on a stray token.
    quote = "The appellant filed a return of income for the assessment yea成为"
    assert find_source_quote(quote, 1, SOURCE_PAGES) == (
        "The appellant filed a return of income for the assessment year",
        1,
    )


def test_partial_last_word_must_still_be_a_prefix():
    assert find_source_quote("The appellant filed a return of incoxe", 1, SOURCE_PAGES) is None
