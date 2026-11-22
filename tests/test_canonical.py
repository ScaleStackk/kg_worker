"""Name canonicalisation used for node keys."""

import pytest

from kg_create import (
    build_graph,
    canonical_key,
    case_key,
    judge_key,
    resolve_statute_keys,
    section_key,
    statute_key,
)


@pytest.mark.parametrize(
    "variants",
    [
        ["Income-tax Act, 1961", "Income Tax Act 1961", "the Income-Tax Act, 1961"],
        ["IPC", "Indian Penal Code, 1860", "Penal Code", "I.P.C."],
        ["Cr.P.C.", "CrPC", "Code of Criminal Procedure, 1973", "Criminal Procedure Code"],
        ["Wild Life (Protection) Act, 1972", "Wild Life Act, 1972"],
        ["Constitution of India", "the Constitution", "Indian Constitution"],
        ["Arbitration & Conciliation Act, 1996", "Arbitration and Conciliation Act 1996"],
    ],
)
def test_statute_variants_share_a_key(variants):
    assert len({statute_key(name) for name in variants}) == 1


def test_statutes_with_different_years_stay_apart():
    assert statute_key("Companies Act, 1956") != statute_key("Companies Act, 2013")


def test_yearless_statute_takes_the_only_year_in_the_judgment():
    keys = resolve_statute_keys(["Income-tax Act, 1961", "Income Tax Act"])
    assert keys["Income Tax Act"] == keys["Income-tax Act, 1961"] == "income tax act 1961"


def test_yearless_statute_ambiguous_year_is_left_alone():
    keys = resolve_statute_keys(["Companies Act, 1956", "Companies Act, 2013", "Companies Act"])
    assert keys["Companies Act"] == "companies act"


@pytest.mark.parametrize(
    "variants",
    [
        ["s. 10(2)", "Sec. 10(2)", "Section 10(2)", "section 10 (2)", "u/s 10(2)", "10(2)"],
        ["Section 482", "S. 482", "482"],
        ["Section 2(1)(d)", "s.2(1)(d)", "Section 2[1](d)"],
    ],
)
def test_section_variants_share_a_key(variants):
    assert len({section_key(value) for value in variants}) == 1


def test_sub_sections_are_kept_apart():
    assert section_key("s. 10(2)") != section_key("s. 10(3)")


@pytest.mark.parametrize(
    "variants",
    [
        ["Ram v. Shyam", "Ram vs. Shyam", "Ram versus Shyam", "Ram Vs Shyam"],
        [
            "M/s. Shakti Trading Co. Ltd. v. State of U.P. & Ors.",
            "Shakti Trading Company Limited vs State of Uttar Pradesh and others",
        ],
        ["CIT v. Walchand", "Commissioner of Income-tax v. Walchand"],
        ["Ram (Since Deceased) Through LRs v. Shyam", "Ram v. Shyam"],
    ],
)
def test_case_name_variants_share_a_key(variants):
    assert len({case_key(name) for name in variants}) == 1


def test_different_cases_stay_apart():
    assert case_key("Ram v. Shyam") != case_key("Shyam v. Ram")


@pytest.mark.parametrize(
    "variants",
    [
        ["S.R. Das", "S. R. Das", "SR Das", "Hon'ble Mr. Justice S.R. Das", "S.R. Das, J."],
        ["Pamidighantam Sri Narasimha", "Pamidighantamsrinarasimha"],
        ["Abhay S. Oka", "Abhay S Oka", "Abhay S.Oka"],
        ["Sanjiv Khanna, CJI", "Sanjiv Khanna"],
    ],
)
def test_judge_variants_share_a_key(variants):
    assert len({judge_key(name) for name in variants}) == 1


def test_canonical_key_court_and_fallback():
    assert canonical_key("COURT", "Supreme Court") == canonical_key("COURT", "The Supreme Court of India")
    assert canonical_key("FACT", "Some, fact!") == "some fact"


def minimal_record():
    return {
        "case": {
            "name": "Ram v. Shyam",
            "court": "Supreme Court of India",
            "date": "",
            "citations": [],
            "judges": ["S.R. Das", "Vikramnath"],
            "author": "S. R. Das",
            "parties": [],
        },
        "issues": [],
        "facts": [],
        "procedural_history": [],
        "holdings": [],
        "legal_rules": [],
        "cited_cases": [
            {"name": "Ram vs. Shyam", "citation": "", "treatment": "CITES", "evidence": "e1", "page": 1},
            {"name": "CIT v. Walchand", "citation": "", "treatment": "CITES", "evidence": "e2", "page": 1},
            {"name": "Commissioner of Income-tax v. Walchand", "citation": "", "treatment": "RELIES_ON", "evidence": "e3", "page": 2},
        ],
        "statutes": [
            {"name": "Income-tax Act, 1961", "sections": ["s. 10(2)"], "treatment": "INTERPRETS", "evidence": "e4", "page": 1},
            {"name": "Income Tax Act", "sections": ["Section 10(2)", "Sec. 37"], "treatment": "APPLIES", "evidence": "e5", "page": 2},
        ],
        "outcome": {"disposition": "", "evidence": "", "page": 0},
    }


def test_build_graph_merges_variants_and_keeps_first_label():
    nodes, edges = build_graph(minimal_record())
    labels = lambda node_type: sorted(n["label"] for n in nodes if n["type"] == node_type)
    assert labels("STATUTE") == ["Income-tax Act, 1961"]
    assert labels("SECTION") == ["Sec. 37 of Income Tax Act", "s. 10(2) of Income-tax Act, 1961"]
    assert labels("JUDGE") == ["S.R. Das", "Vikramnath"]
    assert labels("CASE") == ["CIT v. Walchand", "Ram v. Shyam"]


def test_current_case_flag_survives_self_citation():
    nodes, _ = build_graph(minimal_record())
    current = [n for n in nodes if n["type"] == "CASE" and n["label"] == "Ram v. Shyam"]
    assert current[0]["is_current_case"] is True
