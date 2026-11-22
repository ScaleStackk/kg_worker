"""The answer-key scorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import score  # noqa: E402


def test_match_cases_by_citation_then_name():
    predicted = [
        {"name": "Reliance Cellulose v. ONGC", "citation": "(2018) 9 SCC 266"},
        {"name": "Pam Developments v. State of WB", "citation": ""},
        {"name": "Unrelated v. Someone", "citation": ""},
    ]
    gold = [
        {"name": "Reliance Cellulose Products Ltd. v. Oil and Natural Gas", "citation": "2018 (9) SCC 266"},
        {"name": "Pam Developments Private Limited v. State of West Bengal", "citation": "(2024) 10 SCC 715"},
        {"name": "G.C. Roy", "citation": "(1992) 1 SCC 508"},
    ]
    assert sorted(score.match_cases(predicted, gold)) == [(0, 0), (1, 1)]


def test_score_document_and_summary():
    record = {
        "case": {"court": "Supreme Court of India", "date": "2025-01-02",
                 "citations": ["2025 INSC 20"], "judges": ["C.T. Ravikumar", "Sanjay Karol"],
                 "author": "Sanjay Karol"},
        "cited_cases": [{"name": "Rajnesh v. Neha", "citation": "", "treatment": "RELIES_ON"}],
        "statutes": [{"name": "Senior Citizens Act", "sections": ["Section 23"]}],
    }
    gold = {
        "case": {"court": "Supreme Court of India", "date": "2025-01-02",
                 "citations": ["2025 INSC 20"], "judges": ["Sanjay Karol", "C. T. Ravikumar"],
                 "author": "Sanjay Karol"},
        "cited_cases": [
            {"name": "Rajnesh v. Neha", "citation": "(2021) 2 SCC 324", "treatment": "REFERRED_TO"},
            {"name": "Badshah v. Urmila", "citation": "(2014) 1 SCC 188", "treatment": "RELIES_ON"},
        ],
        "statutes": [{"name": "Senior Citizens Act, 2007", "sections": ["23"]}],
    }
    doc = score.score_document(record, {"validation": {"accepted_assertions": 3, "rejected_assertions": 1}}, gold)
    assert all(doc["header"].values())
    summary = score.summarise({"d": doc})
    assert summary["cited_cases"]["precision"] == 1.0
    assert summary["cited_cases"]["recall"] == 0.5
    assert summary["treatment"]["accuracy"] == 0.0
    assert summary["treatment"]["coarse_accuracy"] == 0.0
    assert summary["statutes"] == {"recall": 1.0, "section_recall": 1.0}
    assert summary["evidence_acceptance"] == 0.75
