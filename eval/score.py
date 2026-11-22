#!/usr/bin/env python3
"""Score pipeline output against the answer key in eval/gold/.

Usage:
    python eval/score.py <output-dir> [--gold eval/gold] [--json scores.json]

<output-dir> holds kg_create.py output files (<pdf stem>.json, with a
`case_record`). Only documents present in both directories are scored.
See eval/README.md for how the answer key was made.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_create import (  # noqa: E402
    TREATMENTS,
    case_key,
    citation_keys,
    distinctive_tokens,
    judge_key,
    normalize_key,
    section_key,
    statute_key,
)

# Coarse buckets: "cited only" vs "used to decide" vs "set aside".
COARSE = {
    "CITES": "MENTION",
    "REFERRED_TO": "MENTION",
    "RELIES_ON": "POSITIVE",
    "APPLIES": "POSITIVE",
    "FOLLOWS": "POSITIVE",
    "DISTINGUISHES": "NEGATIVE",
    "OVERRULES": "NEGATIVE",
}


def lead_tokens(name: str) -> set[str]:
    sides = distinctive_tokens(name)
    return set(sides[0]) if sides else set(normalize_key(name).split())


def same_case(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Match by citation when both have one, otherwise by name."""
    predicted_keys = set(citation_keys(str(predicted.get("citation", ""))))
    gold_keys = set(citation_keys(str(gold.get("citation", ""))))
    if predicted_keys and gold_keys:
        return bool(predicted_keys & gold_keys)
    if case_key(predicted["name"]) == case_key(gold["name"]):
        return True
    predicted_lead, gold_lead = lead_tokens(predicted["name"]), lead_tokens(gold["name"])
    return bool(predicted_lead and gold_lead) and (
        predicted_lead <= gold_lead or gold_lead <= predicted_lead
    )


def match_cases(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, citation matches first."""
    pairs: list[tuple[int, int]] = []
    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    for pass_number in (0, 1):
        for g_index, g in enumerate(gold):
            if g_index in used_gold:
                continue
            for p_index, p in enumerate(predicted):
                if p_index in used_predicted:
                    continue
                by_citation = bool(
                    set(citation_keys(str(p.get("citation", ""))))
                    & set(citation_keys(str(g.get("citation", ""))))
                )
                if (pass_number == 0 and by_citation) or (pass_number == 1 and same_case(p, g)):
                    pairs.append((p_index, g_index))
                    used_predicted.add(p_index)
                    used_gold.add(g_index)
                    break
    return pairs


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def score_document(record: dict[str, Any], metadata: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    case, gold_case = record["case"], gold["case"]
    header = {
        "court": normalize_key(case.get("court", "")) == normalize_key(gold_case["court"]),
        "date": case.get("date", "") == gold_case["date"],
        "citations": {re.sub(r"\s+", " ", c) for c in case.get("citations", [])}
        >= set(gold_case["citations"]),
        "judges": {judge_key(j) for j in case.get("judges", [])}
        == {judge_key(j) for j in gold_case["judges"]},
    }
    if gold_case.get("author"):
        header["author"] = judge_key(case.get("author", "")) == judge_key(gold_case["author"])

    result: dict[str, Any] = {"header": header}
    validation = metadata.get("validation", {})
    result["evidence"] = [
        validation.get("accepted_assertions", 0),
        validation.get("rejected_assertions", 0),
    ]

    if "cited_cases" in gold:
        predicted = record.get("cited_cases", [])
        pairs = match_cases(predicted, gold["cited_cases"])
        result["cited"] = {
            "predicted": len(predicted),
            "gold": len(gold["cited_cases"]),
            "matched": len(pairs),
            "treatments": [
                (gold["cited_cases"][g]["treatment"], predicted[p].get("treatment", "CITES"))
                for p, g in pairs
            ],
            "sources": Counter(predicted[p].get("treatment_source", "llm") for p, _ in pairs),
        }

    if "statutes" in gold:
        predicted_statutes = {statute_key(s["name"]) for s in record.get("statutes", [])}
        predicted_sections = {
            (statute_key(s["name"]), section_key(section))
            for s in record.get("statutes", [])
            for section in s.get("sections", [])
        }
        gold_statutes = [statute_key(s["name"]) for s in gold["statutes"]]

        def statute_found(key: str) -> bool:
            # A yearless prediction counts for the same Act with a year.
            return key in predicted_statutes or re.sub(r"\s\d{4}$", "", key) in predicted_statutes

        gold_sections = [
            (statute_key(s["name"]), section_key(section))
            for s in gold["statutes"]
            for section in s["sections"]
        ]
        result["statutes"] = {
            "gold": len(gold_statutes),
            "found": sum(statute_found(key) for key in gold_statutes),
            "gold_sections": len(gold_sections),
            "found_sections": sum(
                any(sec == p_sec and (p_stat == stat or p_stat == re.sub(r"\s\d{4}$", "", stat))
                    for p_stat, p_sec in predicted_sections)
                for stat, sec in gold_sections
            ),
        }
    return result


def summarise(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    header_fields = ["court", "date", "citations", "judges", "author"]
    header = {}
    for field in header_fields:
        values = [doc["header"][field] for doc in documents.values() if field in doc["header"]]
        header[field] = f"{sum(values)}/{len(values)}"

    accepted = sum(doc["evidence"][0] for doc in documents.values())
    rejected = sum(doc["evidence"][1] for doc in documents.values())

    cited_docs = [doc["cited"] for doc in documents.values() if "cited" in doc]
    predicted = sum(doc["predicted"] for doc in cited_docs)
    gold = sum(doc["gold"] for doc in cited_docs)
    matched = sum(doc["matched"] for doc in cited_docs)
    treatments = [pair for doc in cited_docs for pair in doc["treatments"]]
    sources: Counter[str] = Counter()
    for doc in cited_docs:
        sources.update(doc["sources"])

    confusion: dict[str, dict[str, int]] = {}
    for gold_label, predicted_label in treatments:
        row = confusion.setdefault(gold_label, {})
        row[predicted_label] = row.get(predicted_label, 0) + 1

    statute_docs = [doc["statutes"] for doc in documents.values() if "statutes" in doc]
    return {
        "documents": len(documents),
        "header": header,
        "evidence_acceptance": ratio(accepted, accepted + rejected),
        "cited_cases": {
            "documents": len(cited_docs),
            "precision": ratio(matched, predicted),
            "recall": ratio(matched, gold),
            "predicted": predicted,
            "gold": gold,
            "matched": matched,
        },
        "treatment": {
            "accuracy": ratio(sum(g == p for g, p in treatments), len(treatments)),
            "coarse_accuracy": ratio(
                sum(COARSE.get(g) == COARSE.get(p) for g, p in treatments), len(treatments)
            ),
            "matched_pairs": len(treatments),
            "source_of_matched": dict(sources),
            "confusion_gold_to_predicted": confusion,
        },
        "statutes": {
            "recall": ratio(sum(d["found"] for d in statute_docs), sum(d["gold"] for d in statute_docs)),
            "section_recall": ratio(
                sum(d["found_sections"] for d in statute_docs),
                sum(d["gold_sections"] for d in statute_docs),
            ),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"documents scored: {summary['documents']}")
    print("header fields correct: " + ", ".join(f"{k} {v}" for k, v in summary["header"].items()))
    print(f"evidence acceptance rate: {summary['evidence_acceptance']}")
    cited = summary["cited_cases"]
    print(
        f"cited cases ({cited['documents']} docs): precision {cited['precision']} "
        f"recall {cited['recall']} (matched {cited['matched']}, predicted {cited['predicted']}, "
        f"gold {cited['gold']})"
    )
    treatment = summary["treatment"]
    print(
        f"treatment on matched pairs: accuracy {treatment['accuracy']}, coarse {treatment['coarse_accuracy']} "
        f"(n={treatment['matched_pairs']}, sources {treatment['source_of_matched']})"
    )
    labels = sorted(TREATMENTS)
    print("confusion (rows gold, columns predicted):")
    print(" " * 14 + " ".join(f"{label[:6]:>6}" for label in labels))
    for gold_label in labels:
        row = treatment["confusion_gold_to_predicted"].get(gold_label, {})
        if row:
            print(f"{gold_label:>13} " + " ".join(f"{row.get(label, 0):>6}" for label in labels))
    statutes = summary["statutes"]
    print(f"statute recall {statutes['recall']}, section recall {statutes['section_recall']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gold", type=Path, default=Path(__file__).parent / "gold")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    documents = {}
    for gold_path in sorted(args.gold.glob("*.json")):
        output_path = args.output_dir / gold_path.name
        if not output_path.exists():
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        output = json.loads(output_path.read_text(encoding="utf-8"))
        documents[gold_path.stem] = score_document(output["case_record"], output.get("metadata", {}), gold)
    if not documents:
        print("No documents in common between output and gold.", file=sys.stderr)
        return 1

    summary = summarise(documents)
    print_summary(summary)
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "documents": documents}, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
