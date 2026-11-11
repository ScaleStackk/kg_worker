"""Build an evidence-grounded legal knowledge graph from a judgment PDF.

The LLM extracts a compact legal case record. Python validates every semantic
assertion against the source and deterministically turns the record into a
property graph. This keeps the task reliable enough for a small local model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf
import requests

from document_identity import resolve_document_identity


DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://127.0.0.1:11434/api/generate"
)
DEFAULT_MODEL = os.environ.get("KG_MODEL", "qwen3:4b")
REQUEST_TIMEOUT = int(os.environ.get("KG_REQUEST_TIMEOUT", "900"))
MAX_BATCH_CHARS = int(os.environ.get("KG_MAX_BATCH_CHARS", "24000"))
DEFAULT_NUM_CTX = int(os.environ.get("KG_NUM_CTX", "16384"))

TREATMENTS = {
    "CITES",
    "REFERRED_TO",
    "RELIES_ON",
    "APPLIES",
    "FOLLOWS",
    "DISTINGUISHES",
    "OVERRULES",
}
STATUTE_TREATMENTS = {"MENTIONS", "APPLIES", "INTERPRETS"}


def object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


EVIDENCED_TEXT_SCHEMA = object_schema(
    {
        "text": {"type": "string"},
        "evidence": {"type": "string"},
        "page": {"type": "integer"},
    },
    ["text", "evidence", "page"],
)

EXTRACTION_SCHEMA = object_schema(
    {
        "case": object_schema(
            {
                "name": {"type": "string"},
                "court": {"type": "string"},
                "date": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "judges": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string"},
                "parties": {
                    "type": "array",
                    "items": object_schema(
                        {
                            "name": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "PETITIONER",
                                    "RESPONDENT",
                                    "APPELLANT",
                                    "ASSESSEE",
                                    "REVENUE",
                                    "OTHER",
                                ],
                            },
                            "evidence": {"type": "string"},
                            "page": {"type": "integer"},
                        },
                        ["name", "role", "evidence", "page"],
                    ),
                },
            },
            ["name", "court", "date", "citations", "judges", "author", "parties"],
        ),
        "issues": {"type": "array", "items": EVIDENCED_TEXT_SCHEMA},
        "facts": {"type": "array", "items": EVIDENCED_TEXT_SCHEMA},
        "procedural_history": {
            "type": "array",
            "items": object_schema(
                {
                    "stage": {"type": "string"},
                    "decision": {"type": "string"},
                    "evidence": {"type": "string"},
                    "page": {"type": "integer"},
                },
                ["stage", "decision", "evidence", "page"],
            ),
        },
        "holdings": {"type": "array", "items": EVIDENCED_TEXT_SCHEMA},
        "legal_rules": {"type": "array", "items": EVIDENCED_TEXT_SCHEMA},
        "cited_cases": {
            "type": "array",
            "items": object_schema(
                {
                    "name": {"type": "string"},
                    "citation": {"type": "string"},
                    "treatment": {"type": "string", "enum": sorted(TREATMENTS)},
                    "proposition": {"type": "string"},
                    "evidence": {"type": "string"},
                    "page": {"type": "integer"},
                },
                ["name", "citation", "treatment", "proposition", "evidence", "page"],
            ),
        },
        "statutes": {
            "type": "array",
            "items": object_schema(
                {
                    "name": {"type": "string"},
                    "sections": {"type": "array", "items": {"type": "string"}},
                    "treatment": {
                        "type": "string",
                        "enum": sorted(STATUTE_TREATMENTS),
                    },
                    "evidence": {"type": "string"},
                    "page": {"type": "integer"},
                },
                ["name", "sections", "treatment", "evidence", "page"],
            ),
        },
        "outcome": object_schema(
            {
                "disposition": {"type": "string"},
                "evidence": {"type": "string"},
                "page": {"type": "integer"},
            },
            ["disposition", "evidence", "page"],
        ),
    },
    [
        "case",
        "issues",
        "facts",
        "procedural_history",
        "holdings",
        "legal_rules",
        "cited_cases",
        "statutes",
        "outcome",
    ],
)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    value = re.sub(r"\(\d+\)", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return normalize_space(value)


def display_party_name(value: str) -> str:
    name = re.sub(r"\s*,\s*", ", ", normalize_space(value).title())
    for word in ["Of", "The", "And", "For"]:
        name = name.replace(f" {word} ", f" {word.lower()} ")
    return name


def clean_page_text(raw_text: str) -> str:
    """Remove source-site furniture while retaining judgment wording."""
    lines = [normalize_space(line) for line in raw_text.splitlines()]
    cleaned: list[str] = []
    skip_previous_footer = False

    for line in lines:
        if not line:
            continue
        if line.startswith("Indian Kanoon - http"):
            if cleaned and " vs " in cleaned[-1] and " on " in cleaned[-1]:
                cleaned.pop()
            skip_previous_footer = True
            continue
        if skip_previous_footer and line.isdigit():
            skip_previous_footer = False
            continue
        cleaned.append(line)
        skip_previous_footer = False

    text = " ".join(cleaned)
    text = re.sub(r"(?<=\w)-\s+(?=[A-Za-z0-9])", "-", text)
    return normalize_space(text)


def extract_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = clean_page_text(raw_text)
            if text:
                pages.append({"page": page_number, "text": text, "raw_text": raw_text})
    return pages


def parse_header_metadata(pages: list[dict[str, Any]]) -> dict[str, Any]:
    first_raw = pages[0]["raw_text"]
    first_text = pages[0]["text"]

    title_part = re.split(r"Equivalent citations:", first_raw, maxsplit=1)[0]
    source_title = normalize_space(title_part)

    equivalent = re.search(r"Equivalent citations:\s*([^\n]+)", first_raw, re.I)
    citations = []
    if equivalent:
        citations = [normalize_space(item) for item in equivalent.group(1).split(",")]

    citation_block = re.search(
        r"CITATION:\s*(.*?)\s*CITATOR INFO", first_raw, re.I | re.S
    )
    if citation_block:
        citation_pattern = (
            r"\b(?:19|20)\d{2}\s+(?:"
            r"AIR\s+\d+|"
            r"SCR\s*\([^)]*\)\s*\d+|"
            r"SCC\s*\([^)]*\)\s*\d+)"
        )
        for citation in re.findall(citation_pattern, citation_block.group(1)):
            normalized = normalize_space(citation)
            if normalized not in citations:
                citations.append(normalized)

    author_match = re.search(r"Author:\s*([^\n]+)", first_raw, re.I)
    author = normalize_space(author_match.group(1)) if author_match else ""

    bench_match = re.search(r"Bench:\s*([^\n]+)", first_raw, re.I)
    judges = []
    if bench_match:
        judges = [normalize_space(name) for name in bench_match.group(1).split(",")]

    date_match = re.search(
        r"DATE OF JUDGMENT:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})",
        first_text,
        re.I,
    )

    parties = []
    party_match = re.search(
        r"PETITIONER:\s*(.*?)\s+Vs\.\s+RESPONDENT:\s*(.*?)\s+DATE OF JUDGMENT:",
        first_raw,
        re.I | re.S,
    )
    if party_match:
        petitioner = normalize_space(party_match.group(1))
        petitioner = re.split(r"\s*REPRESENTEDBY", petitioner, flags=re.I)[0]
        respondent = normalize_space(party_match.group(2))
        if petitioner:
            parties.append(
                {
                    "name": display_party_name(petitioner),
                    "role": "PETITIONER",
                    "evidence": petitioner,
                    "page": 1,
                }
            )
        if respondent:
            parties.append(
                {
                    "name": display_party_name(respondent),
                    "role": "RESPONDENT",
                    "evidence": respondent,
                    "page": 1,
                }
            )

    title = source_title
    if len(parties) == 2:
        title = f"{parties[0]['name']} v. {parties[1]['name']}"

    judgment_date = ""
    if date_match:
        judgment_date = datetime.strptime(date_match.group(1), "%d/%m/%Y").date().isoformat()

    court = ""
    complete_text = " ".join(page["text"] for page in pages)
    if (
        "CIVIL APPELLATE JURISDICTION" in complete_text
        and "Appeals by special leave from the judgment" in complete_text
    ):
        court = "Supreme Court of India"

    return {
        "name": title,
        "source_title": source_title,
        "court": court,
        "date": judgment_date,
        "citations": citations,
        "judges": judges,
        "author": author,
        "parties": parties,
    }


def batch_pages(
    pages: list[dict[str, Any]], max_chars: int = MAX_BATCH_CHARS
) -> list[list[dict[str, Any]]]:
    """Group complete pages without cutting sentences or paragraphs."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0

    for page in pages:
        page_size = len(page["text"])
        if current and current_size + page_size > max_chars:
            batches.append(current)
            current = []
            current_size = 0
        current.append(page)
        current_size += page_size

    if current:
        batches.append(current)
    return batches


def render_batch(batch: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[[PAGE {page['page']}]]\n{page['text']}" for page in batch
    )


def extraction_prompt(
    document_text: str,
    header: dict[str, Any],
    batch_number: int,
    batch_count: int,
) -> str:
    return f"""
Extract a precise, useful legal case record from the Indian judgment text below.
This is batch {batch_number} of {batch_count}. The complete-document identity is:

CURRENT CASE: {header['name']}
KNOWN COURT: {header['court'] or 'not parsed'}
KNOWN CITATIONS: {', '.join(header['citations']) or 'not parsed'}

Focus only on legally useful information: outcome-determinative facts, issues,
procedural decisions, holdings, governing rules, authorities actually discussed,
statutory provisions, and final disposition.

Rules:
1. The current case is the source of every holding. Never create "this Court" as
   a separate court or case.
2. Evidence must be one short, exact, contiguous quotation copied from one page
   (prefer one sentence and never exceed 350 characters). Do not paraphrase
   evidence and do not include [[PAGE n]] markers in it.
3. Set page to the page containing that exact evidence.
4. Keep the semantic text concise, but make it understandable outside the quote.
5. Do not treat counsel, headnote editors, or assessment officers as judges.
6. Ignore a headnote's generic "referred to" label. Use DISTINGUISHES when the
   court explains a material factual/legal difference;
   RELIES_ON when reasoning is adopted; APPLIES when a stated test is applied;
   REFERRED_TO for background only. Direction is always current case -> cited case.
7. A statute SECTION belongs in the statute's sections array, not as a case.
8. Omit trivial chronology and duplicate propositions. Use empty arrays when a
   category is absent from this batch.
9. Do not rely on outside knowledge. Return only JSON matching the schema.

JUDGMENT TEXT:
{document_text}
""".strip()


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    num_ctx: int,
    attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": EXTRACTION_SCHEMA,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": num_ctx,
                        "num_predict": 6000,
                        "seed": 7,
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            result = json.loads(payload["response"])
            if not isinstance(result, dict):
                raise ValueError("Model response was not a JSON object")
            return result
        except (
            requests.RequestException,
            KeyError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt < attempts:
                print(f"  Retry {attempt}/{attempts - 1}: {error}", file=sys.stderr)
    raise RuntimeError(f"Ollama extraction failed: {last_error}")


def empty_extraction() -> dict[str, Any]:
    return {
        "case": {
            "name": "",
            "court": "",
            "date": "",
            "citations": [],
            "judges": [],
            "author": "",
            "parties": [],
        },
        "issues": [],
        "facts": [],
        "procedural_history": [],
        "holdings": [],
        "legal_rules": [],
        "cited_cases": [],
        "statutes": [],
        "outcome": {"disposition": "", "evidence": "", "page": 0},
    }


def merge_extractions(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = empty_extraction()
    array_fields = [
        "issues",
        "facts",
        "procedural_history",
        "holdings",
        "legal_rules",
        "cited_cases",
        "statutes",
    ]

    for result in results:
        case = result.get("case", {})
        for field in ["name", "court", "date", "author"]:
            if not merged["case"][field] and case.get(field):
                merged["case"][field] = case[field]
        for field in ["citations", "judges", "parties"]:
            merged["case"][field].extend(case.get(field, []))
        for field in array_fields:
            merged[field].extend(result.get(field, []))
        outcome = result.get("outcome", {})
        if outcome.get("disposition"):
            merged["outcome"] = outcome
    return merged


def find_source_quote(
    evidence: Any,
    requested_page: Any,
    pages: list[dict[str, Any]],
) -> tuple[str, int] | None:
    if not isinstance(evidence, str):
        return None
    quote = normalize_space(evidence).strip('"“”')
    if len(quote) < 8:
        return None

    ordered = sorted(
        pages,
        key=lambda page: 0 if page["page"] == requested_page else 1,
    )
    for page in ordered:
        source = page["text"]
        index = source.casefold().find(quote.casefold())
        if index >= 0:
            return source[index : index + len(quote)], page["page"]

    # A model may copy a source quote while normalizing OCR punctuation such as
    # "of' regeneration" or "roots ,of". Match the same contiguous word sequence
    # and still return the exact source characters.
    quote_tokens = re.findall(r"[a-z0-9]+", quote.casefold())
    if len(quote_tokens) >= 3:
        for page in ordered:
            source = page["text"]
            matches = list(re.finditer(r"[a-z0-9]+", source.casefold()))
            source_tokens = [match.group(0) for match in matches]
            width = len(quote_tokens)
            for start in range(0, len(source_tokens) - width + 1):
                if source_tokens[start : start + width] == quote_tokens:
                    first, last = matches[start], matches[start + width - 1]
                    return source[first.start() : last.end()], page["page"]

    # If the model used an ellipsis, validate only its longest literal segment.
    # This avoids accepting the omitted material as if it had been quoted.
    if "..." in quote or "…" in quote:
        segments = sorted(
            (normalize_space(part) for part in re.split(r"\.{3,}|…", quote)),
            key=len,
            reverse=True,
        )
        for segment in segments:
            if len(segment) >= 20:
                matched = find_source_quote(segment, requested_page, pages)
                if matched:
                    return matched

    # PDF footnotes are sometimes injected into the middle of a sentence. Allow
    # extra source tokens only when the beginning and end still align and the
    # candidate retains a strong ordered-token match.
    if len(quote_tokens) >= 12:
        best: tuple[float, str, int] | None = None
        for page in ordered:
            source = page["text"]
            matches = list(re.finditer(r"[a-z0-9]+", source.casefold()))
            source_tokens = [match.group(0) for match in matches]
            for start, token in enumerate(source_tokens):
                if token != quote_tokens[0]:
                    continue
                upper = min(len(source_tokens), start + len(quote_tokens) + 45)
                for end in range(start + len(quote_tokens) - 1, upper):
                    if source_tokens[end] != quote_tokens[-1]:
                        continue
                    candidate = source_tokens[start : end + 1]
                    score = SequenceMatcher(None, quote_tokens, candidate).ratio()
                    if score >= 0.82 and (best is None or score > best[0]):
                        exact = source[matches[start].start() : matches[end].end()]
                        best = (score, exact, page["page"])
        if best:
            return best[1], best[2]
    return None


def deduplicate(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = tuple(normalize_key(str(item.get(field, ""))) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def citation_is_near_case(
    case_name: str, citation: str, pages: list[dict[str, Any]]
) -> bool:
    """Reject citations borrowed from an unrelated case elsewhere in the PDF."""
    if not citation:
        return True
    distinctive = [
        token
        for token in normalize_key(case_name).split()
        if len(token) >= 6
        and token not in {"commissioner", "income", "tax", "south", "company"}
    ]
    if not distinctive:
        return False
    citation_key = normalize_key(citation)
    for page in pages:
        page_key = normalize_key(page["text"])
        if citation_key in page_key and any(token in page_key for token in distinctive):
            return True
    return False


def refine_case_treatment(case_name: str, pages: list[dict[str, Any]]) -> str | None:
    """Use explicit judicial language to refine small-model citation treatment."""
    generic = {
        "commissioner",
        "income",
        "company",
        "bombay",
        "bengal",
        "kerala",
        "south",
    }
    distinctive = [
        token
        for token in normalize_key(case_name).split()
        if len(token) >= 7 and token not in generic
    ]
    full_text = " ".join(page["text"] for page in pages)
    full_key = normalize_key(full_text)
    positions = sorted(
        {
            match.start()
            for token in distinctive
            for match in re.finditer(rf"\b{re.escape(token)}\b", full_key)
        }
    )
    if not positions:
        return None
    contexts = [full_key[max(0, position - 200) : position + 2400] for position in positions]
    if any(
        phrase in context
        for context in contexts
        for phrase in [
            "question was apparently left open",
            "question does not arise",
            "did not relate to",
        ]
    ):
        return "DISTINGUISHES"
    if any(
        "test laid down" in context and "according to that test" in context
        for context in contexts
    ):
        return "APPLIES"
    if any(
        phrase in context
        for context in contexts
        for phrase in ["said that", "this is what was observed"]
    ):
        return "RELIES_ON"
    return None


def validate_extraction(
    extraction: dict[str, Any],
    pages: list[dict[str, Any]],
    header: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    validated = empty_extraction()
    stats = {"accepted_assertions": 0, "rejected_assertions": 0}

    case = extraction.get("case", {})
    validated["case"] = {
        "name": header["name"] or normalize_space(str(case.get("name", ""))),
        "court": header["court"] or normalize_space(str(case.get("court", ""))),
        "date": header["date"] or normalize_space(str(case.get("date", ""))),
        "citations": list(
            dict.fromkeys(header["citations"] + case.get("citations", []))
        ),
        "judges": list(dict.fromkeys(header["judges"] + case.get("judges", []))),
        "author": header["author"] or normalize_space(str(case.get("author", ""))),
        "parties": list(header["parties"]),
    }
    stats["accepted_assertions"] += len(header["parties"])

    def validate_items(field: str, text_field: str) -> None:
        for raw in extraction.get(field, []):
            if not isinstance(raw, dict) or not normalize_space(
                str(raw.get(text_field, ""))
            ):
                stats["rejected_assertions"] += 1
                continue
            source = find_source_quote(raw.get("evidence"), raw.get("page"), pages)
            if source is None:
                stats["rejected_assertions"] += 1
                continue
            item = dict(raw)
            item[text_field] = normalize_space(str(item[text_field]))
            item["evidence"], item["page"] = source
            validated[field].append(item)
            stats["accepted_assertions"] += 1

    for raw in case.get("parties", []):
        if not isinstance(raw, dict) or not raw.get("name"):
            stats["rejected_assertions"] += 1
            continue
        source = find_source_quote(raw.get("evidence"), raw.get("page"), pages)
        if source is None:
            stats["rejected_assertions"] += 1
            continue
        item = dict(raw)
        item["name"] = normalize_space(str(item["name"]))
        item["evidence"], item["page"] = source
        validated["case"]["parties"].append(item)
        stats["accepted_assertions"] += 1

    for field in ["issues", "facts", "holdings", "legal_rules"]:
        validate_items(field, "text")
    validate_items("procedural_history", "decision")
    validate_items("cited_cases", "name")
    validate_items("statutes", "name")

    for cited in validated["cited_cases"]:
        citation = normalize_space(str(cited.get("citation", "")))
        if citation and not citation_is_near_case(cited["name"], citation, pages):
            cited["citation"] = ""
        refined = refine_case_treatment(cited["name"], pages)
        if refined:
            cited["treatment"] = refined

    outcome = extraction.get("outcome", {})
    if outcome.get("disposition"):
        source = find_source_quote(outcome.get("evidence"), outcome.get("page"), pages)
        if source:
            validated["outcome"] = {
                "disposition": normalize_space(str(outcome["disposition"])),
                "evidence": source[0],
                "page": source[1],
            }
            stats["accepted_assertions"] += 1
        else:
            stats["rejected_assertions"] += 1

    validated["case"]["parties"] = deduplicate(
        validated["case"]["parties"], ["name", "role"]
    )
    for field, keys in {
        "issues": ["text"],
        "facts": ["text"],
        "procedural_history": ["stage", "decision"],
        "holdings": ["text"],
        "legal_rules": ["text"],
        "cited_cases": ["name"],
        "statutes": ["name"],
    }.items():
        validated[field] = deduplicate(validated[field], keys)

    return validated, stats


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def add_node(
        self,
        node_type: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        key = f"{node_type}:{normalize_key(label)}"
        node_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        node = self.nodes.setdefault(
            node_id,
            {"id": node_id, "type": node_type, "label": normalize_space(label)},
        )
        if properties:
            for name, value in properties.items():
                if value not in (None, "", []):
                    node[name] = value
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        evidence: str = "",
        page: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        key = f"{source}|{relation}|{target}|{normalize_key(evidence)}"
        edge_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
        }
        if evidence:
            edge["evidence"] = evidence
        if page:
            edge["page"] = page
        if properties:
            edge.update({name: value for name, value in properties.items() if value})
        self.edges[edge_id] = edge


def build_graph(
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = GraphBuilder()
    case = record["case"]
    case_id = graph.add_node(
        "CASE",
        case["name"],
        {
            "citations": case["citations"],
            "date": case["date"],
            "is_current_case": True,
        },
    )

    if case["court"]:
        court_id = graph.add_node("COURT", case["court"])
        graph.add_edge(case_id, court_id, "DECIDED_AT")

    for judge in case["judges"]:
        judge_id = graph.add_node("JUDGE", judge)
        graph.add_edge(case_id, judge_id, "DECIDED_BY")
    if case["author"]:
        author_id = graph.add_node("JUDGE", case["author"])
        graph.add_edge(case_id, author_id, "AUTHORED_BY")

    for party in case["parties"]:
        party_id = graph.add_node("PARTY", party["name"], {"role": party["role"]})
        graph.add_edge(
            case_id,
            party_id,
            "HAS_PARTY",
            party["evidence"],
            party["page"],
            {"role": party["role"]},
        )

    semantic_fields = [
        ("issues", "ISSUE", "RAISES_ISSUE", "text"),
        ("facts", "FACT", "HAS_MATERIAL_FACT", "text"),
        ("holdings", "HOLDING", "HAS_HOLDING", "text"),
        ("legal_rules", "LEGAL_RULE", "APPLIES_RULE", "text"),
        (
            "procedural_history",
            "PROCEDURAL_EVENT",
            "HAS_PROCEDURAL_HISTORY",
            "decision",
        ),
    ]
    for field, node_type, relation, text_field in semantic_fields:
        for item in record[field]:
            properties = {"evidence": item["evidence"], "page": item["page"]}
            if field == "procedural_history":
                properties["stage"] = item.get("stage", "")
            node_id = graph.add_node(node_type, item[text_field], properties)
            graph.add_edge(
                case_id, node_id, relation, item["evidence"], item["page"]
            )

    for cited in record["cited_cases"]:
        cited_id = graph.add_node(
            "CASE",
            cited["name"],
            {"citation": cited.get("citation", ""), "is_current_case": False},
        )
        treatment = cited.get("treatment", "CITES")
        if treatment not in TREATMENTS:
            treatment = "CITES"
        graph.add_edge(
            case_id,
            cited_id,
            treatment,
            cited["evidence"],
            cited["page"],
            {"proposition": cited.get("proposition", "")},
        )

    for statute in record["statutes"]:
        statute_id = graph.add_node("STATUTE", statute["name"])
        treatment = statute.get("treatment", "MENTIONS")
        if treatment not in STATUTE_TREATMENTS:
            treatment = "MENTIONS"
        graph.add_edge(
            case_id,
            statute_id,
            treatment,
            statute["evidence"],
            statute["page"],
        )
        for section in statute.get("sections", []):
            section_label = f"{section} of {statute['name']}"
            section_id = graph.add_node(
                "SECTION", section_label, {"section": section}
            )
            graph.add_edge(section_id, statute_id, "PART_OF")
            graph.add_edge(
                case_id,
                section_id,
                treatment,
                statute["evidence"],
                statute["page"],
            )

    outcome = record["outcome"]
    if outcome["disposition"]:
        outcome_id = graph.add_node(
            "OUTCOME",
            outcome["disposition"],
            {"evidence": outcome["evidence"], "page": outcome["page"]},
        )
        graph.add_edge(
            case_id,
            outcome_id,
            "HAS_OUTCOME",
            outcome["evidence"],
            outcome["page"],
        )

    return list(graph.nodes.values()), list(graph.edges.values())


def create_knowledge_graph(
    pdf_path: Path,
    model: str,
    ollama_url: str,
    num_ctx: int,
    document_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pages = extract_pdf(pdf_path)
    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. Run OCR on scanned PDFs first."
        )

    header = parse_header_metadata(pages)
    batches = batch_pages(pages)
    raw_results: list[dict[str, Any]] = []
    print(f"Extracted {len(pages)} text pages in {len(batches)} semantic batch(es)")

    for index, batch in enumerate(batches, start=1):
        page_range = f"{batch[0]['page']}-{batch[-1]['page']}"
        print(f"Processing batch {index}/{len(batches)} (pages {page_range})")
        prompt = extraction_prompt(render_batch(batch), header, index, len(batches))
        raw_results.append(call_ollama(prompt, model, ollama_url, num_ctx))

    merged = merge_extractions(raw_results)
    record, validation = validate_extraction(merged, pages, header)
    nodes, edges = build_graph(record)

    canonical_document_id, content_sha256 = resolve_document_identity(
        pdf_path, document_id
    )
    output = {
        "metadata": {
            "schema_version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "document_id": canonical_document_id,
            "content_sha256": content_sha256,
            "source_pdf": pdf_path.name,
            "model": model,
            "page_count": len(pages),
            "batch_count": len(batches),
            "validation": validation,
        },
        "case_record": record,
        "nodes": nodes,
        "edges": edges,
    }
    return output, raw_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an evidence-grounded legal knowledge graph from a PDF."
    )
    parser.add_argument("pdf", help="Path to the judgment PDF")
    parser.add_argument("--output", default="knowledge_graph.json")
    parser.add_argument("--debug-output", help="Optional path for raw model responses")
    parser.add_argument(
        "--document-id",
        help=(
            "Canonical ID assigned by the upload API. If omitted, the PDF SHA-256 "
            "is used for backwards-compatible one-file runs."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    output, raw_results = create_knowledge_graph(
        pdf_path=pdf_path,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
        document_id=args.document_id,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.debug_output:
        debug_path = Path(args.debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps(raw_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Saved graph to: {output_path}")
    print(f"Nodes: {len(output['nodes'])}")
    print(f"Edges: {len(output['edges'])}")
    validation = output["metadata"]["validation"]
    print(
        "Evidence validation: "
        f"{validation['accepted_assertions']} accepted, "
        f"{validation['rejected_assertions']} rejected"
    )


if __name__ == "__main__":
    main()
