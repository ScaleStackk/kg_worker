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


DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://127.0.0.1:11434/api/generate"
)
DEFAULT_MODEL = os.environ.get("KG_MODEL", "qwen3:4b")
REQUEST_TIMEOUT = int(os.environ.get("KG_REQUEST_TIMEOUT", "900"))
MAX_BATCH_CHARS = int(os.environ.get("KG_MAX_BATCH_CHARS", "24000"))
DEFAULT_NUM_CTX = int(os.environ.get("KG_NUM_CTX", "16384"))
NUM_PREDICT = int(os.environ.get("KG_NUM_PREDICT", "3000"))

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


# Length limits are enforced by Ollama's constrained decoding. Without them a
# small model can start copying page text into one string and never close it,
# so the response hits num_predict and is invalid JSON; with temperature 0 the
# retry repeats the same failure. A quote cut at the limit still validates,
# because find_source_quote matches on word tokens.
def text(max_length: int) -> dict[str, Any]:
    return {"type": "string", "maxLength": max_length}


def array(items: dict[str, Any], max_items: int = 25) -> dict[str, Any]:
    return {"type": "array", "items": items, "maxItems": max_items}


EVIDENCE = text(400)
PAGE = {"type": "integer"}

EVIDENCED_TEXT_SCHEMA = object_schema(
    {"text": text(500), "evidence": EVIDENCE, "page": PAGE},
    ["text", "evidence", "page"],
)

EXTRACTION_SCHEMA = object_schema(
    {
        "case": object_schema(
            {
                "name": text(300),
                "court": text(150),
                "date": text(40),
                "citations": array(text(120), 10),
                "judges": array(text(120), 10),
                "author": text(120),
                "parties": array(
                    object_schema(
                        {
                            "name": text(200),
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
                            "evidence": EVIDENCE,
                            "page": PAGE,
                        },
                        ["name", "role", "evidence", "page"],
                    ),
                    10,
                ),
            },
            ["name", "court", "date", "citations", "judges", "author", "parties"],
        ),
        "issues": array(EVIDENCED_TEXT_SCHEMA),
        "facts": array(EVIDENCED_TEXT_SCHEMA),
        "procedural_history": array(
            object_schema(
                {
                    "stage": text(200),
                    "decision": text(500),
                    "evidence": EVIDENCE,
                    "page": PAGE,
                },
                ["stage", "decision", "evidence", "page"],
            )
        ),
        "holdings": array(EVIDENCED_TEXT_SCHEMA),
        "legal_rules": array(EVIDENCED_TEXT_SCHEMA),
        "cited_cases": array(
            object_schema(
                {
                    "name": text(300),
                    "citation": text(120),
                    "treatment": {"type": "string", "enum": sorted(TREATMENTS)},
                    "proposition": text(500),
                    "evidence": EVIDENCE,
                    "page": PAGE,
                },
                ["name", "citation", "treatment", "proposition", "evidence", "page"],
            ),
            40,
        ),
        "statutes": array(
            object_schema(
                {
                    "name": text(200),
                    "sections": array(text(60), 20),
                    "treatment": {
                        "type": "string",
                        "enum": sorted(STATUTE_TREATMENTS),
                    },
                    "evidence": EVIDENCE,
                    "page": PAGE,
                },
                ["name", "sections", "treatment", "evidence", "page"],
            )
        ),
        "outcome": object_schema(
            {
                "disposition": text(400),
                "evidence": EVIDENCE,
                "page": PAGE,
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


SIGNATURE_LINE = re.compile(
    r"^(?:Signature Not Verified|Digitally signed by|Reason:|"
    r"Date:\s*\d{4}\.\d{2}\.\d{2}(?:\s+\d{2}:\d{2}:\d{2}\s*IST)?(?:\s+Reason:)?|"
    r"\d{2}:\d{2}:\d{2}\s*IST(?:\s+Reason:)?)$",
    re.I,
)
SIGNATURE_INLINE = re.compile(
    r"\s*(?:Signature Not Verified|Digitally signed by|"
    r"Date:\s*\d{4}\.\d{2}\.\d{2}(?:\s+\d{2}:\d{2}:\d{2}\s*IST)?|"
    r"\d{2}:\d{2}:\d{2}\s*IST)(?:\s+Reason:)?",
    re.I,
)
SIGNER_NAME = re.compile(r"^[A-Z][A-Z .]{1,40}$")


def strip_signature_lines(lines: list[str]) -> list[str]:
    """Drop the digital-signature stamp the Supreme Court PDFs carry on page 1.

    The stamp is "Signature Not Verified / Digitally signed by / NAME /
    Date: 2025.04.04 / 16:28:38 IST / Reason:"; PyMuPDF often interleaves it
    with judgment text, so whole stamp lines are removed and inline fragments
    are cut out of the remaining lines.
    """
    result: list[str] = []
    after_signed_by = False
    for line in lines:
        if after_signed_by and SIGNER_NAME.match(line):
            after_signed_by = False
            continue
        after_signed_by = line.casefold() == "digitally signed by"
        if SIGNATURE_LINE.match(line):
            continue
        line = normalize_space(SIGNATURE_INLINE.sub("", line))
        if line:
            result.append(line)
    return result


def clean_page_text(raw_text: str) -> str:
    """Remove source-site furniture while retaining judgment wording."""
    lines = [normalize_space(line) for line in raw_text.splitlines()]
    lines = strip_signature_lines([line for line in lines if line])
    cleaned: list[str] = []
    skip_previous_footer = False

    for line in lines:
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


MAX_SOURCE_TITLE = 300
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
TITLE_LINE = re.compile(
    rf"^(?P<title>.{{3,{MAX_SOURCE_TITLE}}}?)\s+on\s+"
    rf"(?P<date>\d{{1,2}}\s+(?:{MONTHS}),?\s+\d{{4}})",
    re.S,
)
NEUTRAL_CITATION = re.compile(r"\b(?:19|20)\d{2}\s+INSC\s+\d+\b")
COURT_HEADING = re.compile(
    r"^\s*(?:\d{4}\s+INSC\s+\d+\s+)?(?:IN\s+(?:THE\s+)?)?"
    r"(SUPREME\s+COURT\s+OF\s+INDIA|HIGH\s+COURT\s+OF\s+[A-Z][A-Z &,.]+?)"
    r"(?:\s+AT\s+[A-Z ]+?)?\s*$",
    re.I | re.M,
)
VERSUS_LINE = re.compile(r"^[-−–\s]*(?:versus|vs?\.?)[-−–\s]*$", re.I)
BLOCK_END_LINE = re.compile(
    r"^(?:J\s*U\s*D\s*G\s*E?\s*M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R|WITH|AND|"
    r"COMMON\s+JUDGMENT|INDEX|Signature Not Verified)\b",
    re.I,
)
BLOCK_START_LINE = re.compile(
    r"JURISDICTION|\bNOS?\b\.?|\bNO\s*\(S\)|ARISING|DIARY|\bOF\s+(?:19|20)\d{2}\b|"
    r"^IN\s+THE\s+(?:SUPREME|HIGH)\s+COURT|REPORTABLE",
    re.I,
)
CASE_NUMBER = re.compile(
    r"\bNOS?\s*(?:\(\s*S\s*\))?\s*\.?\s*[\d\s,\-–−/]+"
    r"(?:\s*\bO(?:F)?\b\s*\d{0,4})?\s*[)\]]?",
    re.I,
)
ROLES = {
    "APPELLANT": "APPELLANT",
    "PETITIONER": "PETITIONER",
    "APPLICANT": "PETITIONER",
    "RESPONDENT": "RESPONDENT",
}
ROLE_TOKEN = re.compile(
    r"(?P<lead>(?:\.{2,}|…)[.…]*)?\s*\b(?P<word>[A-Za-z]+)(?![A-Za-z0-9])"
    r"(?:\s*\(\s*s?\s*\)?)?(?:\s*/)?[.…]*",
    re.I,
)
PARTY_TAIL = re.compile(
    r"(?:\s*(?:,|&|\band\b)\s*(?:ANR|ANOTHER|ORS|OTHERS|OTHER)\.?|\s+ETC\.?)+\s*$",
    re.I,
)
SIGNATURE_JUDGE = re.compile(
    r"(?:[.…]\s*){3,}J\s*\.?\s*[(\[]\s*(?P<name>[^)\]]{3,80}?)\s*[)\]]"
)
OPENING_JUDGE = re.compile(r"^(?P<name>[A-Z][A-Za-z.' ]{1,60}?)\s*,\s*J\s*\.?\s*:?$")


def role_label(text: str) -> tuple[str, str] | None:
    """Split a (possibly truncated) role label out of a party block side.

    Indian Kanoon wraps the right-hand column, so labels arrive as
    "...APPELLANT(S)", "…RESPON", "APPELLANT(s)", ".....PETI", "… A", or in the
    middle of a wrapped name ("CUDDALORE POWERGEN …APPELLANT(S) CORPORATION").
    """
    found = None
    for match in ROLE_TOKEN.finditer(text):
        word = match.group("word").upper()
        if word in {"PETITION", "PETITIONS"}:
            continue
        dotted = bool(match.group("lead"))
        at_end = not text[match.end() :].strip()
        for full, role in ROLES.items():
            if word in (full, full + "S") or (
                full.startswith(word) and (dotted or (at_end and len(word) >= 4))
            ):
                found = (match, role)
                break
    if not found:
        return None
    match, role = found
    name = normalize_space(f"{text[: match.start()]} {text[match.end():]}")
    return name, role


def clean_party_name(value: str) -> str:
    # Remove any further labels ("...APPLICANT(S)/PETITIONER(S)").
    while (labelled := role_label(value)) and labelled[0] != value:
        value = labelled[0]
    # Enumerated parties ("A1: X A2: Y"): keep the lead name.
    enumerated = re.split(r"\s*\b[ARP]\d+\s*:\s*", value)
    value = next((part for part in enumerated if part.strip()), value)
    value = re.sub(r"^[\d/\s-]+(?=[A-Za-z])", "", value)
    value = normalize_space(value).strip(" .,:;-…")
    return normalize_space(PARTY_TAIL.sub("", value)).strip(" .,:;-…")


def parse_party_block(lines: list[str], appellate: bool) -> list[dict[str, Any]]:
    """Parse the first "PARTY ...ROLE / VERSUS / PARTY ...ROLE" block."""
    versus = next((index for index, line in enumerate(lines) if VERSUS_LINE.match(line)), None)
    if versus is None:
        return []

    left: list[str] = []
    for line in reversed(lines[max(0, versus - 6) : versus]):
        if BLOCK_START_LINE.search(line):
            # "(@ SLP (C) NO. 27632 OF 2024) ANMOL APPELLANT(s)": keep the tail.
            number = CASE_NUMBER.search(line)
            tail = line[number.end() :].strip() if number else ""
            if tail and role_label(tail):
                left.insert(0, tail)
            break
        left.insert(0, line)

    right: list[str] = []
    for line in lines[versus + 1 : versus + 7]:
        if BLOCK_END_LINE.match(line) or OPENING_JUDGE.match(line) or re.match(r"^\d+\.\s", line):
            break
        right.append(line)
        if role_label(line):
            break

    parties = []
    defaults = ["APPELLANT" if appellate else "PETITIONER", "RESPONDENT"]
    for side, default_role in zip([left, right], defaults):
        evidence = normalize_space(" ".join(side))
        labelled = role_label(evidence)
        name, role = labelled if labelled else (evidence, default_role)
        name = clean_party_name(name)
        if 2 <= len(name) <= 200:
            parties.append(
                {
                    "name": display_party_name(name),
                    "role": role,
                    "evidence": evidence,
                    "page": 1,
                }
            )
    return parties if len(parties) == 2 else []


def judge_display_name(value: str) -> str:
    value = normalize_space(value)
    tokens = value.split()
    if len(tokens) > 3 and all(len(token) == 1 for token in tokens):
        # Letter-spaced signatures: "P A M I D I G H A N T A M S R I ..." can't be
        # split back into words reliably, so keep it as one word.
        value = "".join(tokens)
    value = re.sub(r"\s*\.\s*", ". ", value).replace(". ", ".").strip()
    value = re.sub(r"\.(?=[A-Za-z]{2,})", ". ", value)
    return normalize_space(value.title())


def signature_judges(pages: list[dict[str, Any]]) -> list[str]:
    """Judges named in the closing "....J. (NAME)" signature block."""
    for page in reversed(pages[-2:]):
        names = [
            judge_display_name(match.group("name"))
            for match in SIGNATURE_JUDGE.finditer(page["text"])
        ]
        if names:
            return list(dict.fromkeys(names))
    return []


def resolve_judge_names(names: list[str], known: list[str]) -> list[str]:
    """Replace letter-spaced signature names ("Vikramnath") with a known spelling."""
    def compact(name: str) -> str:
        return normalize_key(name).replace(" ", "")

    spellings = {compact(name): name for name in known if name}
    resolved: dict[str, str] = {}
    for name in names:
        resolved.setdefault(compact(name), spellings.get(compact(name), name))
    return list(resolved.values())


def surname_key(name: str) -> str:
    tokens = normalize_key(name).split()
    return tokens[-1] if tokens else ""


def opening_author(lines: list[str], judges: list[str]) -> str:
    """Author from the opening "<NAME>, J." line, matched to the bench."""
    for line in lines[:60]:
        match = OPENING_JUDGE.match(line)
        if not match:
            continue
        name = judge_display_name(match.group("name"))
        for judge in judges:
            if surname_key(judge) == surname_key(name):
                return judge
        return name
    return ""


def parse_modern_header(pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse the 2020s Indian Kanoon layout: "X vs Y on 10 February, 2025"."""
    first_raw = pages[0]["raw_text"]
    if re.search(r"Equivalent citations:|^\s*PETITIONER:", first_raw, re.I | re.M):
        return None
    lines = strip_signature_lines(
        [normalize_space(line) for line in first_raw.splitlines() if line.strip()]
    )
    title_match = TITLE_LINE.match(normalize_space(" ".join(lines[:4])))
    if not title_match:
        return None

    title = title_match.group("title")
    raw_date = title_match.group("date").replace(",", "")
    judgment_date = datetime.strptime(raw_date, "%d %B %Y").date().isoformat()
    source_title = f"{title} on {title_match.group('date')}"

    header_text = "\n".join(lines[:40])
    citations = list(
        dict.fromkeys(normalize_space(c) for c in NEUTRAL_CITATION.findall(header_text))
    )

    court = ""
    court_match = COURT_HEADING.search(header_text)
    if court_match:
        heading = normalize_space(court_match.group(1)).title()
        court = re.sub(r"\b(Of|And)\b", lambda m: m.group(1).lower(), heading)

    author_match = re.search(r"^Author:\s*(.+)$", header_text, re.M)
    header_author = judge_display_name(author_match.group(1)) if author_match else ""
    bench_match = re.search(r"^Bench:\s*(.+)$", header_text, re.M)
    header_judges = (
        [judge_display_name(name) for name in bench_match.group(1).split(",")]
        if bench_match
        else []
    )

    signed = resolve_judge_names(signature_judges(pages), header_judges + [header_author])
    judges = signed or header_judges
    body_author = opening_author(lines, judges)
    author = body_author or header_author

    appellate = "APPELLATE JURISDICTION" in header_text.upper()
    parties = parse_party_block(lines[:60], appellate)
    name = (
        f"{parties[0]['name']} v. {parties[1]['name']}"
        if len(parties) == 2
        else re.sub(r"\s+vs\.?\s+", " v. ", title)
    )

    return {
        "name": name,
        "source_title": source_title,
        "court": court,
        "date": judgment_date,
        "citations": citations,
        "judges": judges,
        "author": author,
        "parties": parties,
        "header_judges": header_judges,
        "header_author": header_author,
    }


def parse_header_metadata(pages: list[dict[str, Any]]) -> dict[str, Any]:
    modern = parse_modern_header(pages)
    if modern:
        return modern

    first_raw = pages[0]["raw_text"]
    first_text = pages[0]["text"]

    title_part = re.split(r"Equivalent citations:", first_raw, maxsplit=1)[0]
    source_title = normalize_space(title_part)[:MAX_SOURCE_TITLE]

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

    # Never fall back to raw page text as a case name; an unparsed name is
    # filled from the model or the file name later.
    title = ""
    title_match = TITLE_LINE.match(source_title)
    if title_match:
        title = re.sub(r"\s+vs\.?\s+", " v. ", title_match.group("title"))
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

CURRENT CASE: {header['name'] or 'not parsed'}
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
                        "num_predict": NUM_PREDICT,
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
                # The last word may be cut short by the schema's maxLength.
                if (
                    source_tokens[start : start + width - 1] == quote_tokens[:-1]
                    and source_tokens[start + width - 1].startswith(quote_tokens[-1])
                ):
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


SENTENCE_ABBREVIATIONS = {
    "v", "vs", "ltd", "no", "nos", "s", "ss", "sec", "secs", "co", "pvt", "dr", "mr",
    "mrs", "ms", "j", "jj", "cji", "ie", "eg", "viz", "etc", "art", "arts", "cl",
    "para", "paras", "p", "pp", "vol", "cf", "cr", "crl", "civ", "govt", "ors", "anr",
    "smt", "sri", "shri", "hon", "ble", "corpn", "supp", "sc", "scc", "air", "ibid",
}
SENTENCE_END = re.compile(r"[.?!][\"”’')\]]*\s+(?=[A-Z0-9“\"‘'(\[])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of sentences, not splitting after legal abbreviations."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        before = re.search(r"([A-Za-z]+)\W*$", text[start : match.start() + 1])
        word = before.group(1).casefold() if before else ""
        if len(word) == 1 or word in SENTENCE_ABBREVIATIONS:
            continue
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


GENERIC_NAME_WORDS = {
    "state", "states", "union", "india", "government", "commissioner", "income",
    "tax", "limited", "company", "private", "corporation", "bank", "insurance",
    "others", "another", "director", "officer", "secretary", "general", "district",
    "collector", "municipal", "authority", "board", "national", "central", "through",
    "maharashtra", "rajasthan", "pradesh", "uttar", "madhya", "andhra", "himachal",
    "arunachal", "karnataka", "kerala", "tamil", "nadu", "gujarat", "punjab",
    "haryana", "bihar", "odisha", "orissa", "bengal", "west", "assam", "telangana",
    "jharkhand", "chhattisgarh", "uttarakhand", "delhi", "bombay", "madras",
    "calcutta", "kolkata", "chennai", "mumbai", "goa", "jammu", "kashmir", "police",
    "inspector", "manager", "managing", "chief", "deputy", "assistant", "appellant",
    "respondent", "petitioner", "anr", "ors", "kumar", "singh", "devi", "prasad",
    "sharma", "gupta", "yadav", "khan", "mohd", "mohammad", "shri", "sons",
    "association", "society", "trust", "industries", "enterprises", "supra",
}


def distinctive_tokens(case_name: str) -> list[list[str]]:
    """Distinctive words of each side of "X v. Y", for locating mentions."""
    sides = re.split(r"\s+(?:v|vs|versus)\.?\s+", case_name, flags=re.I)
    result = []
    for side in sides:
        tokens = [
            token
            for token in normalize_key(side).split()
            if len(token) >= 4 and token not in GENERIC_NAME_WORDS
        ]
        if tokens:
            result.append(tokens)
    return result


# A following sentence that names another authority is about that authority.
OTHER_AUTHORITY = re.compile(
    r"\(supra\)|\s(?:v|vs)\.?\s|\b(?:SCC|INSC|SCR|AIR)\b|"
    r"\b(?:other|another|later|subsequent|earlier)\s+(?:decision|judgment|case)s?\b",
    re.I,
)


def mention_contexts(cited: dict[str, Any], pages: list[dict[str, Any]]) -> list[str]:
    """The sentences that cite a case, each joined with the sentence after it.

    A mention is the validated evidence quote, the citation string, or a
    sentence containing a distinctive word of the lead party (judgments refer
    back to a case as "Taherakhatoon (supra)").
    """
    sides = distinctive_tokens(str(cited.get("name", "")))[:1]
    citation = normalize_space(str(cited.get("citation", ""))).casefold()
    evidence = normalize_space(str(cited.get("evidence", ""))).casefold()
    contexts: list[str] = []
    for page in pages:
        text = page["text"]
        spans = split_sentences(text)
        for index, (start, end) in enumerate(spans):
            sentence = text[start:end]
            folded = sentence.casefold()
            words = set(normalize_key(sentence).split())
            hit = (
                (evidence and len(evidence) >= 8 and evidence in folded)
                or (citation and len(citation) >= 6 and citation in folded)
                or (sides and all(any(token in words for token in side) for side in sides))
            )
            if not hit:
                continue
            following = text[spans[index + 1][0] : spans[index + 1][1]] if index + 1 < len(spans) else ""
            if OTHER_AUTHORITY.search(following):
                following = ""
            contexts.append(normalize_space(f"{sentence} {following}"))
    return list(dict.fromkeys(contexts))


NEGATION_BEFORE = re.compile(r"\b(?:not|never|cannot|can't|no|nor|neither)\b(?:\W+\w+){0,3}\W*$")
COUNSEL_VOICE = re.compile(
    r"\b(?:learned\s+(?:senior\s+)?counsel|counsel\s+(?:for|appearing)|"
    r"(?:it\s+(?:is|was)\s+)?(?:submitted|contended|argued|urged|canvassed)|"
    r"on\s+behalf\s+of\s+the\s+(?:appellant|respondent|petitioner|state)s?|"
    r"relied\s+(?:up)?on\s+by\s+the\s+(?:appellant|respondent|petitioner)s?|"
    r"(?:appellant|respondent|petitioner)s?\s+(?:has|have|had)\s+relied)\b",
    re.I,
)
TREATMENT_CUES: list[tuple[str, re.Pattern[str]]] = [
    (
        "OVERRULES",
        re.compile(
            r"\b(?:(?:is|are|stands?|be|hereby)\s+(?:hereby\s+)?overruled|we\s+overrule|"
            r"(?:does|do|did)\s+not\s+lay\s+down\s+(?:the\s+)?(?:correct|good)\s+law|"
            r"(?:is|are)\s+no\s+longer\s+good\s+law|(?:is|are)\s+not\s+good\s+law)\b",
            re.I,
        ),
    ),
    (
        "DISTINGUISHES",
        re.compile(
            r"\b(?:distinguishable|(?:is|are|was|were|be)\s+(?:clearly\s+)?distinguished|"
            r"(?:does|do|would)\s+not\s+(?:apply|help|assist|come\s+to\s+the\s+(?:aid|rescue))|"
            r"(?:has|have)\s+no\s+(?:application|applicability)|"
            r"(?:not|in)applicable\s+to\s+the\s+(?:facts|present\s+case)|"
            r"facts\s+(?:of|in)\s+(?:that|the\s+said|those)\s+cases?\s+(?:are|were)\s+"
            r"(?:entirely\s+|completely\s+|clearly\s+|quite\s+)?(?:different|distinct))\b",
            re.I,
        ),
    ),
    (
        "FOLLOWS",
        re.compile(
            r"\b(?:we\s+(?:are\s+)?(?:respectfully\s+)?(?:agree|follow|concur)|"
            r"(?:we|this\s+court)\s+(?:have|has|had)\s+(?:consistently\s+)?followed|"
            r"(?:is|are)\s+squarely\s+(?:applicable|covered)|squarely\s+covers|"
            r"(?:is|are)\s+(?:fully\s+)?covered\s+by\s+(?:the\s+)?(?:decision|judgment)|"
            r"(?:is|are)\s+binding\s+on\s+us)\b",
            re.I,
        ),
    ),
    (
        "APPLIES",
        re.compile(
            r"\b(?:applying\s+the\s+(?:test|principles?|ratio|law)|"
            r"(?:test|principles?|ratio)\s+(?:laid\s+down|enunciated|propounded)\s+.{0,80}?"
            r"(?:applies|apply|applied|is\s+attracted))\b",
            re.I,
        ),
    ),
    (
        "RELIES_ON",
        re.compile(
            r"\b(?:(?:we|this\s+court)\s+(?:may\s+|would\s+)?(?:also\s+)?(?:rely|relied|place\s+reliance)|"
            r"(?:we|this\s+court)\s+(?:have|has|had)\s+(?:also\s+)?(?:relied|placed\s+reliance)|"
            r"(?:reliance|support)\s+(?:is|can\s+be|may\s+be)\s+(?:placed|drawn|derived|found)|"
            r"(?:fortified|supported)\s+by\s+(?:the\s+)?(?:decision|judgment|view))\b",
            re.I,
        ),
    ),
]
UPGRADES = {"RELIES_ON", "APPLIES", "FOLLOWS"}


def cue_fires(pattern: re.Pattern[str], context: str) -> bool:
    for match in pattern.finditer(context):
        if not NEGATION_BEFORE.search(context[: match.start()]):
            return True
    return False


def refine_case_treatment(
    cited: dict[str, Any], pages: list[dict[str, Any]]
) -> str | None:
    """Treatment implied by explicit judicial language around the citation.

    Returns None unless a cue fires in a sentence that cites the case (or the
    sentence after it). Sentences that report counsel's argument never
    upgrade the treatment; when every mention is counsel's, a model label of
    RELIES_ON/APPLIES/FOLLOWS is lowered to REFERRED_TO.
    """
    contexts = mention_contexts(cited, pages)
    if not contexts:
        return None
    judicial = [context for context in contexts if not COUNSEL_VOICE.search(context)]
    for treatment, pattern in TREATMENT_CUES:
        if any(cue_fires(pattern, context) for context in judicial):
            return treatment
    if not judicial and cited.get("treatment") in UPGRADES:
        return "REFERRED_TO"
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
        # Indian Kanoon's Author/Bench lines, kept when they disagree with the
        # judgment body (signature block and opening "<NAME>, J." line).
        "header_author": "",
        "header_judges": [],
    }
    if header.get("header_author", "") != validated["case"]["author"]:
        validated["case"]["header_author"] = header.get("header_author", "")
    if set(header.get("header_judges", [])) != set(validated["case"]["judges"]):
        validated["case"]["header_judges"] = header.get("header_judges", [])
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
        refined = refine_case_treatment(cited, pages)
        cited["treatment_source"] = "llm"
        if refined and refined != cited.get("treatment"):
            cited["llm_treatment"] = cited.get("treatment", "")
            cited["treatment"] = refined
            cited["treatment_source"] = "rule"

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


STATUTE_ALIASES = {
    "ipc": "indian penal code 1860",
    "i p c": "indian penal code 1860",
    "penal code": "indian penal code 1860",
    "crpc": "code of criminal procedure 1973",
    "cr p c": "code of criminal procedure 1973",
    "criminal procedure code": "code of criminal procedure 1973",
    "cpc": "code of civil procedure 1908",
    "c p c": "code of civil procedure 1908",
    "civil procedure code": "code of civil procedure 1908",
    "bns": "bharatiya nyaya sanhita 2023",
    "bnss": "bharatiya nagarik suraksha sanhita 2023",
    "bsa": "bharatiya sakshya adhiniyam 2023",
    "ni act": "negotiable instruments act 1881",
    "n i act": "negotiable instruments act 1881",
    "ibc": "insolvency and bankruptcy code 2016",
    "i b c": "insolvency and bankruptcy code 2016",
    "ndps act": "narcotic drugs and psychotropic substances act 1985",
    "pc act": "prevention of corruption act 1988",
    "evidence act": "indian evidence act 1872",
    "constitution": "constitution of india",
    "indian constitution": "constitution of india",
}
CASE_WORD_ALIASES = {
    "vs": "v",
    "versus": "v",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "corpn": "corporation",
    "govt": "government",
    "cit": "commissioner of income tax",
    "u p": "uttar pradesh",
    "m p": "madhya pradesh",
    "a p": "andhra pradesh",
    "h p": "himachal pradesh",
    "nct": "national capital territory",
}
LITIGANT_NOISE = re.compile(
    r"\b(?:m s|messrs|the|and (?:another|others|ors|anr)|ors|anr|etc|"
    r"since deceased|dead|deceased|through lrs|by lrs|lrs)\b"
)
JUDGE_TITLES = re.compile(
    r"\b(?:hon ble|honble|honourable|hon|the|mr|mrs|ms|dr|justice|chief|cji|j|jj|"
    r"retd|retired|as his lordship then was|as he then was)\b"
)
SECTION_PREFIX = re.compile(
    r"^\s*(?:u\s*/\s*s\b\.?|sections?\b|secs?\b\.?|ss?\.)\s*", re.I
)
YEAR_SUFFIX = re.compile(r"\s(1[89]\d{2}|20\d{2})$")


def _words(value: str) -> str:
    """normalize_key without dropping parenthesised numbers."""
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return normalize_space(value)


def _replace_phrases(value: str, aliases: dict[str, str]) -> str:
    padded = f" {value} "
    for phrase, replacement in sorted(aliases.items(), key=lambda item: -len(item[0])):
        padded = padded.replace(f" {phrase} ", f" {replacement} ")
    return normalize_space(padded)


def statute_key(name: str) -> str:
    value = _words(re.sub(r"\([^)]*\)", " ", name))
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"\b(?:act\s+)?no\s+\d+\s+of\s+", "", value)
    value = STATUTE_ALIASES.get(value, value)
    year = YEAR_SUFFIX.search(value)
    if year:
        base = value[: year.start()]
        value = f"{STATUTE_ALIASES.get(base, base)} {year.group(1)}"
        # An alias carries its own year; keep the stated one.
        value = re.sub(r"(\s\d{4})(\s\d{4})$", r"\2", value)
    return value


def section_key(section: str) -> str:
    """'s. 10(2)', 'Sec.10 (2)' and 'Section 10(2)' all become '10(2)'."""
    value = SECTION_PREFIX.sub("", normalize_space(section))
    value = re.sub(r"\s+(?=[(\[])", "", value)
    value = re.sub(r"[\[\]]", lambda m: "(" if m.group(0) == "[" else ")", value)
    return re.sub(r"[^a-z0-9()]+", " ", value.casefold()).strip()


def litigant_key(name: str) -> str:
    value = _words(re.sub(r"\([^)]*\)", " ", name))
    value = _replace_phrases(value, CASE_WORD_ALIASES)
    return normalize_space(LITIGANT_NOISE.sub(" ", value))


def case_key(name: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", name)
    value = re.sub(r"\s+(?:v|vs|versus)\.?\s+", " v ", value, flags=re.I)
    sides = [litigant_key(side) for side in re.split(r"\s+v\s+", value)]
    return " v ".join(side for side in sides if side)


def judge_key(name: str) -> str:
    value = JUDGE_TITLES.sub(" ", _words(name))
    # Spacing between initials varies ("S.R. Das", "S. R. Das", "SR Das") and
    # letter-spaced signatures lose all spaces, so compare without spaces.
    return value.replace(" ", "")


def court_key(name: str) -> str:
    value = re.sub(r"^the\s+", "", _words(name))
    return "supreme court of india" if value == "supreme court" else value


def canonical_key(node_type: str, label: str) -> str:
    """Matching key for a node; the display label is kept separately."""
    canonicalisers = {
        "STATUTE": statute_key,
        "CASE": case_key,
        "PARTY": litigant_key,
        "JUDGE": judge_key,
        "COURT": court_key,
    }
    canonicaliser = canonicalisers.get(node_type)
    key = canonicaliser(label) if canonicaliser else ""
    return key or normalize_key(label)


CITATION_PATTERNS = [
    # 2025 INSC 174
    (re.compile(r"\b((?:19|20)\d{2})\s+INSC\s+(\d+)\b", re.I), "insc:{0}:{1}"),
    # (2019) 5 SCC 1, 2019 (5) SCC 1, (2019) 5 SCC (Cri) 1 is left out on purpose
    (
        re.compile(r"\(?((?:19|20)\d{2})\)?\s*\(?(\d{1,2})\)?\s+SCC\s+(\d+)\b", re.I),
        "scc:{0}:{1}:{2}",
    ),
    # [1955] 1 SCR 829, 1955 SCR (1) 829
    (
        re.compile(r"\[?((?:19|20)\d{2})\]?\s*\(?(\d{1,2})\)?\s+SCR\s+(\d+)\b", re.I),
        "scr:{0}:{1}:{2}",
    ),
    (
        re.compile(r"\b((?:19|20)\d{2})\s+SCR\s*\((\d{1,2})\)\s*(\d+)\b", re.I),
        "scr:{0}:{1}:{2}",
    ),
    # AIR 1955 SC 352, 1955 AIR 352
    (re.compile(r"\bAIR\s+((?:19|20)\d{2})\s+SC\s+(\d+)\b", re.I), "air:{0}:sc:{1}"),
    (re.compile(r"\b((?:19|20)\d{2})\s+AIR\s+(\d+)\b", re.I), "air:{0}:sc:{1}"),
]


def citation_keys(citation: str) -> list[str]:
    """Reporter-independent keys for the citations in a string.

    "(2019) 5 SCC 1" -> "scc:2019:5:1"; "2025 INSC 174" -> "insc:2025:174".
    Unrecognised formats give no key, so they never cause a false match.
    """
    keys: list[str] = []
    for pattern, template in CITATION_PATTERNS:
        for match in pattern.finditer(citation):
            key = template.format(*(int(group) if group.isdigit() else group for group in match.groups()))
            if key not in keys:
                keys.append(key)
    return keys


def case_match_keys(name: str, citations: list[str]) -> list[str]:
    """Keys under which a CASE node can be matched across judgments."""
    keys = [key for citation in citations for key in citation_keys(citation)]
    name_key = case_key(name)
    if name_key:
        keys.append(f"name:{name_key}")
    return list(dict.fromkeys(keys))


def resolve_statute_keys(names: list[str]) -> dict[str, str]:
    """Give yearless statute names the year used elsewhere in the judgment.

    "Income Tax Act" becomes "income tax act 1961" when the judgment also cites
    "Income-tax Act, 1961" and no other year for that Act.
    """
    keys = {name: statute_key(name) for name in names}
    years: dict[str, set[str]] = {}
    for key in keys.values():
        match = YEAR_SUFFIX.search(key)
        if match:
            years.setdefault(key[: match.start()], set()).add(match.group(1))
    for name, key in keys.items():
        if not YEAR_SUFFIX.search(key) and len(years.get(key, ())) == 1:
            keys[name] = f"{key} {next(iter(years[key]))}"
    return keys


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def add_node(
        self,
        node_type: str,
        label: str,
        properties: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> str:
        key = f"{node_type}:{key or canonical_key(node_type, label)}"
        node_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        node = self.nodes.setdefault(
            node_id,
            {"id": node_id, "type": node_type, "label": normalize_space(label)},
        )
        if properties:
            for name, value in properties.items():
                if name == "is_current_case" and node.get(name):
                    continue  # a citation of the current case must not demote it
                if name == "match_keys" and node.get(name):
                    value = list(dict.fromkeys(node[name] + value))
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
    current_keys = case_match_keys(case["name"], case["citations"])
    case_id = graph.add_node(
        "CASE",
        case["name"],
        {
            "match_keys": current_keys,
            "citations": case["citations"],
            "date": case["date"],
            "is_current_case": True,
            "header_author": case.get("header_author", ""),
            "header_judges": case.get("header_judges", []),
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
        cited_keys = case_match_keys(cited["name"], [cited.get("citation", "")])
        # A reporter citation identifies a case better than its name, so it is
        # the node key when present; match_keys lets the importer join this
        # node to the cited judgment once that judgment is uploaded.
        cited_id = graph.add_node(
            "CASE",
            cited["name"],
            {
                "citation": cited.get("citation", ""),
                "is_current_case": False,
                "match_keys": cited_keys,
            },
            key=cited_keys[0] if cited_keys and not cited_keys[0].startswith("name:") else None,
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
            {
                "proposition": cited.get("proposition", ""),
                "treatment_source": cited.get("treatment_source", ""),
                "llm_treatment": cited.get("llm_treatment", ""),
            },
        )

    statute_keys = resolve_statute_keys([statute["name"] for statute in record["statutes"]])
    for statute in record["statutes"]:
        statute_key_value = statute_keys[statute["name"]]
        statute_id = graph.add_node("STATUTE", statute["name"], key=statute_key_value)
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
                "SECTION",
                section_label,
                {"section": section},
                key=f"{section_key(section)}|{statute_key_value}",
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
    if not record["case"]["name"]:
        record["case"]["name"] = normalize_space(pdf_path.stem.replace("_", " "))
    nodes, edges = build_graph(record)

    output = {
        "metadata": {
            "schema_version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "document_id": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
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
