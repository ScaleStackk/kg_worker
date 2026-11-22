#!/usr/bin/env python3
"""Count how many judgment PDFs get each header field parsed.

Usage: python scripts/header_check.py <pdf-dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_create import extract_pdf, parse_header_metadata  # noqa: E402

FIELDS = ["name", "court", "date", "citations", "judges", "author", "parties"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--json", type=Path, help="Write per-file headers here")
    args = parser.parse_args()

    pdfs = sorted(p for p in args.pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    counts = dict.fromkeys(FIELDS, 0)
    long_names = 0
    results = {}
    for pdf in pdfs:
        header = parse_header_metadata(extract_pdf(pdf))
        results[pdf.name] = header
        for field in FIELDS:
            if field == "parties":
                counts[field] += len(header[field]) >= 2
            else:
                counts[field] += bool(header[field])
        long_names += len(header["name"]) > 200

    total = len(pdfs)
    for field in FIELDS:
        label = "parties (both sides)" if field == "parties" else field
        print(f"{label:22} {counts[field]:4}/{total}")
    print(f"{'name > 200 chars':22} {long_names:4}/{total}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
