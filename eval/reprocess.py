#!/usr/bin/env python3
"""Re-run validation and graph building on saved raw model output.

The model call is the slow step. Save it once with
`kg_create.py --debug-output <stem>.raw.json`, then reprocess the same raw
output with any version of the deterministic code (header parsing, quote
validation, treatment rules, graph building) and score the result:

    python eval/reprocess.py <raw-dir> <pdf-dir> <out-dir>
    python eval/score.py <out-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_create import (  # noqa: E402
    build_graph,
    extract_pdf,
    merge_extractions,
    parse_header_metadata,
    validate_extraction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path, help="Directory of <stem>.raw.json files")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(args.raw_dir.glob("*.raw.json")):
        stem = raw_path.name[: -len(".raw.json")]
        pdf_path = next((p for p in args.pdf_dir.glob(f"{stem}.*") if p.suffix.lower() == ".pdf"), None)
        if pdf_path is None:
            print(f"skip {stem}: PDF not found", file=sys.stderr)
            continue
        pages = extract_pdf(pdf_path)
        header = parse_header_metadata(pages)
        raw_results = json.loads(raw_path.read_text(encoding="utf-8"))
        record, validation = validate_extraction(merge_extractions(raw_results), pages, header)
        nodes, edges = build_graph(record)
        output = {
            "metadata": {
                "document_id": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "source_pdf": pdf_path.name,
                "validation": validation,
            },
            "case_record": record,
            "nodes": nodes,
            "edges": edges,
        }
        (args.out_dir / f"{stem}.json").write_text(json.dumps(output, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
