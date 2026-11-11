#!/usr/bin/env python3
"""Wait for an external Ollama service, then run the KG worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


APP_PATH = "/app/kg_create.py"
IMPORT_PATH = "/app/import_neo4j.py"
DATA_DIR = Path("/data")


def ollama_tags_url() -> str:
    generate_url = os.environ.get(
        "OLLAMA_URL", "http://host.docker.internal:11434/api/generate"
    )
    parsed = urlsplit(generate_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/tags", "", ""))


def wait_for_ollama(timeout: int = 120) -> dict:
    tags_url = ollama_tags_url()
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(tags_url, timeout=3) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(1)

    raise TimeoutError(f"Ollama did not become ready at {tags_url}: {last_error}")


def default_arguments() -> list[str]:
    configured_input = os.environ.get("INPUT_PDF")
    if configured_input:
        pdf_path = Path(configured_input)
    else:
        conventional_path = DATA_DIR / "input.pdf"
        if conventional_path.is_file():
            pdf_path = conventional_path
        else:
            pdfs = sorted(DATA_DIR.glob("*.pdf"))
            if len(pdfs) != 1:
                raise RuntimeError(
                    "Mount exactly one PDF in /data, name it /data/input.pdf, "
                    "or set INPUT_PDF to its container path."
                )
            pdf_path = pdfs[0]

    output_path = os.environ.get(
        "OUTPUT_JSON", str(DATA_DIR / "knowledge_graph.json")
    )
    arguments = [str(pdf_path), "--output", output_path]
    document_id = os.environ.get("DOCUMENT_ID", "").strip()
    if document_id:
        arguments.extend(["--document-id", document_id])
    debug_path = os.environ.get("DEBUG_JSON")
    if debug_path:
        arguments.extend(["--debug-output", debug_path])
    return arguments


def model_is_available(tags: dict, model: str) -> bool:
    available = {
        item.get("name")
        for item in tags.get("models", [])
        if isinstance(item, dict)
    }
    return model in available or (":" not in model and f"{model}:latest" in available)


def main() -> None:
    mode = os.environ.get("WORKER_MODE", "oneshot").strip().lower()
    if mode not in {"oneshot", "kafka"}:
        raise ValueError("WORKER_MODE must be 'oneshot' or 'kafka'")

    tags = wait_for_ollama()
    model = os.environ.get("KG_MODEL", "qwen3:4b")
    if not model_is_available(tags, model):
        raise RuntimeError(
            f"Model {model!r} is unavailable at the configured Ollama service"
        )

    if mode == "kafka":
        os.execv(
            sys.executable,
            [sys.executable, "/app/kafka_consumer.py", *sys.argv[1:]],
        )

    arguments = sys.argv[1:] or default_arguments()
    completed = subprocess.run([sys.executable, APP_PATH, *arguments], check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

    if os.environ.get("NEO4J_URL"):
        output_path = os.environ.get("OUTPUT_JSON", str(DATA_DIR / "knowledge_graph.json"))
        for index, argument in enumerate(arguments[:-1]):
            if argument == "--output":
                output_path = arguments[index + 1]
                break
        imported = subprocess.run(
            [sys.executable, IMPORT_PATH, output_path], check=False
        )
        if imported.returncode:
            raise SystemExit(imported.returncode)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"KG worker startup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
