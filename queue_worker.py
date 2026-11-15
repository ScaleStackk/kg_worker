#!/usr/bin/env python3
"""Consume uploaded judgments from Kafka and populate Neo4j."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from kafka import KafkaConsumer, KafkaProducer


DOCUMENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
DATA_ROOT = Path(os.environ.get("WORKER_DATA_DIR", "/data/jobs"))


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def now() -> str:
    return datetime.now(UTC).isoformat()


def event_details(event: dict[str, Any]) -> tuple[str, str, str, str]:
    if event.get("eventType") != "judgment.uploaded":
        raise ValueError("Unsupported event type")
    document = event.get("document") or {}
    storage = event.get("storage") or {}
    document_id = str(document.get("documentId", "")).strip()
    filename = str(document.get("originalFilename", "source.pdf")).strip()
    bucket = str(storage.get("bucket", "")).strip()
    key = str(storage.get("key", "")).strip()
    if not DOCUMENT_ID.fullmatch(document_id):
        raise ValueError("Invalid document ID")
    if not bucket or not key:
        raise ValueError("S3 bucket and key are required")
    return document_id, filename, bucket, key


def publish(producer: KafkaProducer, topic: str, document_id: str, payload: dict[str, Any]) -> None:
    producer.send(topic, key=document_id, value=payload).get(timeout=30)


def process(event: dict[str, Any], s3: Any) -> tuple[str, dict[str, Any]]:
    document_id, filename, bucket, key = event_details(event)
    job_dir = DATA_ROOT / document_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    pdf = job_dir / "source.pdf"
    graph = job_dir / "knowledge_graph.json"
    debug = job_dir / "raw_extraction.json"

    s3.download_file(bucket, key, str(pdf))
    subprocess.run(
        [
            sys.executable,
            "/app/kg_create.py",
            str(pdf),
            "--output",
            str(graph),
            "--debug-output",
            str(debug),
        ],
        check=True,
    )

    graph_data = json.loads(graph.read_text(encoding="utf-8"))
    metadata = graph_data.setdefault("metadata", {})
    metadata["document_id"] = document_id
    metadata["source_pdf"] = filename
    graph.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")

    subprocess.run([sys.executable, "/app/import_neo4j.py", str(graph)], check=True)
    result = {
        "nodes": len(graph_data.get("nodes", [])),
        "edges": len(graph_data.get("edges", [])),
    }
    if os.environ.get("KEEP_JOB_FILES", "false").lower() not in {"1", "true", "yes"}:
        shutil.rmtree(job_dir)
    return document_id, result


def main() -> None:
    brokers = [item.strip() for item in required("KAFKA_BROKERS").split(",") if item.strip()]
    topic = os.environ.get("KAFKA_UPLOAD_TOPIC", "judgment.uploaded")
    group = os.environ.get("KAFKA_CONSUMER_GROUP", "questacase-kg-workers")
    status_topic = os.environ.get("KAFKA_STATUS_TOPIC", "processing.status")
    failed_topic = os.environ.get("KAFKA_FAILED_TOPIC", "processing.failed")
    attempts = int(os.environ.get("WORKER_MAX_ATTEMPTS", "3"))

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=brokers,
        group_id=group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=brokers,
        key_serializer=lambda value: value.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
    )
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"KG worker consuming {topic} as {group}", flush=True)

    for message in consumer:
        event = message.value
        document_id = str((event.get("document") or {}).get("documentId", "unknown"))
        started = time.perf_counter()
        error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                document_id, result = process(event, s3)
                publish(
                    producer,
                    status_topic,
                    document_id,
                    {
                        "schemaVersion": "1.0",
                        "eventType": "processing.status",
                        "eventId": str(uuid.uuid4()),
                        "occurredAt": now(),
                        "documentId": document_id,
                        "worker": "kg",
                        "status": "completed",
                        "elapsedSeconds": round(time.perf_counter() - started, 3),
                        "result": result,
                    },
                )
                error = None
                break
            except Exception as caught:
                error = caught
                print(f"KG attempt {attempt}/{attempts} failed for {document_id}: {caught}", file=sys.stderr, flush=True)
                if attempt < attempts:
                    time.sleep(min(30, 2**attempt))

        if error is not None:
            publish(
                producer,
                failed_topic,
                document_id,
                {
                    "schemaVersion": "1.0",
                    "eventType": "processing.failed",
                    "eventId": str(uuid.uuid4()),
                    "occurredAt": now(),
                    "documentId": document_id,
                    "worker": "kg",
                    "error": str(error)[:1000],
                },
            )
        consumer.commit()


if __name__ == "__main__":
    main()
