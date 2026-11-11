#!/usr/bin/env python3
"""Consume uploaded-judgment events and persist extracted graphs in Neo4j."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINE = "knowledge_graph"
KG_CREATE_PATH = "/app/kg_create.py"
IMPORT_PATH = "/app/import_neo4j.py"
STOP_REQUESTED = False


@dataclass(frozen=True)
class UploadJob:
    event_id: str
    document_id: str
    original_filename: str
    size_bytes: int
    bucket: str
    key: str
    uri: str


def text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(text(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def boolean(name: str, default: bool = False) -> bool:
    value = text(name, str(default)).lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


def parse_upload_event(value: bytes | str | dict[str, Any]) -> UploadJob:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("Kafka value must be a JSON object")
    if payload.get("eventType") != "judgment.uploaded":
        raise ValueError("Kafka eventType must be judgment.uploaded")

    document = payload.get("document")
    storage = payload.get("storage")
    if not isinstance(document, dict) or not isinstance(storage, dict):
        raise ValueError("Kafka event must contain document and storage objects")

    event_id = str(payload.get("eventId", "")).strip()
    document_id = str(document.get("documentId", "")).strip()
    filename = Path(str(document.get("originalFilename", "source.pdf"))).name
    content_type = str(document.get("contentType", "")).lower()
    size_bytes = int(document.get("sizeBytes", 0))
    bucket = str(storage.get("bucket", "")).strip()
    key = str(storage.get("key", "")).strip()
    uri = str(storage.get("uri", f"s3://{bucket}/{key}")).strip()

    if not event_id or not document_id:
        raise ValueError("Kafka event is missing eventId or documentId")
    if content_type != "application/pdf" or size_bytes < 1:
        raise ValueError("Kafka event does not describe a non-empty PDF")
    if not bucket or not key:
        raise ValueError("Kafka event is missing its S3 bucket or key")
    if not filename.lower().endswith(".pdf"):
        filename = "source.pdf"
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename)[:200] or "source.pdf"

    return UploadJob(
        event_id=event_id,
        document_id=document_id,
        original_filename=filename,
        size_bytes=size_bytes,
        bucket=bucket,
        key=key,
        uri=uri,
    )


def kafka_connection_options() -> dict[str, Any]:
    ssl_enabled = boolean("KAFKA_SSL", False)
    mechanism = text("KAFKA_SASL_MECHANISM").upper()
    options: dict[str, Any] = {
        "bootstrap_servers": [
            broker.strip()
            for broker in text("KAFKA_BROKERS", "kafka:19092").split(",")
            if broker.strip()
        ],
        "security_protocol": (
            "SASL_SSL" if mechanism and ssl_enabled else
            "SASL_PLAINTEXT" if mechanism else
            "SSL" if ssl_enabled else
            "PLAINTEXT"
        ),
    }
    if mechanism:
        username = text("KAFKA_SASL_USERNAME")
        password = text("KAFKA_SASL_PASSWORD")
        if not username or not password:
            raise ValueError("Kafka SASL username and password are required")
        options.update(
            sasl_mechanism=mechanism,
            sasl_plain_username=username,
            sasl_plain_password=password,
        )
    ca_file = text("KAFKA_SSL_CA_FILE")
    if ca_file:
        options["ssl_cafile"] = ca_file
    return options


def s3_client() -> Any:
    import boto3
    from botocore.config import Config

    addressing_style = "path" if boolean("S3_FORCE_PATH_STYLE", False) else "auto"
    return boto3.client(
        "s3",
        region_name=text("AWS_REGION", "ap-south-1"),
        endpoint_url=text("S3_ENDPOINT") or None,
        config=Config(s3={"addressing_style": addressing_style}),
    )


def run_pipeline(job: UploadJob, client: Any) -> dict[str, int]:
    if not text("NEO4J_URL"):
        raise RuntimeError("NEO4J_URL is required in Kafka worker mode")

    temp_parent = text("WORKER_TEMP_DIRECTORY") or None
    with tempfile.TemporaryDirectory(prefix="questacase-kg-", dir=temp_parent) as directory:
        work = Path(directory)
        pdf_path = work / job.original_filename
        graph_path = work / "knowledge_graph.json"
        debug_path = work / "raw_extraction.json"

        client.download_file(job.bucket, job.key, str(pdf_path))
        downloaded_size = pdf_path.stat().st_size
        if downloaded_size != job.size_bytes:
            raise RuntimeError(
                f"S3 download size mismatch: expected {job.size_bytes}, got {downloaded_size}"
            )

        subprocess.run(
            [
                sys.executable,
                KG_CREATE_PATH,
                str(pdf_path),
                "--output",
                str(graph_path),
                "--debug-output",
                str(debug_path),
                "--document-id",
                job.document_id,
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, IMPORT_PATH, str(graph_path)],
            check=True,
        )

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        return {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
        }


def status_event(
    job: UploadJob | None,
    status: str,
    *,
    details: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "1.0",
        "eventType": "processing.status" if status != "failed" else "processing.failed",
        "eventId": str(uuid.uuid4()),
        "occurredAt": datetime.now(timezone.utc).isoformat(),
        "pipeline": PIPELINE,
        "status": status,
        "sourceEventId": job.event_id if job else None,
        "documentId": job.document_id if job else None,
    }
    if details:
        payload["details"] = details
    if error:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
    return payload


def publish(producer: Any, topic: str, key: str, payload: dict[str, Any]) -> None:
    producer.send(
        topic,
        key=key.encode("utf-8"),
        value=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    ).get(timeout=30)


def process_with_retries(job: UploadJob, client: Any) -> dict[str, int]:
    attempts = integer("WORKER_MAX_ATTEMPTS", 3)
    delay = integer("WORKER_RETRY_SECONDS", 5)
    for attempt in range(1, attempts + 1):
        try:
            return run_pipeline(job, client)
        except Exception:
            if attempt == attempts:
                raise
            print(
                f"KG processing attempt {attempt}/{attempts} failed for "
                f"{job.document_id}; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> int:
    from kafka import KafkaConsumer, KafkaProducer

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    topic = text("KAFKA_UPLOAD_TOPIC", "judgment.uploaded")
    status_topic = text("KAFKA_STATUS_TOPIC", "processing.status")
    failed_topic = text("KAFKA_FAILED_TOPIC", "processing.failed")
    group_id = text("KAFKA_CONSUMER_GROUP", "questacase-kg-worker-v1")
    options = kafka_connection_options()
    consumer = KafkaConsumer(
        topic,
        group_id=group_id,
        client_id=text("KAFKA_CLIENT_ID", "questacase-kg-worker"),
        enable_auto_commit=False,
        auto_offset_reset=text("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        max_poll_records=1,
        max_poll_interval_ms=integer("KAFKA_MAX_POLL_INTERVAL_MS", 3_600_000),
        **options,
    )
    producer = KafkaProducer(
        client_id=f"{text('KAFKA_CLIENT_ID', 'questacase-kg-worker')}-status",
        acks="all",
        retries=5,
        **options,
    )
    client = s3_client()
    print(
        f"KG consumer ready: topic={topic} group={group_id}",
        flush=True,
    )

    try:
        while not STOP_REQUESTED:
            batches = consumer.poll(timeout_ms=1000, max_records=1)
            for messages in batches.values():
                for message in messages:
                    job: UploadJob | None = None
                    handled = False
                    try:
                        job = parse_upload_event(message.value)
                        publish(
                            producer,
                            status_topic,
                            f"{job.document_id}:{PIPELINE}",
                            status_event(job, "started"),
                        )
                        details = process_with_retries(job, client)
                        publish(
                            producer,
                            status_topic,
                            f"{job.document_id}:{PIPELINE}",
                            status_event(job, "completed", details=details),
                        )
                        handled = True
                        print(
                            f"KG processing completed for {job.document_id}: {details}",
                            flush=True,
                        )
                    except Exception as error:
                        key = job.document_id if job else f"invalid-{message.partition}-{message.offset}"
                        print(f"KG processing failed for {key}: {error}", file=sys.stderr, flush=True)
                        try:
                            publish(
                                producer,
                                failed_topic,
                                f"{key}:{PIPELINE}",
                                status_event(job, "failed", error=error),
                            )
                            handled = True
                        except Exception as publish_error:
                            print(
                                f"Could not publish failure status; offset will not be committed: "
                                f"{publish_error}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if handled:
                        consumer.commit()
    finally:
        producer.flush(timeout=10)
        producer.close(timeout=10)
        consumer.close(autocommit=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
