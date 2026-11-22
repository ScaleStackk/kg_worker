#!/usr/bin/env python3
"""Consume uploaded judgments from Kafka and populate Neo4j."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DOCUMENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
DATA_ROOT = Path(os.environ.get("WORKER_DATA_DIR", "/data/jobs"))
PIPELINE = "knowledge_graph"
WORKER = "kg"
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


def required(name: str) -> str:
    value = text(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


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


def now() -> str:
    return datetime.now(UTC).isoformat()


def parse_upload_event(value: bytes | str | dict[str, Any]) -> UploadJob:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise TypeError("Kafka value must be a JSON object")
    if payload.get("eventType") != "judgment.uploaded":
        raise ValueError("Unsupported event type")

    document = payload.get("document")
    storage = payload.get("storage")
    if not isinstance(document, dict) or not isinstance(storage, dict):
        raise TypeError("Kafka event must contain document and storage objects")

    event_id = str(payload.get("eventId", "")).strip()
    document_id = str(document.get("documentId", "")).strip()
    filename = Path(str(document.get("originalFilename", "source.pdf"))).name
    content_type = str(document.get("contentType", "")).lower()
    try:
        size_bytes = int(document.get("sizeBytes", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("Kafka event has an invalid PDF size") from error
    bucket = str(storage.get("bucket", "")).strip()
    key = str(storage.get("key", "")).strip()
    uri = str(storage.get("uri", f"s3://{bucket}/{key}")).strip()

    if not event_id:
        raise ValueError("Kafka event is missing eventId")
    if not DOCUMENT_ID.fullmatch(document_id):
        raise ValueError("Invalid document ID")
    if content_type != "application/pdf" or size_bytes < 1:
        raise ValueError("Kafka event does not describe a non-empty PDF")
    if not bucket or not key:
        raise ValueError("S3 bucket and key are required")
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
    brokers = [item.strip() for item in required("KAFKA_BROKERS").split(",") if item.strip()]
    if not brokers:
        raise ValueError("KAFKA_BROKERS must contain at least one broker")

    ssl_enabled = boolean("KAFKA_SSL", False)
    mechanism = text("KAFKA_SASL_MECHANISM").upper()
    if mechanism:
        security_protocol = "SASL_SSL" if ssl_enabled else "SASL_PLAINTEXT"
    else:
        security_protocol = "SSL" if ssl_enabled else "PLAINTEXT"
    options: dict[str, Any] = {
        "bootstrap_servers": brokers,
        "security_protocol": security_protocol,
    }
    if mechanism:
        username = required("KAFKA_SASL_USERNAME")
        password = required("KAFKA_SASL_PASSWORD")
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
        region_name=text("AWS_REGION") or None,
        endpoint_url=text("S3_ENDPOINT") or None,
        config=Config(s3={"addressing_style": addressing_style}),
    )


def publish(producer: Any, topic: str, key: str, payload: dict[str, Any]) -> None:
    producer.send(topic, key=key, value=payload).get(timeout=30)


def status_event(
    job: UploadJob | None,
    status: str,
    *,
    elapsed_seconds: float | None = None,
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "1.0",
        "eventType": "processing.failed" if status == "failed" else "processing.status",
        "eventId": str(uuid.uuid4()),
        "occurredAt": now(),
        "documentId": job.document_id if job else None,
        "sourceEventId": job.event_id if job else None,
        "worker": WORKER,
        "pipeline": PIPELINE,
        "status": status,
    }
    if elapsed_seconds is not None:
        payload["elapsedSeconds"] = round(elapsed_seconds, 3)
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = str(error)[:1000]
        payload["errorType"] = type(error).__name__
    return payload


def process(job: UploadJob, s3: Any) -> dict[str, int]:
    job_dir = DATA_ROOT / job.document_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    pdf = job_dir / "source.pdf"
    graph = job_dir / "knowledge_graph.json"
    debug = job_dir / "raw_extraction.json"

    try:
        s3.download_file(job.bucket, job.key, str(pdf))
        downloaded_size = pdf.stat().st_size
        if downloaded_size != job.size_bytes:
            raise RuntimeError(
                f"S3 download size mismatch: expected {job.size_bytes}, got {downloaded_size}"
            )

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
        metadata["document_id"] = job.document_id
        metadata["source_pdf"] = job.original_filename
        graph.write_text(
            json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        subprocess.run([sys.executable, "/app/import_neo4j.py", str(graph)], check=True)
        return {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
        }
    finally:
        if not boolean("KEEP_JOB_FILES", False):
            shutil.rmtree(job_dir, ignore_errors=True)


def process_with_retries(job: UploadJob, s3: Any) -> dict[str, int]:
    attempts = integer("WORKER_MAX_ATTEMPTS", 3)
    retry_seconds = integer("WORKER_RETRY_SECONDS", 2)
    for attempt in range(1, attempts + 1):
        try:
            return process(job, s3)
        except Exception:
            if attempt == attempts:
                raise
            delay = min(30, retry_seconds * (2 ** (attempt - 1)))
            print(
                f"KG attempt {attempt}/{attempts} failed for {job.document_id}; "
                f"retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> None:
    from kafka import KafkaConsumer, KafkaProducer

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    topic = text("KAFKA_UPLOAD_TOPIC", "judgment.uploaded")
    group = text("KAFKA_CONSUMER_GROUP", "questacase-kg-workers")
    status_topic = text("KAFKA_STATUS_TOPIC", "processing.status")
    failed_topic = text("KAFKA_FAILED_TOPIC", "processing.failed")
    options = kafka_connection_options()

    consumer = KafkaConsumer(
        topic,
        group_id=group,
        client_id=text("KAFKA_CLIENT_ID", "questacase-kg-worker"),
        enable_auto_commit=False,
        auto_offset_reset=text("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        max_poll_interval_ms=integer("KAFKA_MAX_POLL_INTERVAL_MS", 21_600_000),
        max_poll_records=1,
        **options,
    )
    producer = KafkaProducer(
        client_id=f"{text('KAFKA_CLIENT_ID', 'questacase-kg-worker')}-status",
        acks="all",
        retries=5,
        key_serializer=lambda value: value.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode(
            "utf-8"
        ),
        **options,
    )
    s3 = s3_client()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"KG worker consuming {topic} as {group}", flush=True)

    try:
        while not STOP_REQUESTED:
            batches = consumer.poll(timeout_ms=1000, max_records=1)
            for messages in batches.values():
                for message in messages:
                    job: UploadJob | None = None
                    handled = False
                    started = time.perf_counter()
                    try:
                        job = parse_upload_event(message.value)
                        publish(
                            producer,
                            status_topic,
                            job.document_id,
                            status_event(job, "started"),
                        )
                        result = process_with_retries(job, s3)
                        publish(
                            producer,
                            status_topic,
                            job.document_id,
                            status_event(
                                job,
                                "completed",
                                elapsed_seconds=time.perf_counter() - started,
                                result=result,
                            ),
                        )
                        handled = True
                        print(
                            f"KG processing completed for {job.document_id}: {result}",
                            flush=True,
                        )
                    except Exception as error:  # noqa: BLE001 - publish worker failures
                        key = (
                            job.document_id
                            if job
                            else f"invalid-{message.partition}-{message.offset}"
                        )
                        print(
                            f"KG processing failed for {key}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        try:
                            publish(
                                producer,
                                failed_topic,
                                key,
                                status_event(
                                    job,
                                    "failed",
                                    elapsed_seconds=time.perf_counter() - started,
                                    error=error,
                                ),
                            )
                            handled = True
                        except Exception as publish_error:  # noqa: BLE001 - preserve offset
                            print(
                                "Could not publish failure status; offset will not be "
                                f"committed: {publish_error}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if handled:
                        consumer.commit()
    finally:
        producer.flush(timeout=10)
        producer.close(timeout=10)
        consumer.close(autocommit=False)


if __name__ == "__main__":
    main()
