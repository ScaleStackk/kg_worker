import json
from pathlib import Path

import pytest

import queue_worker


def upload_event() -> dict:
    return {
        "schemaVersion": "1.0",
        "eventType": "judgment.uploaded",
        "eventId": "event-12345678",
        "document": {
            "documentId": "document-12345678",
            "originalFilename": "Example Judgment.pdf",
            "contentType": "application/pdf",
            "sizeBytes": 4,
        },
        "storage": {
            "bucket": "judgment-bucket",
            "key": "judgments/document-12345678/source.pdf",
            "uri": "s3://judgment-bucket/judgments/document-12345678/source.pdf",
        },
    }


def test_parse_upload_event_matches_current_webapi_contract() -> None:
    job = queue_worker.parse_upload_event(json.dumps(upload_event()).encode())

    assert job.event_id == "event-12345678"
    assert job.document_id == "document-12345678"
    assert job.original_filename == "Example Judgment.pdf"
    assert job.size_bytes == 4
    assert job.bucket == "judgment-bucket"


def test_parse_upload_event_rejects_incomplete_pdf_metadata() -> None:
    event = upload_event()
    event["document"]["contentType"] = "text/plain"

    with pytest.raises(ValueError, match="non-empty PDF"):
        queue_worker.parse_upload_event(event)


def test_kafka_defaults_preserve_current_plaintext_infrastructure(monkeypatch) -> None:
    monkeypatch.setenv("KAFKA_BROKERS", "kafka-1:9092,kafka-2:9092")
    monkeypatch.delenv("KAFKA_SSL", raising=False)
    monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)

    options = queue_worker.kafka_connection_options()

    assert options == {
        "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
        "security_protocol": "PLAINTEXT",
    }


def test_kafka_sasl_ssl_is_optional_and_additive(monkeypatch) -> None:
    monkeypatch.setenv("KAFKA_BROKERS", "kafka.example:9093")
    monkeypatch.setenv("KAFKA_SSL", "true")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "plain")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "worker")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "secret")
    monkeypatch.setenv("KAFKA_SSL_CA_FILE", "/certs/ca.pem")

    options = queue_worker.kafka_connection_options()

    assert options["security_protocol"] == "SASL_SSL"
    assert options["sasl_mechanism"] == "PLAIN"
    assert options["sasl_plain_username"] == "worker"
    assert options["ssl_cafile"] == "/certs/ca.pem"


def test_status_event_keeps_existing_fields_and_adds_traceability() -> None:
    job = queue_worker.parse_upload_event(upload_event())

    payload = queue_worker.status_event(
        job,
        "completed",
        elapsed_seconds=1.2345,
        result={"nodes": 2, "edges": 1},
    )

    assert payload["eventType"] == "processing.status"
    assert payload["documentId"] == job.document_id
    assert payload["worker"] == "kg"
    assert payload["status"] == "completed"
    assert payload["result"] == {"nodes": 2, "edges": 1}
    assert payload["sourceEventId"] == job.event_id
    assert payload["pipeline"] == "knowledge_graph"
    assert payload["elapsedSeconds"] == 1.234


def test_process_preserves_uploaded_document_id(monkeypatch, tmp_path: Path) -> None:
    event = upload_event()
    job = queue_worker.parse_upload_event(event)
    monkeypatch.setattr(queue_worker, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("KEEP_JOB_FILES", "true")

    class FakeS3:
        def download_file(self, bucket: str, key: str, target: str) -> None:
            assert bucket == job.bucket
            assert key == job.key
            Path(target).write_bytes(b"%PDF")

    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        calls.append(command)
        if command[1] == "/app/kg_create.py":
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps({"metadata": {}, "nodes": [{}, {}], "edges": [{}]}),
                encoding="utf-8",
            )

    monkeypatch.setattr(queue_worker.subprocess, "run", fake_run)

    result = queue_worker.process(job, FakeS3())

    graph_path = tmp_path / job.document_id / "knowledge_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["metadata"]["document_id"] == job.document_id
    assert graph["metadata"]["source_pdf"] == job.original_filename
    assert result == {"nodes": 2, "edges": 1}
    assert calls[-1][1] == "/app/import_neo4j.py"


def test_process_rejects_s3_size_mismatch(monkeypatch, tmp_path: Path) -> None:
    job = queue_worker.parse_upload_event(upload_event())
    monkeypatch.setattr(queue_worker, "DATA_ROOT", tmp_path)

    class ShortDownload:
        def download_file(self, _bucket: str, _key: str, target: str) -> None:
            Path(target).write_bytes(b"x")

    with pytest.raises(RuntimeError, match="S3 download size mismatch"):
        queue_worker.process(job, ShortDownload())
