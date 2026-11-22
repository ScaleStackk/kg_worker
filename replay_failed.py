#!/usr/bin/env python3
"""Replay one failed KG event by its Kafka topic partition and offset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime

from kafka import KafkaConsumer, KafkaProducer, TopicPartition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, type=int)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--execute", action="store_true", help="Publish the replay; otherwise show a preview")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    brokers = [value.strip() for value in os.environ.get("KAFKA_BROKERS", "").split(",") if value.strip()]
    if not brokers:
        raise ValueError("KAFKA_BROKERS is required")
    failed_topic = os.environ.get("KAFKA_FAILED_TOPIC", "processing.failed")
    upload_topic = os.environ.get("KAFKA_UPLOAD_TOPIC", "judgment.uploaded")
    topic_partition = TopicPartition(failed_topic, args.partition)
    consumer = KafkaConsumer(
        bootstrap_servers=brokers,
        enable_auto_commit=False,
        auto_offset_reset="none",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    try:
        consumer.assign([topic_partition])
        consumer.seek(topic_partition, args.offset)
        records = consumer.poll(timeout_ms=10_000, max_records=1).get(topic_partition, [])
        record = next((item for item in records if item.offset == args.offset), None)
        if record is None:
            raise LookupError(f"No failure event at {failed_topic}[{args.partition}] offset {args.offset}")
        failure = record.value
        if failure.get("eventType") != "processing.failed" or failure.get("worker") != "kg":
            raise ValueError("The selected Kafka record is not a KG processing.failed event")
        source_event = failure.get("sourceEvent")
        if not isinstance(source_event, dict) or source_event.get("eventType") != "judgment.uploaded":
            raise ValueError("This failure record has no replayable sourceEvent; replay it from the original upload event")
        document_id = str(failure.get("documentId", ""))
        source_document_id = str((source_event.get("document") or {}).get("documentId", ""))
        if not document_id or source_document_id != document_id:
            raise ValueError("Failure and source event document IDs do not match")
        replay_event = {
            **source_event,
            "eventId": str(uuid.uuid4()),
            "occurredAt": datetime.now(UTC).isoformat(),
            "replayOf": {
                "topic": failed_topic,
                "partition": args.partition,
                "offset": args.offset,
            },
        }
        print(json.dumps({"documentId": document_id, "uploadTopic": upload_topic, "replayOf": replay_event["replayOf"]}))
        if not args.execute:
            print("Dry run only. Add --execute to publish this event.")
            return 0
    finally:
        consumer.close()

    producer = KafkaProducer(
        bootstrap_servers=brokers,
        key_serializer=lambda value: value.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
    )
    try:
        producer.send(upload_topic, key=document_id, value=replay_event).get(timeout=30)
        print(f"Replayed document {document_id}; the KG worker will process it again.")
    finally:
        producer.flush(timeout=30)
        producer.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Replay failed: {error}", file=sys.stderr)
        raise SystemExit(1)
