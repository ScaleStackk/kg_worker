# PDF knowledge graph container

This is a lightweight Python KG worker image. Ollama and its model are
intentionally external services, matching the later AWS/Kubernetes design where
multiple workers can share an inference service.

The extractor uses a hybrid small-model pipeline:

- deterministic parsing for case identity, court, date, citations, bench, and parties;
- complete-page semantic batches instead of fixed character cuts;
- constrained JSON extraction for facts, issues, holdings, rules, procedure,
  authorities, statutes, and outcome;
- source-quote validation with page provenance;
- deterministic entity resolution and graph construction.

## Run locally

Start Ollama and process a judgment:

```bash
ollama serve
python kg_create.py /path/to/judgment.pdf \
  --output knowledge_graph.json \
  --debug-output raw_extraction.json
```

The output contains `metadata`, a query-friendly `case_record`, and `nodes` plus
`edges` for graph storage. The optional debug file preserves the model response
before evidence validation.

## Build a transferable image

The supplied build script targets standard x86-64 EC2 instances and exports a
Docker image archive:

```bash
./build-image.sh
```

This creates `legal-kg-worker-amd64.tar`. It contains no Ollama binaries or
model weights.

For Graviton instances, build an ARM64 archive instead:

```bash
PLATFORM=linux/arm64 \
OUTPUT_TAR=legal-kg-worker-arm64.tar \
./build-image.sh
```

## Run with Ollama on the host

On macOS, `host.docker.internal` routes the worker to the host's Ollama service:

```bash
docker run --rm \
  --volume "$PWD/data:/data" \
  --env OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  legal-kg-worker:latest
```

The result is written to `data/knowledge_graph.json`. The container also
auto-detects a PDF when `/data` contains exactly one PDF, so renaming it to
`input.pdf` is optional.

To specify paths explicitly:

```bash
docker run --rm \
  --volume "$PWD/data:/data" \
  --env OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  legal-kg-worker:latest \
  /data/judgment.pdf --output /data/judgment-graph.json
```

In Docker Compose or Kubernetes, use the inference service DNS name instead,
for example `http://ollama:11434/api/generate`. On AWS, publish the worker image
to ECR and configure `OLLAMA_URL` through the Pod or task environment. The
worker and inference containers can then be scaled independently.

All target instances must match the image architecture (`amd64` or `arm64`).
For a mixed fleet, publish a multi-platform worker image to ECR.

Scanned PDFs must be OCRed first because the extractor reads embedded PDF text.

## Run as the Kafka ingestion worker

Set `WORKER_MODE=kafka` to consume `judgment.uploaded` events continuously. The
consumer downloads each PDF from the event's private S3 location, runs the
existing extractor, imports the resulting graph into Neo4j, and commits the
Kafka offset only after the import succeeds. It uses the independent consumer
group `questacase-kg-worker-v1`, so the embedding group receives the same event.

On EC2, S3 credentials come from the instance IAM role through boto3. Do not put
long-lived AWS access keys in the image. For local S3-compatible storage, set
`S3_ENDPOINT` and `S3_FORCE_PATH_STYLE=true`.

Start Kafka, Neo4j, and Ollama on the shared `questacase` Docker network, then:

```bash
docker compose -f compose.yaml -f compose.kafka.yaml up --build
```

The consumer publishes `started` and `completed` records to `processing.status`.
After its bounded retries are exhausted, it publishes the error to
`processing.failed` before committing the poison event.

The `document.documentId` UUID from the upload event is passed to
`kg_create.py --document-id` and stored as `metadata.document_id`. The PDF hash
is retained separately as `metadata.content_sha256`. For manual one-file runs,
omitting `--document-id` preserves the original hash-derived ID behavior.
