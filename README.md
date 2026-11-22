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

`KG_NUM_PREDICT` controls the maximum number of tokens generated for each
structured extraction response. It defaults to `3000`, which avoids the long
CPU inference times caused by the previous 6000-token ceiling. Increase it only
if a large judgment's JSON response is being truncated.

All target instances must match the image architecture (`amd64` or `arm64`).
For a mixed fleet, publish a multi-platform worker image to ECR.

Scanned PDFs must be OCRed first because the extractor reads embedded PDF text.

## Kafka worker mode

The image also includes `/app/queue_worker.py`. It consumes
`judgment.uploaded` using the `questacase-kg-workers` consumer group, downloads
the referenced PDF from S3 using the EC2 IAM role, runs extraction, imports the
validated graph into Neo4j, and publishes completion or failure events. Run it
by overriding the image entrypoint with `python /app/queue_worker.py` and set
`KAFKA_BROKERS`, `AWS_REGION`, `NEO4J_URL`, `NEO4J_PASSWORD`, and `OLLAMA_URL`.

### Failed-job artifacts and retention

Successful job directories are removed after import unless `KEEP_JOB_FILES=true`
is set. Failed jobs keep their local PDF and intermediate output so the last
attempt can be inspected. The worker removes job directories older than
`FAILED_JOB_RETENTION_DAYS` at startup; the default is 7 days and the value must
be at least 1. This retention also bounds artifacts kept with
`KEEP_JOB_FILES=true`. Ensure `WORKER_DATA_DIR` is on a volume with enough space
for up to seven days of failed-job artifacts, and restrict access to that volume
because it can contain judgment PDFs and raw model output.

Failure events include the original `judgment.uploaded` event as `sourceEvent`
(S3 location and document metadata only; the PDF itself is not placed on Kafka).
The worker log records the failure topic partition and offset for replay.
To replay exactly one KG failure, identify its `processing.failed` partition and
offset from your Kafka tooling, then run a dry run in the worker container:

```bash
docker compose exec kg-worker python /app/replay_failed.py \
  --partition 0 --offset 123
```

Review the printed document ID and replay source. Add `--execute` to publish a
new `judgment.uploaded` event:

```bash
docker compose exec kg-worker python /app/replay_failed.py \
  --partition 0 --offset 123 --execute
```

The replay uses the original S3 object and a fresh event ID; it does not alter or
delete the failure record. Re-importing is scoped to the document ID, so the KG
importer replaces that document's graph while leaving other judgments intact.
This replays only the KG consumer path; the independent embedding worker is not
triggered.

## Tests

Unit tests cover the deterministic parts of the extractor and mock the Ollama
call, so they run without a model:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Graph storage across judgments

`import_neo4j.py` connects judgments into one graph:

- CASE, STATUTE, SECTION, JUDGE and COURT nodes are shared. Their id is
  `shared:<hash of type + canonical key>` and they carry `document_ids`, the
  judgments that mention them. A cited case is keyed by its reporter or
  neutral citation when it has one, and every CASE node has `match_keys`
  (citation keys plus a name key). On import, a CASE node joins the existing
  node with an overlapping key, so a citation resolves to the cited judgment
  once that judgment is uploaded, in either upload order. The uploaded
  judgment's node has `judgment_document_id`. Two uploaded judgments are never
  merged.
- FACT, ISSUE, HOLDING, LEGAL_RULE, PROCEDURAL_EVENT, OUTCOME and PARTY nodes
  and all edges stay scoped to one document (`document_id`).
- Re-importing a document deletes only its scoped nodes and edges, removes it
  from shared nodes' `document_ids`, and deletes shared nodes nothing uses.

Data imported before this change is document-scoped; re-import every judgment
(or start from an empty database) to connect it. Integration tests against a
throwaway Neo4j are in `tests/test_neo4j_integration.py`.
