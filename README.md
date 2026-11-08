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
