# syntax=docker/dockerfile:1.7
# The worker image intentionally does not contain Ollama or any model weights.
FROM python:3.12-slim-bookworm

ENV KG_MODEL=qwen3:4b \
    KG_NUM_PREDICT=3000 \
    OLLAMA_URL=http://host.docker.internal:11434/api/generate \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY kg_create.py import_neo4j.py container_entrypoint.py queue_worker.py replay_failed.py ./

WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/container_entrypoint.py"]
