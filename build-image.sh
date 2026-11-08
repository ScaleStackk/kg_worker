#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-legal-kg-worker:latest}"
platform="${PLATFORM:-linux/amd64}"
output_tar="${OUTPUT_TAR:-legal-kg-worker-amd64.tar}"

docker buildx build \
  --platform "${platform}" \
  --load \
  --tag "${image_name}" \
  .

docker image save --output "${output_tar}" "${image_name}"

echo "Created ${output_tar} (${platform})"
