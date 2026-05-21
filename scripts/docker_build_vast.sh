#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-smlm-labflow-vast}"
DOCKERFILE="docker/Dockerfile.vast"

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}" \
  .

echo "[OK] Built image: ${IMAGE_NAME}"
