#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-smlm-labflow-vast}"

INPUT_PATH="${1:-}"
OUT_DIR="${2:-}"
PROFILE_PATH="${3:-profiles/standard_2d.yaml}"

if [[ -z "${INPUT_PATH}" || -z "${OUT_DIR}" ]]; then
  echo "Usage:"
  echo "  ./scripts/docker_run_vast.sh /absolute/path/to/movie.tif /absolute/path/to/output_dir profiles/standard_2d.yaml"
  exit 1
fi

INPUT_DIR="$(dirname "$(realpath "${INPUT_PATH}")")"
INPUT_FILE="$(basename "${INPUT_PATH}")"
OUT_DIR_ABS="$(realpath -m "${OUT_DIR}")"

mkdir -p "${OUT_DIR_ABS}"

echo "[INFO] Image:   ${IMAGE_NAME}"
echo "[INFO] Input:   ${INPUT_PATH}"
echo "[INFO] Output:  ${OUT_DIR_ABS}"
echo "[INFO] Profile: ${PROFILE_PATH}"

docker run --rm --gpus all \
  -v "${INPUT_DIR}:/data:ro" \
  -v "${OUT_DIR_ABS}:/results" \
  "${IMAGE_NAME}" \
  python3 run_pipeline.py infer \
    --input "/data/${INPUT_FILE}" \
    --out /results \
    --profile "${PROFILE_PATH}" \
    --backend liteloc
