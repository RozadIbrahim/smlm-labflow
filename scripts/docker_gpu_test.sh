#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-smlm-labflow}"

docker run --rm --gpus all "${IMAGE_NAME}" python3 - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available inside the container.")
PY
