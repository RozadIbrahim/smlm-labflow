FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/smlm-labflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements/remote.txt /tmp/remote.txt

RUN echo "===== /tmp/remote.txt =====" && \
    cat /tmp/remote.txt && \
    echo "===== Python / pip versions =====" && \
    python --version && \
    python -m pip --version && \
    echo "===== Installing main Python dependencies =====" && \
    python -m pip install --no-cache-dir --prefer-binary -r /tmp/remote.txt && \
    echo "===== Installing hdfdict without dependency resolver =====" && \
    python -m pip install --no-cache-dir --no-deps hdfdict==0.3.1

RUN echo "===== Patching hdfdict version import =====" && \
    python - <<'PY'
from pathlib import Path
import site

replacement = '''# -- coding: utf-8 --
"""hdfdict helps h5py to dump and load python dictionaries."""

from importlib.metadata import version as _dist_version

from .hdfdict import load, dump

__version__ = _dist_version("hdfdict")
__all__ = ["load", "dump"]
'''

candidates = [
    Path(site_dir) / "hdfdict" / "__init__.py"
    for site_dir in site.getsitepackages()
]

for init_py in candidates:
    if init_py.exists():
        init_py.write_text(replacement, encoding="utf-8")
        print(f"Patched {init_py}")
        break
else:
    raise FileNotFoundError("Could not find hdfdict/__init__.py in site-packages")
PY

RUN echo "===== Import smoke test =====" && \
    python - <<'PY'
import torch
import yaml
import numpy
import pandas
import tifffile
import h5py
import hdfdict
import omegaconf

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("PyYAML:", yaml.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("h5py:", h5py.__version__)
print("hdfdict:", hdfdict.__version__)
print("hdfdict import: OK")
PY

COPY . /workspace/smlm-labflow

CMD ["python", "run_pipeline.py", "--help"]
