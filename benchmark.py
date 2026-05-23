#!/usr/bin/env python3
"""
benchmark.py

Comprehensive benchmark utility for the SMLM lab pipeline.

It keeps the old RuntimeBenchmark API:

    bench = RuntimeBenchmark(out_dir=benchmarks_dir)
    with bench.stage("qc", batch_index=1):
        ...
    summary = bench.finalize()

But it also adds scientist-facing benchmark layers:

    1. runtime_benchmark.csv/.json
    2. resource_benchmark.csv/.json
    3. input_qc_benchmark.csv
    4. localization_qc_benchmark.csv
    5. resolution_benchmark.csv
    6. drift_benchmark.csv
    7. truth_benchmark.csv + truth_matching_pairs.csv
    8. export_validation.csv
    9. benchmark_summary.json
    10. machine_specs.json/.csv
    11. figures/*.png when matplotlib is available

Design principle:
    This script must never crash the whole pipeline just because an optional
    benchmark dependency is missing. Missing layers are written as
    status="not_available" or status="skipped".

Recommended location inside one parent run folder:

    parent_run_folder/
    ├── results/
    ├── benchmarks/
    │   ├── runtime_benchmark.csv
    │   ├── resource_benchmark.csv
    │   ├── input_qc_benchmark.csv
    │   ├── localization_qc_benchmark.csv
    │   ├── resolution_benchmark.csv
    │   ├── drift_benchmark.csv
    │   ├── truth_benchmark.csv
    │   ├── export_validation.csv
    │   ├── machine_specs.json
    │   ├── machine_specs.csv
    │   ├── benchmark_summary.json
    │   └── figures/
    ├── reports/
    └── registry/
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# Small utilities
# =============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def bytes_to_mb(value: Optional[int | float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) / (1024**2), 3)
    except Exception:
        return None


def mb_to_gb(value: Optional[int | float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) / 1024.0, 3)
    except Exception:
        return None


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def safe_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            clean[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            clean[key] = value
    return clean


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def read_rows_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_or_replace_rows_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    replace_where: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Write rows without silently losing previous batches.

    If replace_where is provided, existing rows matching all those key/value
    pairs are removed first, then the new rows are appended. This gives stable
    rerun behavior for a batch while preserving previous batches in the same
    CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_rows_csv(path)
    if replace_where:
        criteria = {key: "" if value is None else str(value) for key, value in replace_where.items()}

        def keep(row: Mapping[str, Any]) -> bool:
            for key, expected in criteria.items():
                if str(row.get(key, "")) != expected:
                    return True
            return False

        existing = [row for row in existing if keep(row)]

    merged: List[Mapping[str, Any]] = list(existing) + list(rows)
    write_rows_csv(merged, path)


def safe_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def safe_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def safe_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def safe_import_tifffile():
    try:
        import tifffile  # type: ignore

        return tifffile
    except Exception:
        return None


def safe_import_scipy_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return cKDTree
    except Exception:
        return None


def safe_import_pynvml():
    """
    Import NVIDIA Management Library Python bindings if available.

    Usually installed with:
        pip install nvidia-ml-py
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def numeric_percentile(values: Sequence[float], q: float) -> Optional[float]:
    np = safe_import_numpy()

    vals: List[float] = []
    for value in values:
        out = safe_float(value)
        if out is not None:
            vals.append(out)

    if not vals:
        return None

    if np is None:
        vals = sorted(vals)
        idx = int(round((q / 100.0) * (len(vals) - 1)))
        idx = max(0, min(idx, len(vals) - 1))
        return float(vals[idx])

    try:
        return float(np.percentile(np.asarray(vals, dtype=float), q))
    except Exception:
        return None


# =============================================================================
# Machine specification capture
# =============================================================================


PACKAGE_VERSION_NAMES: Sequence[str] = (
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "scipy",
    "scikit-image",
    "tifffile",
    "h5py",
    "h5py-wrapper",
    "hdfdict",
    "omegaconf",
    "PyYAML",
    "psutil",
    "natsort",
    "thop",
)

REPRODUCIBILITY_ENV_VARS: Sequence[str] = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_GPUS",
)


def package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def read_linux_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return ""
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return ""
    return ""


def run_command_capture(args: Sequence[str], timeout_sec: float = 3.0) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def collect_cpu_specs(psutil_module: Any) -> Dict[str, Any]:
    cpu_model = (
        read_linux_cpu_model()
        or platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER", "")
    )
    specs: Dict[str, Any] = {
        "model": cpu_model,
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "frequency_current_mhz": None,
        "frequency_min_mhz": None,
        "frequency_max_mhz": None,
    }

    if psutil_module is not None:
        try:
            specs["physical_cores"] = psutil_module.cpu_count(logical=False)
            specs["logical_cores"] = psutil_module.cpu_count(logical=True)
        except Exception:
            pass
        try:
            freq = psutil_module.cpu_freq()
            if freq is not None:
                specs["frequency_current_mhz"] = safe_float(getattr(freq, "current", None))
                specs["frequency_min_mhz"] = safe_float(getattr(freq, "min", None))
                specs["frequency_max_mhz"] = safe_float(getattr(freq, "max", None))
        except Exception:
            pass

    return specs


def collect_memory_specs(psutil_module: Any) -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "ram_total_mb": None,
        "ram_total_gb": None,
        "ram_available_mb": None,
        "swap_total_mb": None,
        "swap_total_gb": None,
    }

    if psutil_module is None:
        return specs

    try:
        vm = psutil_module.virtual_memory()
        ram_total_mb = bytes_to_mb(getattr(vm, "total", None))
        specs["ram_total_mb"] = ram_total_mb
        specs["ram_total_gb"] = mb_to_gb(ram_total_mb)
        specs["ram_available_mb"] = bytes_to_mb(getattr(vm, "available", None))
    except Exception:
        pass

    try:
        swap = psutil_module.swap_memory()
        swap_total_mb = bytes_to_mb(getattr(swap, "total", None))
        specs["swap_total_mb"] = swap_total_mb
        specs["swap_total_gb"] = mb_to_gb(swap_total_mb)
    except Exception:
        pass

    return specs


def collect_disk_specs(path: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        total_mb = bytes_to_mb(usage.total)
        free_mb = bytes_to_mb(usage.free)
        used_mb = bytes_to_mb(usage.used)
        return {
            "path": str(path),
            "total_mb": total_mb,
            "total_gb": mb_to_gb(total_mb),
            "used_mb": used_mb,
            "free_mb": free_mb,
            "free_gb": mb_to_gb(free_mb),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "error": repr(exc),
        }


def collect_torch_cuda_specs(torch_module: Any) -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "torch_available": torch_module is not None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "current_device": None,
        "devices": [],
    }

    if torch_module is None:
        return specs

    try:
        specs["torch_version"] = getattr(torch_module, "__version__", None)
        specs["cuda_runtime_version"] = getattr(torch_module.version, "cuda", None)
    except Exception:
        pass

    try:
        cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
        if cudnn is not None:
            specs["cudnn_version"] = cudnn.version()
    except Exception:
        pass

    try:
        specs["cuda_available"] = bool(torch_module.cuda.is_available())
    except Exception:
        return specs

    if not specs["cuda_available"]:
        return specs

    try:
        specs["device_count"] = int(torch_module.cuda.device_count())
    except Exception:
        specs["device_count"] = 0

    try:
        specs["current_device"] = int(torch_module.cuda.current_device())
    except Exception:
        pass

    devices: List[Dict[str, Any]] = []
    for index in range(int(specs["device_count"] or 0)):
        device: Dict[str, Any] = {"index": index}
        try:
            props = torch_module.cuda.get_device_properties(index)
            total_mb = bytes_to_mb(getattr(props, "total_memory", None))
            device.update(
                {
                    "name": getattr(props, "name", None),
                    "total_memory_mb": total_mb,
                    "total_memory_gb": mb_to_gb(total_mb),
                    "compute_capability": (
                        f"{getattr(props, 'major', '')}.{getattr(props, 'minor', '')}"
                        if hasattr(props, "major") and hasattr(props, "minor")
                        else ""
                    ),
                    "multi_processor_count": getattr(
                        props, "multi_processor_count", None
                    ),
                }
            )
        except Exception as exc:
            device["error"] = repr(exc)
        devices.append(device)

    specs["devices"] = devices
    return specs


def collect_nvml_specs(pynvml_module: Any) -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "available": pynvml_module is not None,
        "driver_version": None,
        "device_count": 0,
        "devices": [],
    }

    if pynvml_module is None:
        return specs

    try:
        specs["driver_version"] = safe_str(pynvml_module.nvmlSystemGetDriverVersion())
    except Exception:
        pass

    try:
        specs["device_count"] = int(pynvml_module.nvmlDeviceGetCount())
    except Exception as exc:
        specs["error"] = repr(exc)
        return specs

    devices: List[Dict[str, Any]] = []
    for index in range(int(specs["device_count"] or 0)):
        device: Dict[str, Any] = {"index": index}
        try:
            handle = pynvml_module.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
            total_mb = bytes_to_mb(getattr(mem, "total", None))
            device.update(
                {
                    "name": safe_str(pynvml_module.nvmlDeviceGetName(handle)),
                    "total_memory_mb": total_mb,
                    "total_memory_gb": mb_to_gb(total_mb),
                }
            )
            try:
                device["uuid"] = safe_str(pynvml_module.nvmlDeviceGetUUID(handle))
            except Exception:
                pass
            try:
                pci = pynvml_module.nvmlDeviceGetPciInfo(handle)
                device["pci_bus_id"] = safe_str(getattr(pci, "busId", ""))
            except Exception:
                pass
            try:
                cc_major, cc_minor = pynvml_module.nvmlDeviceGetCudaComputeCapability(
                    handle
                )
                device["compute_capability"] = f"{cc_major}.{cc_minor}"
            except Exception:
                pass
            try:
                device["power_limit_w"] = round(
                    float(pynvml_module.nvmlDeviceGetPowerManagementLimit(handle))
                    / 1000.0,
                    3,
                )
            except Exception:
                pass
        except Exception as exc:
            device["error"] = repr(exc)
        devices.append(device)

    specs["devices"] = devices
    return specs


def collect_nvidia_smi_specs() -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    specs: Dict[str, Any] = {
        "available": bool(nvidia_smi),
        "path": nvidia_smi or "",
        "driver_version": None,
        "cuda_version_reported": None,
        "devices": [],
    }
    if not nvidia_smi:
        return specs

    query = run_command_capture(
        [
            nvidia_smi,
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_sec=5.0,
    )
    specs["query_returncode"] = query.get("returncode")
    specs["query_stderr"] = query.get("stderr", "")

    if query.get("available"):
        devices: List[Dict[str, Any]] = []
        for line in str(query.get("stdout", "")).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            memory_mb = safe_float(parts[5])
            devices.append(
                {
                    "index": safe_int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "pci_bus_id": parts[3],
                    "driver_version": parts[4],
                    "total_memory_mb": memory_mb,
                    "total_memory_gb": mb_to_gb(memory_mb),
                }
            )
        specs["devices"] = devices
        if devices:
            specs["driver_version"] = devices[0].get("driver_version")

    banner = run_command_capture([nvidia_smi], timeout_sec=5.0)
    specs["banner_returncode"] = banner.get("returncode")
    if banner.get("available"):
        banner_lines = str(banner.get("stdout", "")).splitlines()
        specs["banner_first_line"] = banner_lines[0] if banner_lines else ""
        for line in banner_lines[:5]:
            if "CUDA Version:" in line:
                specs["cuda_version_reported"] = (
                    line.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
                )
                break

    return specs


def collect_package_specs() -> Dict[str, Optional[str]]:
    return {name: package_version(name) for name in PACKAGE_VERSION_NAMES}


def collect_environment_specs() -> Dict[str, str]:
    return {
        name: os.environ.get(name, "")
        for name in REPRODUCIBILITY_ENV_VARS
        if os.environ.get(name, "") != ""
    }


def collect_container_specs() -> Dict[str, Any]:
    cgroup_path = Path("/proc/1/cgroup")
    cgroup = ""
    try:
        if cgroup_path.exists():
            cgroup = cgroup_path.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:
        cgroup = ""

    return {
        "in_container": Path("/.dockerenv").exists() or "docker" in cgroup.lower(),
        "hostname_env": os.environ.get("HOSTNAME", ""),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        "cgroup_excerpt": cgroup,
    }


def collect_machine_specs(benchmark_dir: Path) -> Dict[str, Any]:
    psutil_module = safe_import_psutil()
    torch_module = safe_import_torch()
    pynvml_module = safe_import_pynvml()

    return {
        "schema_version": "1.0",
        "captured_at": now_iso(),
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "python": {
            "version": sys.version.replace("\n", " "),
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": collect_cpu_specs(psutil_module),
        "memory": collect_memory_specs(psutil_module),
        "disk_for_benchmark_dir": collect_disk_specs(benchmark_dir),
        "torch_cuda": collect_torch_cuda_specs(torch_module),
        "nvml": collect_nvml_specs(pynvml_module),
        "nvidia_smi": collect_nvidia_smi_specs(),
        "packages": collect_package_specs(),
        "environment": collect_environment_specs(),
        "container": collect_container_specs(),
    }


def summarize_machine_specs(specs: Mapping[str, Any]) -> Dict[str, Any]:
    cpu = specs.get("cpu", {}) if isinstance(specs.get("cpu"), Mapping) else {}
    memory = specs.get("memory", {}) if isinstance(specs.get("memory"), Mapping) else {}
    torch_cuda = (
        specs.get("torch_cuda", {})
        if isinstance(specs.get("torch_cuda"), Mapping)
        else {}
    )
    nvml = specs.get("nvml", {}) if isinstance(specs.get("nvml"), Mapping) else {}
    nvidia_smi = (
        specs.get("nvidia_smi", {})
        if isinstance(specs.get("nvidia_smi"), Mapping)
        else {}
    )
    packages = (
        specs.get("packages", {})
        if isinstance(specs.get("packages"), Mapping)
        else {}
    )

    gpu_devices = (
        torch_cuda.get("devices")
        or nvml.get("devices")
        or nvidia_smi.get("devices")
        or []
    )
    gpu_names = [
        str(device.get("name", ""))
        for device in gpu_devices
        if isinstance(device, Mapping) and device.get("name")
    ]
    gpu_memory_gb = [
        safe_float(device.get("total_memory_gb"))
        for device in gpu_devices
        if isinstance(device, Mapping)
        and safe_float(device.get("total_memory_gb")) is not None
    ]

    return {
        "hostname": specs.get("hostname", ""),
        "os": specs.get("os", {}).get("platform", "")
        if isinstance(specs.get("os"), Mapping)
        else "",
        "cpu_model": cpu.get("model", ""),
        "cpu_physical_cores": cpu.get("physical_cores"),
        "cpu_logical_cores": cpu.get("logical_cores"),
        "ram_total_gb": memory.get("ram_total_gb"),
        "gpu_count": max(
            safe_int(torch_cuda.get("device_count"), 0) or 0,
            safe_int(nvml.get("device_count"), 0) or 0,
            len(gpu_names),
        ),
        "gpu_names": gpu_names,
        "gpu_total_memory_gb": gpu_memory_gb,
        "nvidia_driver_version": nvml.get("driver_version")
        or nvidia_smi.get("driver_version"),
        "cuda_runtime_version": torch_cuda.get("cuda_runtime_version"),
        "cudnn_version": torch_cuda.get("cudnn_version"),
        "torch_version": packages.get("torch") or torch_cuda.get("torch_version"),
    }


def machine_specs_csv_row(specs: Mapping[str, Any]) -> Dict[str, Any]:
    summary = summarize_machine_specs(specs)
    container = (
        specs.get("container", {})
        if isinstance(specs.get("container"), Mapping)
        else {}
    )
    environment = (
        specs.get("environment", {})
        if isinstance(specs.get("environment"), Mapping)
        else {}
    )
    packages = (
        specs.get("packages", {})
        if isinstance(specs.get("packages"), Mapping)
        else {}
    )
    python_specs = (
        specs.get("python", {}) if isinstance(specs.get("python"), Mapping) else {}
    )

    row: Dict[str, Any] = {
        "captured_at": specs.get("captured_at", ""),
        "hostname": summary.get("hostname", ""),
        "os": summary.get("os", ""),
        "python_version": python_specs.get("version", ""),
        "cpu_model": summary.get("cpu_model", ""),
        "cpu_physical_cores": summary.get("cpu_physical_cores"),
        "cpu_logical_cores": summary.get("cpu_logical_cores"),
        "ram_total_gb": summary.get("ram_total_gb"),
        "gpu_count": summary.get("gpu_count"),
        "gpu_names": "; ".join(summary.get("gpu_names", []) or []),
        "gpu_total_memory_gb": "; ".join(
            str(value) for value in (summary.get("gpu_total_memory_gb", []) or [])
        ),
        "nvidia_driver_version": summary.get("nvidia_driver_version", ""),
        "cuda_runtime_version": summary.get("cuda_runtime_version", ""),
        "cudnn_version": summary.get("cudnn_version", ""),
        "torch_version": summary.get("torch_version", ""),
        "in_container": container.get("in_container"),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": environment.get("NVIDIA_VISIBLE_DEVICES", ""),
    }

    for package_name in PACKAGE_VERSION_NAMES:
        row[f"package_{package_name}"] = packages.get(package_name)

    return row


def first_existing_column(
    columns: Iterable[str], candidates: Sequence[str]
) -> Optional[str]:
    cols = list(columns)
    lower_to_real = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        real = lower_to_real.get(candidate.lower())
        if real:
            return real
    return None


def detect_column_roles(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """
    Flexible SMLM column detection across LiteLoc/canonical/Picasso-like CSVs.
    """
    return {
        "frame": first_existing_column(
            columns, ["frame", "Frame", "frame_ix", "frame_idx", "t"]
        ),
        "x": first_existing_column(
            columns, ["x", "X", "x_nm", "xnm", "x [nm]", "x_px", "x_pix"]
        ),
        "y": first_existing_column(
            columns, ["y", "Y", "y_nm", "ynm", "y [nm]", "y_px", "y_pix"]
        ),
        "z": first_existing_column(
            columns, ["z", "Z", "z_nm", "znm", "z [nm]", "z_px"]
        ),
        "photons": first_existing_column(
            columns, ["photons", "Photon", "photons_total", "intensity", "I", "amp"]
        ),
        "background": first_existing_column(
            columns, ["background", "bg", "bkg", "offset", "noise"]
        ),
        "confidence": first_existing_column(
            columns, ["confidence", "prob", "probability", "score", "likelihood"]
        ),
        "lpx": first_existing_column(
            columns,
            [
                "lpx",
                "lpx_nm",
                "locprecnm",
                "loc_precision",
                "sigma_x",
                "uncertainty_x",
                "x_precision",
                "precision_x",
            ],
        ),
        "lpy": first_existing_column(
            columns,
            [
                "lpy",
                "lpy_nm",
                "locprecnm",
                "loc_precision",
                "sigma_y",
                "uncertainty_y",
                "y_precision",
                "precision_y",
            ],
        ),
        "lpz": first_existing_column(
            columns,
            ["lpz", "lpz_nm", "sigma_z", "uncertainty_z", "z_precision", "precision_z"],
        ),
    }


def read_table(path: Path):
    """
    Read a CSV-like table.

    Returns:
        pandas DataFrame if pandas is available.
        Otherwise list[dict].
    """
    pd = safe_import_pandas()
    path = path.expanduser().resolve()

    if pd is not None:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep="\t")

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table_columns(table: Any) -> List[str]:
    if hasattr(table, "columns"):
        return [str(c) for c in table.columns]
    if isinstance(table, list) and table:
        return list(table[0].keys())
    return []


def table_len(table: Any) -> int:
    try:
        return int(len(table))
    except Exception:
        return 0


def table_numeric_values(table: Any, col: Optional[str]) -> List[float]:
    if col is None:
        return []

    pd = safe_import_pandas()

    if pd is not None and hasattr(table, "columns"):
        if col not in table.columns:
            return []
        try:
            series = pd.to_numeric(table[col], errors="coerce")
            return [float(v) for v in series.dropna().to_list()]
        except Exception:
            return []

    values: List[float] = []
    if isinstance(table, list):
        for row in table:
            try:
                val = float(row.get(col, ""))
                if not math.isnan(val) and not math.isinf(val):
                    values.append(val)
            except Exception:
                continue
    return values


def table_xy_frame_arrays(
    table: Any,
    roles: Mapping[str, Optional[str]],
    coordinate_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
) -> Tuple[Any, Any, Any]:
    """
    Return aligned x/y/frame arrays in nanometers for image-based metrics.
    """
    np = safe_import_numpy()
    if np is None or roles.get("x") is None or roles.get("y") is None:
        return None, None, None

    x_col = roles["x"]
    y_col = roles["y"]
    frame_col = roles.get("frame")
    unit = str(coordinate_units or "nm").lower().strip()
    scale = 1.0
    if unit in {"pixel", "pixels", "px"}:
        pixel_size = safe_float(pixel_size_nm)
        if pixel_size is None or pixel_size <= 0:
            return None, None, None
        scale = float(pixel_size)

    pd = safe_import_pandas()
    if pd is not None and hasattr(table, "columns"):
        if x_col not in table.columns or y_col not in table.columns:
            return None, None, None
        try:
            x = pd.to_numeric(table[x_col], errors="coerce")
            y = pd.to_numeric(table[y_col], errors="coerce")
            mask = x.notna() & y.notna()
            frames = None
            if frame_col and frame_col in table.columns:
                frames_series = pd.to_numeric(table[frame_col], errors="coerce")
                frames = frames_series[mask].to_numpy(dtype=float)
            return (
                x[mask].to_numpy(dtype=float) * scale,
                y[mask].to_numpy(dtype=float) * scale,
                frames,
            )
        except Exception:
            return None, None, None

    if not isinstance(table, list):
        return None, None, None

    xs: List[float] = []
    ys: List[float] = []
    frames_list: List[float] = []
    have_frame = bool(frame_col)
    for row in table:
        try:
            x = float(row.get(x_col, ""))
            y = float(row.get(y_col, ""))
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                continue
            xs.append(x * scale)
            ys.append(y * scale)
            if have_frame:
                try:
                    frames_list.append(float(row.get(frame_col, "")))
                except Exception:
                    frames_list.append(float("nan"))
        except Exception:
            continue

    frames = np.asarray(frames_list, dtype=float) if have_frame else None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), frames


def next_power_of_two_at_least(value: int, minimum: int = 64, maximum: int = 1024) -> int:
    n = max(minimum, int(value))
    out = 1
    while out < n:
        out *= 2
    return min(maximum, out)


def maybe_plot_frc_curve(
    frequencies: Sequence[float],
    frc_values: Sequence[float],
    path: Path,
    threshold: float,
) -> Optional[str]:
    plt = safe_import_matplotlib()
    if plt is None or not frequencies or not frc_values:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(list(frequencies), list(frc_values), linewidth=1.5)
        plt.axhline(threshold, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Spatial frequency (cycles/nm)")
        plt.ylabel("FRC")
        plt.title("Fourier ring correlation")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


def estimate_frc_resolution(
    canonical_csv: Path,
    out_dir: Path,
    batch_index: Optional[int],
    coordinate_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
    threshold: float = 1.0 / 7.0,
    max_grid_size: int = 1024,
    min_render_pixel_nm: float = 5.0,
    min_localizations: int = 200,
) -> Dict[str, Any]:
    """
    Estimate 2D FRC resolution from two independent half maps.

    The split is odd/even frame when frames are available; otherwise a fixed
    deterministic random split is used. Coordinates are rendered into two
    histograms, Fourier transformed, radially averaged, and crossed against the
    1/7 threshold.
    """
    np = safe_import_numpy()
    figures_dir = ensure_figures_dir(out_dir)
    result: Dict[str, Any] = {
        "benchmark_layer": "resolution",
        "batch_index": batch_index,
        "metric": "frc_resolution_1_7",
        "value": None,
        "unit": "nm",
        "method": "two-half-map Fourier ring correlation, 1/7 threshold",
        "status": "not_available",
        "notes": "",
        "frc_threshold": threshold,
        "frc_curve_csv": "",
        "frc_curve_plot": "",
        "frc_split_method": "",
        "frc_render_pixel_nm": None,
        "frc_grid_size": None,
        "frc_n_localizations": 0,
    }

    if np is None:
        result["notes"] = "NumPy is required for FRC estimation."
        return result
    if not canonical_csv.exists():
        result["status"] = "failed"
        result["notes"] = "Canonical CSV not found."
        return result

    try:
        table = read_table(canonical_csv)
        roles = detect_column_roles(table_columns(table))
        x_nm, y_nm, frames = table_xy_frame_arrays(
            table,
            roles,
            coordinate_units=coordinate_units,
            pixel_size_nm=pixel_size_nm,
        )
        if x_nm is None or y_nm is None or len(x_nm) < min_localizations:
            result["notes"] = (
                f"Need at least {min_localizations} x/y localizations for FRC."
            )
            result["frc_n_localizations"] = 0 if x_nm is None else int(len(x_nm))
            return result

        x_nm = np.asarray(x_nm, dtype=float)
        y_nm = np.asarray(y_nm, dtype=float)
        result["frc_n_localizations"] = int(len(x_nm))

        if frames is not None and len(frames) == len(x_nm):
            finite_frames = np.isfinite(frames)
            if finite_frames.any() and len(set(frames[finite_frames].astype(int).tolist())) >= 2:
                frame_int = np.nan_to_num(frames, nan=0.0).astype(int)
                mask_a = (frame_int % 2) == 0
                mask_b = ~mask_a
                result["frc_split_method"] = "odd_even_frames"
            else:
                mask_a = np.zeros(len(x_nm), dtype=bool)
                mask_a[::2] = True
                mask_b = ~mask_a
                result["frc_split_method"] = "alternating_rows"
        else:
            rng = np.random.default_rng(12345)
            order = rng.permutation(len(x_nm))
            mask_a = np.zeros(len(x_nm), dtype=bool)
            mask_a[order[: len(order) // 2]] = True
            mask_b = ~mask_a
            result["frc_split_method"] = "deterministic_random_rows"

        if int(mask_a.sum()) < 50 or int(mask_b.sum()) < 50:
            result["notes"] = "Too few localizations in one FRC half map."
            return result

        x0 = float(np.nanmin(x_nm))
        y0 = float(np.nanmin(y_nm))
        x = x_nm - x0
        y = y_nm - y0
        width = float(np.nanmax(x) - np.nanmin(x))
        height = float(np.nanmax(y) - np.nanmin(y))
        fov_nm = max(width, height)
        if fov_nm <= 0:
            result["notes"] = "FRC requires nonzero XY field of view."
            return result

        render_pixel_nm = max(min_render_pixel_nm, fov_nm / float(max_grid_size - 1))
        grid_size = next_power_of_two_at_least(
            int(math.ceil(fov_nm / render_pixel_nm)) + 1,
            minimum=64,
            maximum=max_grid_size,
        )
        render_pixel_nm = fov_nm / float(max(grid_size - 1, 1))
        if render_pixel_nm <= 0:
            render_pixel_nm = min_render_pixel_nm

        extent_nm = float(grid_size) * render_pixel_nm
        hist_range = [[0.0, extent_nm], [0.0, extent_nm]]
        img_a, _, _ = np.histogram2d(
            y[mask_a], x[mask_a], bins=grid_size, range=hist_range
        )
        img_b, _, _ = np.histogram2d(
            y[mask_b], x[mask_b], bins=grid_size, range=hist_range
        )

        if not np.any(img_a) or not np.any(img_b):
            result["notes"] = "One FRC half map is empty after rendering."
            return result

        window_1d = np.hanning(grid_size)
        window = np.outer(window_1d, window_1d)
        img_a = (img_a - img_a.mean()) * window
        img_b = (img_b - img_b.mean()) * window

        fft_a = np.fft.fftshift(np.fft.fft2(img_a))
        fft_b = np.fft.fftshift(np.fft.fft2(img_b))
        freqs = np.fft.fftshift(np.fft.fftfreq(grid_size, d=render_pixel_nm))
        fx, fy = np.meshgrid(freqs, freqs)
        radial_freq = np.sqrt(fx**2 + fy**2)
        freq_step = 1.0 / (grid_size * render_pixel_nm)
        ring_index = np.floor(radial_freq / freq_step + 0.5).astype(int)

        curve_rows: List[Dict[str, Any]] = []
        frequency_values: List[float] = []
        frc_values: List[float] = []
        max_ring = grid_size // 2
        for ring in range(1, max_ring + 1):
            mask = ring_index == ring
            if not mask.any():
                continue
            a = fft_a[mask]
            b = fft_b[mask]
            denominator = math.sqrt(float(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)))
            if denominator <= 0:
                continue
            frc = float(np.real(np.sum(a * np.conj(b))) / denominator)
            frequency = float(ring * freq_step)
            frequency_values.append(frequency)
            frc_values.append(frc)
            curve_rows.append(
                {
                    "batch_index": batch_index,
                    "frequency_cycles_per_nm": frequency,
                    "frc": frc,
                    "threshold": threshold,
                }
            )

        if not curve_rows:
            result["notes"] = "FRC curve could not be computed."
            return result

        curve_csv = out_dir / f"frc_curve_batch_{batch_index or 0}.csv"
        write_rows_csv(curve_rows, curve_csv)
        result["frc_curve_csv"] = str(curve_csv)
        result["frc_curve_plot"] = (
            maybe_plot_frc_curve(
                frequency_values,
                frc_values,
                figures_dir / f"frc_curve_batch_{batch_index or 0}.png",
                threshold,
            )
            or ""
        )
        result["frc_render_pixel_nm"] = round(float(render_pixel_nm), 6)
        result["frc_grid_size"] = int(grid_size)

        crossing_frequency = None
        for i in range(1, len(frc_values)):
            prev_frc = frc_values[i - 1]
            curr_frc = frc_values[i]
            if prev_frc >= threshold and curr_frc <= threshold:
                f0 = frequency_values[i - 1]
                f1 = frequency_values[i]
                if curr_frc == prev_frc:
                    crossing_frequency = f1
                else:
                    alpha = (threshold - prev_frc) / (curr_frc - prev_frc)
                    crossing_frequency = f0 + alpha * (f1 - f0)
                break

        if crossing_frequency is not None and crossing_frequency > 0:
            result["value"] = round(float(1.0 / crossing_frequency), 6)
            result["status"] = "passed"
            result["notes"] = "FRC crossed the 1/7 threshold."
        else:
            result["status"] = "not_available"
            result["notes"] = "FRC curve did not cross the 1/7 threshold."

        return result

    except Exception as exc:
        result["status"] = "failed"
        result["notes"] = repr(exc)
        return result


def ensure_figures_dir(out_dir: Path) -> Path:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


# =============================================================================
# Optional plotting
# =============================================================================


def maybe_plot_hist(
    values: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 80,
) -> Optional[str]:
    plt = safe_import_matplotlib()
    clean_values = [v for v in values if safe_float(v) is not None]
    if plt is None or not clean_values:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.hist(clean_values, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


def maybe_plot_line(
    x: Sequence[float],
    y: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Optional[str]:
    plt = safe_import_matplotlib()
    if plt is None or not x or not y:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(list(x), list(y))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


def maybe_plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    path: Path,
    title: str,
    ylabel: str,
) -> Optional[str]:
    plt = safe_import_matplotlib()
    if plt is None or not labels or not values:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.bar(list(labels), list(values))
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


def maybe_plot_density(
    x: Sequence[float], y: Sequence[float], path: Path, title: str
) -> Optional[str]:
    plt = safe_import_matplotlib()
    pairs: List[Tuple[float, float]] = []
    for xv, yv in zip(x, y):
        xf = safe_float(xv)
        yf = safe_float(yv)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))

    if plt is None or not pairs:
        return None

    clean_x = [pair[0] for pair in pairs]
    clean_y = [pair[1] for pair in pairs]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.hist2d(clean_x, clean_y, bins=150)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


# =============================================================================
# GPU and resource monitoring
# =============================================================================


class GPUMonitor:
    """
    Optional GPU monitor.

    Priority:
        1. pynvml / NVIDIA Management Library for device-level metrics.
        2. torch.cuda for process-level PyTorch memory when available.
    """

    def __init__(self) -> None:
        self.pynvml = safe_import_pynvml()
        self.torch = safe_import_torch()
        self.nvml_available = self.pynvml is not None
        self.torch_available = self.torch is not None

    def cuda_available(self) -> bool:
        if self.torch is None:
            return False
        try:
            return bool(self.torch.cuda.is_available())
        except Exception:
            return False

    def sync(self) -> None:
        if not self.cuda_available():
            return
        try:
            self.torch.cuda.synchronize()
        except Exception:
            pass

    def reset_torch_peak(self) -> None:
        if not self.cuda_available():
            return
        try:
            self.torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def torch_snapshot(self) -> Dict[str, Optional[float]]:
        if not self.cuda_available():
            return {
                "torch_cuda_memory_allocated_mb": None,
                "torch_cuda_memory_reserved_mb": None,
                "torch_cuda_peak_memory_allocated_mb": None,
            }

        try:
            return {
                "torch_cuda_memory_allocated_mb": bytes_to_mb(
                    self.torch.cuda.memory_allocated()
                ),
                "torch_cuda_memory_reserved_mb": bytes_to_mb(
                    self.torch.cuda.memory_reserved()
                ),
                "torch_cuda_peak_memory_allocated_mb": bytes_to_mb(
                    self.torch.cuda.max_memory_allocated()
                ),
            }
        except Exception:
            return {
                "torch_cuda_memory_allocated_mb": None,
                "torch_cuda_memory_reserved_mb": None,
                "torch_cuda_peak_memory_allocated_mb": None,
            }

    def nvml_snapshot(self) -> Dict[str, Any]:
        if not self.nvml_available:
            return {
                "nvml_available": False,
                "gpu_count": 0,
                "gpu_name": None,
                "gpu_util_percent": None,
                "gpu_memory_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_temp_c": None,
                "gpu_power_w": None,
                "gpu_devices": [],
            }

        try:
            count = int(self.pynvml.nvmlDeviceGetCount())
            if count < 1:
                raise RuntimeError("NVML found no GPU devices")

            devices: List[Dict[str, Any]] = []
            row: Dict[str, Any] = {
                "nvml_available": True,
                "gpu_count": count,
                "gpu_devices": devices,
                "gpu_name": None,
                "gpu_util_percent": None,
                "gpu_memory_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_temp_c": None,
                "gpu_power_w": None,
                "gpu_max_util_percent": None,
                "gpu_mean_util_percent": None,
                "gpu_max_memory_util_percent": None,
                "gpu_total_mem_used_mb": None,
            }
            util_values: List[float] = []
            mem_util_values: List[float] = []
            used_values: List[float] = []

            for index in range(count):
                device: Dict[str, Any] = {"index": index}
                try:
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
                    name = safe_str(self.pynvml.nvmlDeviceGetName(handle))
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

                    temp_c = None
                    try:
                        temp_c = float(
                            self.pynvml.nvmlDeviceGetTemperature(
                                handle, self.pynvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    except Exception:
                        pass

                    power_w = None
                    try:
                        power_w = (
                            float(self.pynvml.nvmlDeviceGetPowerUsage(handle))
                            / 1000.0
                        )
                    except Exception:
                        pass

                    util_percent = float(getattr(util, "gpu", 0.0))
                    mem_util_percent = float(getattr(util, "memory", 0.0))
                    mem_used_mb = bytes_to_mb(getattr(mem, "used", None))
                    mem_total_mb = bytes_to_mb(getattr(mem, "total", None))
                    device.update(
                        {
                            "name": name,
                            "gpu_util_percent": util_percent,
                            "gpu_memory_util_percent": mem_util_percent,
                            "gpu_mem_used_mb": mem_used_mb,
                            "gpu_mem_total_mb": mem_total_mb,
                            "gpu_temp_c": temp_c,
                            "gpu_power_w": power_w,
                        }
                    )
                    row[f"gpu{index}_name"] = name
                    row[f"gpu{index}_util_percent"] = util_percent
                    row[f"gpu{index}_memory_util_percent"] = mem_util_percent
                    row[f"gpu{index}_mem_used_mb"] = mem_used_mb
                    row[f"gpu{index}_mem_total_mb"] = mem_total_mb
                    row[f"gpu{index}_temp_c"] = temp_c
                    row[f"gpu{index}_power_w"] = power_w
                    util_values.append(util_percent)
                    mem_util_values.append(mem_util_percent)
                    if mem_used_mb is not None:
                        used_values.append(float(mem_used_mb))
                except Exception as exc:
                    device["error"] = repr(exc)
                devices.append(device)

            if devices:
                first = devices[0]
                row["gpu_name"] = first.get("name")
                row["gpu_util_percent"] = first.get("gpu_util_percent")
                row["gpu_memory_util_percent"] = first.get("gpu_memory_util_percent")
                row["gpu_mem_used_mb"] = first.get("gpu_mem_used_mb")
                row["gpu_mem_total_mb"] = first.get("gpu_mem_total_mb")
                row["gpu_temp_c"] = first.get("gpu_temp_c")
                row["gpu_power_w"] = first.get("gpu_power_w")
            if util_values:
                row["gpu_max_util_percent"] = max(util_values)
                row["gpu_mean_util_percent"] = round(
                    sum(util_values) / len(util_values), 6
                )
            if mem_util_values:
                row["gpu_max_memory_util_percent"] = max(mem_util_values)
            if used_values:
                row["gpu_total_mem_used_mb"] = round(sum(used_values), 3)
            return row
        except Exception:
            return {
                "nvml_available": False,
                "gpu_count": 0,
                "gpu_name": None,
                "gpu_util_percent": None,
                "gpu_memory_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_temp_c": None,
                "gpu_power_w": None,
                "gpu_devices": [],
            }

    def snapshot(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        row.update(self.nvml_snapshot())
        row.update(self.torch_snapshot())
        return row


class ResourceSampler:
    """
    Background resource sampler for CPU/RAM/disk/GPU while a stage is running.
    """

    def __init__(
        self,
        out_dir: Path,
        sample_interval_sec: float = 1.0,
        enabled: bool = True,
        write_every_n_samples: int = 10,
    ) -> None:
        self.out_dir = out_dir
        self.sample_interval_sec = max(0.2, float(sample_interval_sec))
        self.enabled = enabled
        self.write_every_n_samples = max(1, int(write_every_n_samples))
        self.csv_path = self.out_dir / "resource_benchmark.csv"
        self.json_path = self.out_dir / "resource_benchmark.json"
        self.psutil = safe_import_psutil()
        self.gpu = GPUMonitor()
        self.process = None

        if self.psutil is not None:
            try:
                self.process = self.psutil.Process(os.getpid())
                self.psutil.cpu_percent(interval=None)
                self.process.cpu_percent(interval=None)
            except Exception:
                self.process = None

        self.rows: List[Dict[str, Any]] = []
        self._samples_since_write = 0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._active_stage: Dict[str, Any] = {}

    def process_tree(self):
        if self.process is None:
            return []
        try:
            return [self.process] + self.process.children(recursive=True)
        except Exception:
            return [self.process]

    def snapshot_process_resources(self) -> Dict[str, Optional[float]]:
        if self.psutil is None or self.process is None:
            return {
                "process_cpu_percent": None,
                "system_cpu_percent": None,
                "rss_mb": None,
                "vms_mb": None,
                "children_count": None,
                "disk_read_mb": None,
                "disk_write_mb": None,
            }

        rss = 0
        vms = 0
        read_bytes = 0
        write_bytes = 0
        child_count = 0

        try:
            system_cpu = float(self.psutil.cpu_percent(interval=None))
        except Exception:
            system_cpu = None

        try:
            process_cpu = float(self.process.cpu_percent(interval=None))
        except Exception:
            process_cpu = None

        for proc in self.process_tree():
            try:
                if proc.pid != self.process.pid:
                    child_count += 1
                with proc.oneshot():
                    mem = proc.memory_info()
                    rss += int(getattr(mem, "rss", 0) or 0)
                    vms += int(getattr(mem, "vms", 0) or 0)
                    try:
                        io = proc.io_counters()
                        read_bytes += int(getattr(io, "read_bytes", 0) or 0)
                        write_bytes += int(getattr(io, "write_bytes", 0) or 0)
                    except Exception:
                        pass
            except Exception:
                continue

        return {
            "process_cpu_percent": process_cpu,
            "system_cpu_percent": system_cpu,
            "rss_mb": bytes_to_mb(rss),
            "vms_mb": bytes_to_mb(vms),
            "children_count": float(child_count),
            "disk_read_mb": bytes_to_mb(read_bytes),
            "disk_write_mb": bytes_to_mb(write_bytes),
        }

    def snapshot(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"timestamp": now_iso(), **self._active_stage}
        row.update(self.snapshot_process_resources())
        row.update(self.gpu.snapshot())
        return row

    def _loop(self) -> None:
        if self._stop_event is None:
            return
        while not self._stop_event.is_set():
            try:
                self.rows.append(self.snapshot())
                self._samples_since_write += 1
                if self._samples_since_write >= self.write_every_n_samples:
                    self.write()
                    self._samples_since_write = 0
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_sec)

    def start(
        self,
        stage: str,
        batch_index: Optional[int] = None,
        input_path: Optional[str | Path] = None,
        out_dir: Optional[str | Path] = None,
    ) -> None:
        if not self.enabled:
            return

        self._active_stage = {
            "stage": stage,
            "batch_index": batch_index,
            "input_path": "" if input_path is None else str(input_path),
            "out_dir": "" if out_dir is None else str(out_dir),
        }
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return

        try:
            self.rows.append(self.snapshot())
            self.write()
            self._samples_since_write = 0
        except Exception:
            pass

        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_sec * 2))

        self._stop_event = None
        self._thread = None
        self._active_stage = {}

    def write(self) -> None:
        write_rows_csv(self.rows, self.csv_path)
        write_json({"samples": self.rows}, self.json_path)

    def summarize(self) -> Dict[str, Any]:
        def values_for(key: str) -> List[float]:
            return [
                safe_float(row.get(key))
                for row in self.rows
                if safe_float(row.get(key)) is not None
            ]  # type: ignore[list-item]

        def max_numeric(key: str) -> Optional[float]:
            values = values_for(key)
            return max(values) if values else None

        def mean_numeric(key: str) -> Optional[float]:
            values = values_for(key)
            if not values:
                return None
            return round(sum(values) / len(values), 6)

        return {
            "resource_csv": str(self.csv_path),
            "resource_json": str(self.json_path),
            "n_resource_samples": len(self.rows),
            "max_rss_mb": max_numeric("rss_mb"),
            "max_vms_mb": max_numeric("vms_mb"),
            "max_gpu_mem_used_mb": max_numeric("gpu_mem_used_mb"),
            "max_gpu_total_mem_used_mb": max_numeric("gpu_total_mem_used_mb"),
            "max_gpu_util_percent": max_numeric("gpu_util_percent"),
            "max_gpu_any_util_percent": max_numeric("gpu_max_util_percent"),
            "mean_gpu_util_percent": mean_numeric("gpu_util_percent"),
            "mean_gpu_any_util_percent": mean_numeric("gpu_mean_util_percent"),
            "max_torch_cuda_peak_memory_allocated_mb": max_numeric(
                "torch_cuda_peak_memory_allocated_mb"
            ),
        }


# =============================================================================
# Scientist-facing benchmark layer functions
# =============================================================================


def benchmark_input_movie(
    input_path: Path,
    out_dir: Path,
    batch_index: Optional[int] = None,
    max_sample_frames: int = 200,
    max_pixels_per_frame: int = 50_000,
) -> Dict[str, Any]:
    """Input QC benchmark for TIFF/OME-TIFF movies."""
    tifffile = safe_import_tifffile()
    np = safe_import_numpy()
    figures_dir = ensure_figures_dir(out_dir)

    row: Dict[str, Any] = {
        "benchmark_layer": "input_qc",
        "status": "not_available",
        "batch_index": batch_index,
        "input_path": str(input_path),
        "input_name": input_path.name,
        "shape": "",
        "dtype": "",
        "n_dimensions": None,
        "n_frames_guess": None,
        "pages": None,
        "sampled_frames": None,
        "sampling_strategy": "",
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p50": None,
        "p99": None,
        "p999": None,
        "zero_fraction": None,
        "saturated_fraction": None,
        "mean_frame_intensity_cv": None,
        "estimated_file_size_mb": None,
        "histogram_png": "",
        "mean_intensity_per_frame_png": "",
        "message": "",
    }

    try:
        row["estimated_file_size_mb"] = round(input_path.stat().st_size / (1024**2), 3)
    except Exception:
        pass

    if tifffile is None or np is None:
        row["status"] = "not_available"
        row["message"] = "tifffile and/or numpy not installed."
        write_or_replace_rows_csv([row], out_dir / "input_qc_benchmark.csv", {"benchmark_layer": "input_qc", "batch_index": batch_index, "input_path": str(input_path)})
        return row

    try:
        with tifffile.TiffFile(str(input_path), is_ome=False) as tif:
            page_count = int(len(tif.pages))
            row["pages"] = page_count

            if page_count <= 0:
                row["status"] = "failed"
                row["message"] = "TIFF has no readable pages."
                write_or_replace_rows_csv([row], out_dir / "input_qc_benchmark.csv", {"benchmark_layer": "input_qc", "batch_index": batch_index, "input_path": str(input_path)})
                return row

            first_page = tif.pages[0]
            first_shape = tuple(int(v) for v in first_page.shape)
            dtype = str(first_page.dtype)
            arr_dtype = np.dtype(first_page.dtype)
            row["dtype"] = dtype

            if page_count > 1 and len(first_shape) == 2:
                row["shape"] = [page_count, *first_shape]
                row["n_dimensions"] = 3
                row["n_frames_guess"] = page_count
                arr = None
            else:
                arr = np.asarray(first_page.asarray())
                shape = tuple(int(v) for v in arr.shape)
                row["shape"] = list(shape)
                row["n_dimensions"] = len(shape)
                row["n_frames_guess"] = int(shape[0]) if len(shape) >= 3 else 1

            if page_count > 1 and arr is None:
                n_frames = page_count
            else:
                n_frames = arr.shape[0] if arr is not None and arr.ndim >= 3 else 1

            if n_frames <= max_sample_frames:
                frame_indices = list(range(n_frames))
            else:
                frame_indices = (
                    np.linspace(0, n_frames - 1, max_sample_frames).astype(int).tolist()
                )

            row["sampled_frames"] = len(frame_indices)
            row["sampling_strategy"] = "direct_page_sampling" if page_count > 1 and arr is None else "single_page_array_sampling"

            samples = []
            frame_means = []
            rng = np.random.default_rng(42)
            for idx in frame_indices:
                if page_count > 1 and arr is None:
                    frame = np.asarray(tif.pages[int(idx)].asarray(), dtype=float)
                elif arr is not None and arr.ndim >= 3:
                    frame = np.asarray(arr[int(idx)], dtype=float)
                elif arr is not None:
                    frame = np.asarray(arr, dtype=float)
                else:
                    continue
                frame_means.append(float(np.mean(frame)))
                flat = frame.ravel()
                if flat.size > max_pixels_per_frame:
                    sample_idx = rng.choice(
                        flat.size, size=max_pixels_per_frame, replace=False
                    )
                    flat = flat[sample_idx]
                samples.append(flat)

            sample_values = (
                np.concatenate(samples) if samples else np.asarray([], dtype=float)
            )
            mean_frame = float(np.mean(frame_means)) if frame_means else 0.0
            row["mean_frame_intensity_cv"] = (
                float(np.std(frame_means) / mean_frame) if mean_frame else None
            )
            row["mean_intensity_per_frame_png"] = (
                maybe_plot_line(
                    x=[float(i) for i in frame_indices],
                    y=frame_means,
                    path=figures_dir
                    / f"input_mean_intensity_batch_{batch_index or 0}.png",
                    title="Mean intensity per sampled frame",
                    xlabel="Frame",
                    ylabel="Mean intensity",
                )
                or ""
            )

        sample_values = sample_values[~np.isnan(sample_values)]
        if sample_values.size == 0:
            row["status"] = "failed"
            row["message"] = "No numeric pixel values could be sampled."
        else:
            row["min"] = float(np.min(sample_values))
            row["max"] = float(np.max(sample_values))
            row["mean"] = float(np.mean(sample_values))
            row["std"] = float(np.std(sample_values))
            row["p01"] = float(np.percentile(sample_values, 1))
            row["p50"] = float(np.percentile(sample_values, 50))
            row["p99"] = float(np.percentile(sample_values, 99))
            row["p999"] = float(np.percentile(sample_values, 99.9))
            row["zero_fraction"] = float(np.mean(sample_values == 0))

            saturated_fraction = None
            try:
                if np.issubdtype(arr_dtype, np.integer):
                    max_possible = np.iinfo(arr_dtype).max
                    saturated_fraction = float(np.mean(sample_values >= max_possible))
            except Exception:
                saturated_fraction = None
            row["saturated_fraction"] = saturated_fraction

            row["histogram_png"] = (
                maybe_plot_hist(
                    values=[
                        float(v)
                        for v in sample_values[: min(sample_values.size, 1_000_000)]
                    ],
                    path=figures_dir / f"input_histogram_batch_{batch_index or 0}.png",
                    title="Input intensity histogram",
                    xlabel="Intensity",
                    bins=100,
                )
                or ""
            )
            row["status"] = "passed"
            row["message"] = "Input QC benchmark completed."

    except Exception as exc:
        row["status"] = "failed"
        row["message"] = repr(exc)

    write_or_replace_rows_csv([row], out_dir / "input_qc_benchmark.csv", {"benchmark_layer": "input_qc", "batch_index": batch_index, "input_path": str(input_path)})
    return row


def benchmark_localizations(
    canonical_csv: Path,
    out_dir: Path,
    batch_index: Optional[int] = None,
    coordinate_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
) -> Dict[str, Any]:
    """Canonical localization QC benchmark."""
    figures_dir = ensure_figures_dir(out_dir)
    row: Dict[str, Any] = {
        "benchmark_layer": "localization_qc",
        "status": "not_available",
        "batch_index": batch_index,
        "canonical_csv": str(canonical_csv),
        "coordinate_units": coordinate_units,
        "pixel_size_nm": pixel_size_nm,
        "n_localizations": 0,
        "n_columns": 0,
        "columns": "",
        "required_columns_ok": False,
        "missing_required_columns": "",
        "nan_coordinate_fraction": None,
        "n_frames": None,
        "median_localizations_per_frame": None,
        "mean_localizations_per_frame": None,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "z_min": None,
        "z_max": None,
        "median_photons": None,
        "median_background": None,
        "median_confidence": None,
        "median_lpx": None,
        "median_lpy": None,
        "median_lpz": None,
        "density_per_um2": None,
        "fig_localizations_per_frame": "",
        "fig_photons": "",
        "fig_background": "",
        "fig_precision": "",
        "fig_density": "",
        "message": "",
    }

    if not canonical_csv.exists():
        row["status"] = "failed"
        row["message"] = "Canonical CSV does not exist."
        write_or_replace_rows_csv([row], out_dir / "localization_qc_benchmark.csv", {"benchmark_layer": "localization_qc", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
        return row

    try:
        table = read_table(canonical_csv)
        cols = table_columns(table)
        roles = detect_column_roles(cols)
        row["n_localizations"] = table_len(table)
        row["n_columns"] = len(cols)
        row["columns"] = cols

        required = ["frame", "x", "y"]
        missing = [role for role in required if roles.get(role) is None]
        row["required_columns_ok"] = len(missing) == 0
        row["missing_required_columns"] = missing

        if missing:
            row["status"] = "failed"
            row["message"] = f"Missing required canonical columns: {missing}"
            write_or_replace_rows_csv([row], out_dir / "localization_qc_benchmark.csv", {"benchmark_layer": "localization_qc", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return row

        x_vals = table_numeric_values(table, roles["x"])
        y_vals = table_numeric_values(table, roles["y"])
        z_vals = table_numeric_values(table, roles["z"])
        frame_vals = table_numeric_values(table, roles["frame"])
        n = max(len(x_vals), len(y_vals), 1)
        row["nan_coordinate_fraction"] = round(
            float(1.0 - (min(len(x_vals), len(y_vals)) / n)), 8
        )

        if x_vals:
            row["x_min"] = min(x_vals)
            row["x_max"] = max(x_vals)
        if y_vals:
            row["y_min"] = min(y_vals)
            row["y_max"] = max(y_vals)
        if z_vals:
            row["z_min"] = min(z_vals)
            row["z_max"] = max(z_vals)

        if frame_vals:
            unique_frames = sorted(set(int(v) for v in frame_vals))
            row["n_frames"] = len(unique_frames)
            counts_by_frame: Dict[int, int] = {}
            for frame in frame_vals:
                f = int(frame)
                counts_by_frame[f] = counts_by_frame.get(f, 0) + 1
            counts = list(counts_by_frame.values())
            row["median_localizations_per_frame"] = numeric_percentile(counts, 50)
            row["mean_localizations_per_frame"] = (
                round(sum(counts) / len(counts), 6) if counts else None
            )
            row["fig_localizations_per_frame"] = (
                maybe_plot_line(
                    x=[float(k) for k in sorted(counts_by_frame)],
                    y=[float(counts_by_frame[k]) for k in sorted(counts_by_frame)],
                    path=figures_dir
                    / f"localizations_per_frame_batch_{batch_index or 0}.png",
                    title="Localizations per frame",
                    xlabel="Frame",
                    ylabel="Localizations",
                )
                or ""
            )

        photon_vals = table_numeric_values(table, roles["photons"])
        bg_vals = table_numeric_values(table, roles["background"])
        conf_vals = table_numeric_values(table, roles["confidence"])
        lpx_vals = table_numeric_values(table, roles["lpx"])
        lpy_vals = table_numeric_values(table, roles["lpy"])
        lpz_vals = table_numeric_values(table, roles["lpz"])
        row["median_photons"] = numeric_percentile(photon_vals, 50)
        row["median_background"] = numeric_percentile(bg_vals, 50)
        row["median_confidence"] = numeric_percentile(conf_vals, 50)
        row["median_lpx"] = numeric_percentile(lpx_vals, 50)
        row["median_lpy"] = numeric_percentile(lpy_vals, 50)
        row["median_lpz"] = numeric_percentile(lpz_vals, 50)

        row["fig_photons"] = (
            maybe_plot_hist(
                photon_vals,
                figures_dir / f"photon_distribution_batch_{batch_index or 0}.png",
                "Photon distribution",
                "Photons",
            )
            or ""
        )
        row["fig_background"] = (
            maybe_plot_hist(
                bg_vals,
                figures_dir / f"background_distribution_batch_{batch_index or 0}.png",
                "Background distribution",
                "Background",
            )
            or ""
        )
        precision_vals = [
            v for v in (lpx_vals + lpy_vals + lpz_vals) if safe_float(v) is not None
        ]
        row["fig_precision"] = (
            maybe_plot_hist(
                precision_vals,
                figures_dir
                / f"localization_precision_distribution_batch_{batch_index or 0}.png",
                "Localization precision distribution",
                f"Precision ({coordinate_units})",
            )
            or ""
        )

        if x_vals and y_vals:
            row["fig_density"] = (
                maybe_plot_density(
                    x_vals,
                    y_vals,
                    figures_dir / f"xy_density_batch_{batch_index or 0}.png",
                    "XY localization density",
                )
                or ""
            )
            width = max(x_vals) - min(x_vals)
            height = max(y_vals) - min(y_vals)
            if width > 0 and height > 0:
                if coordinate_units == "nm":
                    area_um2 = (width / 1000.0) * (height / 1000.0)
                elif coordinate_units == "pixel" and pixel_size_nm:
                    area_um2 = ((width * pixel_size_nm) / 1000.0) * (
                        (height * pixel_size_nm) / 1000.0
                    )
                else:
                    area_um2 = None
                if area_um2 and area_um2 > 0:
                    row["density_per_um2"] = round(
                        float(row["n_localizations"]) / area_um2, 6
                    )

        row["status"] = "passed"
        row["message"] = "Localization QC benchmark completed."

    except Exception as exc:
        row["status"] = "failed"
        row["message"] = repr(exc)

    write_or_replace_rows_csv([row], out_dir / "localization_qc_benchmark.csv", {"benchmark_layer": "localization_qc", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
    return row


def benchmark_resolution_proxy(
    localization_qc_row: Mapping[str, Any],
    canonical_csv: Optional[Path],
    out_dir: Path,
    batch_index: Optional[int] = None,
    coordinate_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Resolution proxy benchmark.

    Does not claim a true FRC result. It gives honest proxies:
        - median localization precision
        - density/Nyquist-like sampling proxy
    """
    rows: List[Dict[str, Any]] = []
    med_lpx = safe_float(localization_qc_row.get("median_lpx"))
    med_lpy = safe_float(localization_qc_row.get("median_lpy"))
    med_lpz = safe_float(localization_qc_row.get("median_lpz"))
    density = safe_float(localization_qc_row.get("density_per_um2"))
    xy_vals = [v for v in [med_lpx, med_lpy] if v is not None]
    median_xy_precision = sum(xy_vals) / len(xy_vals) if xy_vals else None

    rows.append(
        {
            "benchmark_layer": "resolution",
            "batch_index": batch_index,
            "metric": "median_xy_localization_precision",
            "value": median_xy_precision,
            "unit": "same_as_coordinates",
            "method": "median of lpx/lpy columns if available",
            "status": "passed" if median_xy_precision is not None else "not_available",
            "notes": "Precision proxy, not image resolution.",
        }
    )
    rows.append(
        {
            "benchmark_layer": "resolution",
            "batch_index": batch_index,
            "metric": "median_z_localization_precision",
            "value": med_lpz,
            "unit": "same_as_coordinates",
            "method": "median of lpz column if available",
            "status": "passed" if med_lpz is not None else "not_available",
            "notes": "3D precision proxy only.",
        }
    )

    sampling_resolution_nm = None
    if density is not None and density > 0:
        sampling_resolution_nm = 2.0 * 1000.0 / math.sqrt(density)
    rows.append(
        {
            "benchmark_layer": "resolution",
            "batch_index": batch_index,
            "metric": "sampling_limited_resolution_proxy",
            "value": sampling_resolution_nm,
            "unit": "nm",
            "method": "2 * 1000 / sqrt(localization_density_per_um2)",
            "status": "passed"
            if sampling_resolution_nm is not None
            else "not_available",
            "notes": "Sampling density proxy, not FRC.",
        }
    )
    if canonical_csv is not None:
        rows.append(
            estimate_frc_resolution(
                canonical_csv=canonical_csv,
                out_dir=out_dir,
                batch_index=batch_index,
                coordinate_units=coordinate_units,
                pixel_size_nm=pixel_size_nm,
            )
        )
    else:
        rows.append(
            {
                "benchmark_layer": "resolution",
                "batch_index": batch_index,
                "metric": "frc_resolution_1_7",
                "value": None,
                "unit": "nm",
                "method": "two-half-map Fourier ring correlation, 1/7 threshold",
                "status": "not_available",
                "notes": "Canonical CSV not supplied for FRC.",
            }
        )

    write_or_replace_rows_csv(rows, out_dir / "resolution_benchmark.csv", {"benchmark_layer": "resolution", "batch_index": batch_index})
    return {
        "status": "passed"
        if any(row.get("status") == "passed" for row in rows)
        else "not_available",
        "resolution_csv": str(out_dir / "resolution_benchmark.csv"),
        "metrics": rows,
    }


def benchmark_drift_proxy(
    canonical_csv: Path,
    out_dir: Path,
    batch_index: Optional[int] = None,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """
    Drift proxy benchmark.

    This is a frame-binned median XY centroid shift proxy. It is not true drift
    correction because sample structure can bias the centroid.
    """
    figures_dir = ensure_figures_dir(out_dir)
    output_csv = out_dir / "drift_benchmark.csv"
    base: Dict[str, Any] = {
        "benchmark_layer": "drift",
        "status": "not_available",
        "batch_index": batch_index,
        "canonical_csv": str(canonical_csv),
        "method": "frame-binned median XY centroid proxy",
        "max_abs_dx": None,
        "max_abs_dy": None,
        "max_radial_drift": None,
        "median_radial_drift": None,
        "p95_radial_drift": None,
        "linear_drift_slope_x_per_frame": None,
        "linear_drift_slope_y_per_frame": None,
        "linear_radial_drift_slope_per_frame": None,
        "drift_plot": "",
        "message": "",
    }

    if not canonical_csv.exists():
        base["status"] = "failed"
        base["message"] = "Canonical CSV not found."
        write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
        return base

    try:
        pd = safe_import_pandas()
        if pd is None:
            base["status"] = "not_available"
            base["message"] = "pandas not installed; drift proxy skipped."
            write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return base

        table = read_table(canonical_csv)
        roles = detect_column_roles(table_columns(table))
        if (
            roles.get("frame") is None
            or roles.get("x") is None
            or roles.get("y") is None
        ):
            base["status"] = "not_available"
            base["message"] = "frame/x/y columns not available."
            write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return base

        frame_col = roles["frame"]
        x_col = roles["x"]
        y_col = roles["y"]
        df = table.copy()
        df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.dropna(subset=[frame_col, x_col, y_col])

        if len(df) < 10:
            base["status"] = "not_available"
            base["message"] = "Too few localizations for drift proxy."
            write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return base

        min_frame = int(df[frame_col].min())
        max_frame = int(df[frame_col].max())
        if max_frame <= min_frame:
            base["status"] = "not_available"
            base["message"] = "Only one frame detected."
            write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return base

        bins = max(2, min(n_bins, max_frame - min_frame + 1))
        df["_drift_bin"] = pd.cut(
            df[frame_col], bins=bins, labels=False, include_lowest=True
        )
        grouped = (
            df.groupby("_drift_bin")
            .agg(
                frame_mid=(frame_col, "median"),
                median_x=(x_col, "median"),
                median_y=(y_col, "median"),
                n=(x_col, "size"),
            )
            .reset_index()
        )

        if len(grouped) < 2:
            base["status"] = "not_available"
            base["message"] = "Not enough populated frame bins."
            write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
            return base

        x0 = float(grouped["median_x"].iloc[0])
        y0 = float(grouped["median_y"].iloc[0])
        grouped["dx"] = grouped["median_x"] - x0
        grouped["dy"] = grouped["median_y"] - y0
        grouped["radial_drift"] = (grouped["dx"] ** 2 + grouped["dy"] ** 2) ** 0.5
        frame_span = float(grouped["frame_mid"].max() - grouped["frame_mid"].min())
        if frame_span > 0:
            np = safe_import_numpy()
            if np is not None:
                base["linear_drift_slope_x_per_frame"] = float(
                    np.polyfit(grouped["frame_mid"], grouped["dx"], deg=1)[0]
                )
                base["linear_drift_slope_y_per_frame"] = float(
                    np.polyfit(grouped["frame_mid"], grouped["dy"], deg=1)[0]
                )
                base["linear_radial_drift_slope_per_frame"] = float(
                    np.polyfit(grouped["frame_mid"], grouped["radial_drift"], deg=1)[0]
                )

        rows = grouped.to_dict(orient="records")
        for row in rows:
            row["benchmark_layer"] = "drift"
            row["status"] = "passed"
            row["batch_index"] = batch_index
            row["method"] = base["method"]

        base["status"] = "passed"
        base["max_abs_dx"] = float(grouped["dx"].abs().max())
        base["max_abs_dy"] = float(grouped["dy"].abs().max())
        base["max_radial_drift"] = float(grouped["radial_drift"].max())
        base["median_radial_drift"] = float(grouped["radial_drift"].median())
        base["p95_radial_drift"] = float(grouped["radial_drift"].quantile(0.95))
        base["message"] = "Drift proxy completed; interpret carefully."
        base["drift_plot"] = (
            maybe_plot_line(
                x=[float(v) for v in grouped["frame_mid"].tolist()],
                y=[float(v) for v in grouped["radial_drift"].tolist()],
                path=figures_dir / f"drift_proxy_batch_{batch_index or 0}.png",
                title="Drift proxy over time",
                xlabel="Frame",
                ylabel="Radial drift proxy",
            )
            or ""
        )

        write_or_replace_rows_csv([base] + rows, output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
        return base

    except Exception as exc:
        base["status"] = "failed"
        base["message"] = repr(exc)
        write_or_replace_rows_csv([base], output_csv, {"benchmark_layer": "drift", "batch_index": batch_index, "canonical_csv": str(canonical_csv)})
        return base


def validate_exports(
    exports: Mapping[str, str | Path | None], out_dir: Path
) -> Dict[str, Any]:
    """Validate downstream export files."""
    output_csv = out_dir / "export_validation.csv"
    rows: List[Dict[str, Any]] = []

    for name, value in exports.items():
        path = safe_path(value)
        row: Dict[str, Any] = {
            "benchmark_layer": "export_validation",
            "export_name": name,
            "path": "" if path is None else str(path),
            "exists": False,
            "rows": None,
            "columns": "",
            "columns_ok": False,
            "status": "skipped",
            "message": "",
        }

        if path is None:
            row["status"] = "skipped"
            row["message"] = "No path provided."
            rows.append(row)
            continue

        if not path.exists():
            row["status"] = "failed"
            row["message"] = "Export file does not exist."
            rows.append(row)
            continue

        row["exists"] = True
        try:
            table = read_table(path)
            cols = table_columns(table)
            row["rows"] = table_len(table)
            row["columns"] = cols
            roles = detect_column_roles(cols)

            if name in {"canonical", "picasso", "locan"}:
                ok = (
                    roles.get("frame") is not None
                    and roles.get("x") is not None
                    and roles.get("y") is not None
                )
            elif name == "napari":
                ok = roles.get("x") is not None and roles.get("y") is not None
            elif name == "smap":
                ok = table_len(table) > 0 and len(cols) > 0
            else:
                ok = table_len(table) > 0

            row["columns_ok"] = bool(ok)
            row["status"] = "passed" if ok else "warning"
            row["message"] = (
                "Export validation completed."
                if ok
                else "Export exists, but expected columns were not fully recognized."
            )
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = repr(exc)

        rows.append(row)

    write_or_replace_rows_csv(rows, output_csv, {"benchmark_layer": "export_validation"})
    return {
        "status": "passed"
        if all(r["status"] in {"passed", "skipped"} for r in rows)
        else "warning",
        "export_validation_csv": str(output_csv),
        "exports": rows,
    }


def benchmark_truth_matching(
    prediction_csv: Path,
    truth_csv: Path,
    out_dir: Path,
    batch_index: Optional[int] = None,
    match_radius_xy_nm: float = 50.0,
    match_radius_z_nm: float = 100.0,
) -> Dict[str, Any]:
    """Ground-truth benchmark for simulated/challenge/demo data."""
    output_csv = out_dir / "truth_benchmark.csv"
    pairs_csv = out_dir / "truth_matching_pairs.csv"
    row: Dict[str, Any] = {
        "benchmark_layer": "truth",
        "status": "not_available",
        "batch_index": batch_index,
        "prediction_csv": str(prediction_csv),
        "truth_csv": str(truth_csv),
        "match_radius_xy_nm": match_radius_xy_nm,
        "match_radius_z_nm": match_radius_z_nm,
        "n_pred": 0,
        "n_truth": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "jaccard": None,
        "rmse_xy": None,
        "rmse_z": None,
        "bias_x": None,
        "bias_y": None,
        "bias_z": None,
        "matching_pairs_csv": str(pairs_csv),
        "message": "",
    }

    if not prediction_csv.exists() or not truth_csv.exists():
        row["status"] = "not_available"
        row["message"] = "Prediction or truth CSV missing."
        write_or_replace_rows_csv([row], output_csv, {"benchmark_layer": "truth", "batch_index": batch_index, "prediction_csv": str(prediction_csv), "truth_csv": str(truth_csv)})
        write_rows_csv([], pairs_csv)
        return row

    try:
        pd = safe_import_pandas()
        np = safe_import_numpy()
        if pd is None or np is None:
            row["status"] = "not_available"
            row["message"] = "pandas/numpy required for truth benchmark."
            write_rows_csv([row], output_csv)
            write_rows_csv([], pairs_csv)
            return row

        pred = read_table(prediction_csv)
        truth = read_table(truth_csv)
        pred_roles = detect_column_roles(table_columns(pred))
        truth_roles = detect_column_roles(table_columns(truth))

        if pred_roles.get("x") is None or pred_roles.get("y") is None:
            row["status"] = "failed"
            row["message"] = "Prediction x/y columns missing."
            write_rows_csv([row], output_csv)
            write_rows_csv([], pairs_csv)
            return row
        if truth_roles.get("x") is None or truth_roles.get("y") is None:
            row["status"] = "failed"
            row["message"] = "Truth x/y columns missing."
            write_rows_csv([row], output_csv)
            write_rows_csv([], pairs_csv)
            return row

        use_z = pred_roles.get("z") is not None and truth_roles.get("z") is not None
        use_frame = (
            pred_roles.get("frame") is not None and truth_roles.get("frame") is not None
        )

        p = pred.copy()
        t = truth.copy()
        for col in [
            pred_roles["x"],
            pred_roles["y"],
            pred_roles.get("z"),
            pred_roles.get("frame"),
        ]:
            if col:
                p[col] = pd.to_numeric(p[col], errors="coerce")
        for col in [
            truth_roles["x"],
            truth_roles["y"],
            truth_roles.get("z"),
            truth_roles.get("frame"),
        ]:
            if col:
                t[col] = pd.to_numeric(t[col], errors="coerce")

        p_required = (
            [pred_roles["x"], pred_roles["y"]]
            + ([pred_roles["z"]] if use_z else [])
            + ([pred_roles["frame"]] if use_frame else [])
        )
        t_required = (
            [truth_roles["x"], truth_roles["y"]]
            + ([truth_roles["z"]] if use_z else [])
            + ([truth_roles["frame"]] if use_frame else [])
        )
        p = p.dropna(subset=[c for c in p_required if c])
        t = t.dropna(subset=[c for c in t_required if c])
        row["n_pred"] = int(len(p))
        row["n_truth"] = int(len(t))

        if len(p) == 0 or len(t) == 0:
            row["status"] = "failed"
            row["message"] = "Prediction or truth table has zero usable rows."
            write_rows_csv([row], output_csv)
            write_rows_csv([], pairs_csv)
            return row

        pairs: List[Dict[str, Any]] = []
        used_truth_indices: set[int] = set()

        if use_frame:
            frames = sorted(
                set(p[pred_roles["frame"]].dropna().astype(int)).union(
                    set(t[truth_roles["frame"]].dropna().astype(int))
                )
            )
        else:
            frames = [None]

        cKDTree = safe_import_scipy_ckdtree()

        for frame in frames:
            if use_frame:
                p_frame = p[p[pred_roles["frame"]].astype(int) == int(frame)]
                t_frame = t[t[truth_roles["frame"]].astype(int) == int(frame)]
            else:
                p_frame = p
                t_frame = t

            if len(p_frame) == 0 or len(t_frame) == 0:
                continue

            p_indices = list(p_frame.index)
            t_indices = list(t_frame.index)
            p_xy = p_frame[[pred_roles["x"], pred_roles["y"]]].to_numpy(dtype=float)
            t_xy = t_frame[[truth_roles["x"], truth_roles["y"]]].to_numpy(dtype=float)

            if cKDTree is not None:
                tree = cKDTree(t_xy)
                distances, local_indices = tree.query(
                    p_xy, k=1, distance_upper_bound=match_radius_xy_nm
                )
                for local_pred_i, (dist, local_truth_i) in enumerate(
                    zip(distances, local_indices)
                ):
                    if math.isinf(float(dist)) or int(local_truth_i) >= len(t_indices):
                        continue
                    pred_idx = int(p_indices[local_pred_i])
                    truth_idx = int(t_indices[int(local_truth_i)])
                    if truth_idx in used_truth_indices:
                        continue
                    if use_z:
                        dz_check = float(p.loc[pred_idx, pred_roles["z"]]) - float(
                            t.loc[truth_idx, truth_roles["z"]]
                        )
                        if abs(dz_check) > match_radius_z_nm:
                            continue
                    else:
                        dz_check = None
                    used_truth_indices.add(truth_idx)
                    dx = float(p.loc[pred_idx, pred_roles["x"]]) - float(
                        t.loc[truth_idx, truth_roles["x"]]
                    )
                    dy = float(p.loc[pred_idx, pred_roles["y"]]) - float(
                        t.loc[truth_idx, truth_roles["y"]]
                    )
                    pairs.append(
                        {
                            "frame": frame,
                            "prediction_index": pred_idx,
                            "truth_index": truth_idx,
                            "dx": dx,
                            "dy": dy,
                            "dz": dz_check,
                            "xy_error": float(dist),
                        }
                    )
            else:
                for pred_idx in p_indices:
                    best_truth_idx = None
                    best_dist = None
                    px = float(p.loc[pred_idx, pred_roles["x"]])
                    py = float(p.loc[pred_idx, pred_roles["y"]])
                    for truth_idx in t_indices:
                        truth_idx_int = int(truth_idx)
                        if truth_idx_int in used_truth_indices:
                            continue
                        tx = float(t.loc[truth_idx, truth_roles["x"]])
                        ty = float(t.loc[truth_idx, truth_roles["y"]])
                        dist = math.sqrt((px - tx) ** 2 + (py - ty) ** 2)
                        if dist <= match_radius_xy_nm and (
                            best_dist is None or dist < best_dist
                        ):
                            if use_z:
                                dz_test = float(
                                    p.loc[pred_idx, pred_roles["z"]]
                                ) - float(t.loc[truth_idx, truth_roles["z"]])
                                if abs(dz_test) > match_radius_z_nm:
                                    continue
                            best_truth_idx = truth_idx_int
                            best_dist = dist
                    if best_truth_idx is not None and best_dist is not None:
                        used_truth_indices.add(best_truth_idx)
                        dx = float(p.loc[pred_idx, pred_roles["x"]]) - float(
                            t.loc[best_truth_idx, truth_roles["x"]]
                        )
                        dy = float(p.loc[pred_idx, pred_roles["y"]]) - float(
                            t.loc[best_truth_idx, truth_roles["y"]]
                        )
                        dz = (
                            float(p.loc[pred_idx, pred_roles["z"]])
                            - float(t.loc[best_truth_idx, truth_roles["z"]])
                            if use_z
                            else None
                        )
                        pairs.append(
                            {
                                "frame": frame,
                                "prediction_index": int(pred_idx),
                                "truth_index": int(best_truth_idx),
                                "dx": dx,
                                "dy": dy,
                                "dz": dz,
                                "xy_error": best_dist,
                            }
                        )

        tp = len(pairs)
        fp = int(len(p)) - tp
        fn = int(len(t)) - tp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else None

        row.update(
            {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
            }
        )

        if pairs:
            dxs = [float(pair["dx"]) for pair in pairs]
            dys = [float(pair["dy"]) for pair in pairs]
            row["rmse_xy"] = math.sqrt(
                sum(dx * dx + dy * dy for dx, dy in zip(dxs, dys)) / len(pairs)
            )
            row["bias_x"] = sum(dxs) / len(dxs)
            row["bias_y"] = sum(dys) / len(dys)
            dzs = [
                safe_float(pair.get("dz"))
                for pair in pairs
                if safe_float(pair.get("dz")) is not None
            ]
            if dzs:
                row["rmse_z"] = math.sqrt(sum(float(dz) ** 2 for dz in dzs) / len(dzs))
                row["bias_z"] = sum(float(dz) for dz in dzs) / len(dzs)

        row["status"] = "passed"
        row["message"] = "Truth benchmark completed."
        write_or_replace_rows_csv([row], output_csv, {"benchmark_layer": "truth", "batch_index": batch_index, "prediction_csv": str(prediction_csv), "truth_csv": str(truth_csv)})
        write_rows_csv(pairs, pairs_csv)
        return row

    except Exception as exc:
        row["status"] = "failed"
        row["message"] = repr(exc)
        write_or_replace_rows_csv([row], output_csv, {"benchmark_layer": "truth", "batch_index": batch_index, "prediction_csv": str(prediction_csv), "truth_csv": str(truth_csv)})
        write_rows_csv([], pairs_csv)
        return row


# =============================================================================
# RuntimeBenchmark main class
# =============================================================================


class RuntimeBenchmark:
    """
    Integrated benchmark manager.

    Backward-compatible use:
        bench = RuntimeBenchmark(out_dir)
        with bench.stage("qc"):
            ...
        bench.finalize()

    Extended use:
        bench.benchmark_input_movie(movie)
        bench.benchmark_localizations(canonical_csv)
        bench.validate_exports({...})
        bench.benchmark_truth(pred_csv, truth_csv)
        bench.finalize()
    """

    def __init__(
        self,
        out_dir: str | Path,
        sample_interval_sec: float = 1.0,
        enable_resource_sampling: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = ensure_figures_dir(self.out_dir)
        self.runtime_csv_path = self.out_dir / "runtime_benchmark.csv"
        self.runtime_json_path = self.out_dir / "runtime_benchmark.json"
        self.summary_json_path = self.out_dir / "benchmark_summary.json"
        self.machine_specs_json_path = self.out_dir / "machine_specs.json"
        self.machine_specs_csv_path = self.out_dir / "machine_specs.csv"
        self.comparison_summary_csv_path = self.out_dir / "comparison_ready_summary.csv"
        self.comparison_summary_json_path = self.out_dir / "comparison_ready_summary.json"
        self.rows: List[Dict[str, Any]] = []
        self.layer_outputs: Dict[str, Any] = {}
        self.psutil = safe_import_psutil()
        self.torch = safe_import_torch()
        self.gpu = GPUMonitor()
        self.machine_specs = collect_machine_specs(self.out_dir)
        self.machine_specs_summary = summarize_machine_specs(self.machine_specs)
        self.write_machine_specs()
        self.sampler = ResourceSampler(
            self.out_dir,
            sample_interval_sec=sample_interval_sec,
            enabled=enable_resource_sampling,
        )
        self.process = None

        if self.psutil is not None:
            try:
                self.process = self.psutil.Process(os.getpid())
                self.psutil.cpu_percent(interval=None)
                self.process.cpu_percent(interval=None)
            except Exception:
                self.process = None

        self.run_metadata = {
            "created_at": now_iso(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "psutil_available": self.psutil is not None,
            "torch_available": self.torch is not None,
            "cuda_available": self.cuda_available(),
            "nvml_available": self.gpu.nvml_available,
            "benchmark_dir": str(self.out_dir),
            "machine_specs": self.machine_specs_summary,
            "machine_specs_json": str(self.machine_specs_json_path),
            "machine_specs_csv": str(self.machine_specs_csv_path),
        }
        self.write_runtime()

    def cuda_available(self) -> bool:
        return self.gpu.cuda_available()

    def cuda_sync(self) -> None:
        self.gpu.sync()

    def reset_cuda_peak(self) -> None:
        self.gpu.reset_torch_peak()

    def get_ram_mb(self) -> Dict[str, Optional[float]]:
        if self.process is None:
            return {"rss_mb": None, "vms_mb": None}
        try:
            mem = self.process.memory_info()
            return {"rss_mb": bytes_to_mb(mem.rss), "vms_mb": bytes_to_mb(mem.vms)}
        except Exception:
            return {"rss_mb": None, "vms_mb": None}

    def get_cpu_percent(self) -> Dict[str, Optional[float]]:
        if self.psutil is None or self.process is None:
            return {"process_cpu_percent": None, "system_cpu_percent": None}
        try:
            return {
                "process_cpu_percent": self.process.cpu_percent(interval=None),
                "system_cpu_percent": self.psutil.cpu_percent(interval=None),
            }
        except Exception:
            return {"process_cpu_percent": None, "system_cpu_percent": None}

    def get_cuda_memory_mb(self) -> Dict[str, Optional[float]]:
        snapshot = self.gpu.torch_snapshot()
        return {
            "gpu_memory_allocated_mb": snapshot.get("torch_cuda_memory_allocated_mb"),
            "gpu_memory_reserved_mb": snapshot.get("torch_cuda_memory_reserved_mb"),
            "gpu_peak_memory_allocated_mb": snapshot.get(
                "torch_cuda_peak_memory_allocated_mb"
            ),
        }

    @contextmanager
    def stage(
        self,
        stage_name: str,
        batch_index: Optional[int] = None,
        input_path: Optional[str | Path] = None,
        out_dir: Optional[str | Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.cuda_sync()
        self.reset_cuda_peak()
        start = time.perf_counter()
        start_time = now_iso()
        self.sampler.start(
            stage=stage_name,
            batch_index=batch_index,
            input_path=input_path,
            out_dir=out_dir,
        )
        status = "passed"
        error = ""

        try:
            yield
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            raise
        finally:
            self.sampler.stop()
            self.cuda_sync()
            end = time.perf_counter()
            end_time = now_iso()
            row: Dict[str, Any] = {
                "benchmark_layer": "runtime",
                "stage": stage_name,
                "batch_index": batch_index,
                "input_path": "" if input_path is None else str(input_path),
                "out_dir": "" if out_dir is None else str(out_dir),
                "status": status,
                "error": error,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_sec": round(end - start, 6),
            }
            row.update(self.get_ram_mb())
            row.update(self.get_cpu_percent())
            row.update(self.get_cuda_memory_mb())
            row.update(self.gpu.nvml_snapshot())
            if extra:
                row.update(extra)
            self.rows.append(row)
            self.write_runtime()

    def benchmark_input_movie(
        self,
        input_path: str | Path,
        batch_index: Optional[int] = None,
        max_sample_frames: int = 200,
        max_pixels_per_frame: int = 50_000,
    ) -> Dict[str, Any]:
        result = benchmark_input_movie(
            Path(input_path).expanduser().resolve(),
            self.out_dir,
            batch_index=batch_index,
            max_sample_frames=max_sample_frames,
            max_pixels_per_frame=max_pixels_per_frame,
        )
        self.layer_outputs.setdefault("input_qc", []).append(result)
        self.write_summary()
        return result

    def benchmark_localizations(
        self,
        canonical_csv: str | Path,
        batch_index: Optional[int] = None,
        coordinate_units: str = "nm",
        pixel_size_nm: Optional[float] = None,
        compute_resolution: bool = True,
        compute_drift: bool = True,
    ) -> Dict[str, Any]:
        canonical_csv_path = Path(canonical_csv).expanduser().resolve()
        loc_result = benchmark_localizations(
            canonical_csv_path,
            self.out_dir,
            batch_index=batch_index,
            coordinate_units=coordinate_units,
            pixel_size_nm=pixel_size_nm,
        )
        self.layer_outputs.setdefault("localization_qc", []).append(loc_result)
        if compute_resolution:
            res = benchmark_resolution_proxy(
                loc_result,
                canonical_csv_path,
                self.out_dir,
                batch_index=batch_index,
                coordinate_units=coordinate_units,
                pixel_size_nm=pixel_size_nm,
            )
            self.layer_outputs.setdefault("resolution", []).append(res)
        if compute_drift:
            drift = benchmark_drift_proxy(
                canonical_csv_path, self.out_dir, batch_index=batch_index
            )
            self.layer_outputs.setdefault("drift", []).append(drift)
        self.write_summary()
        return loc_result

    def validate_exports(
        self, exports: Mapping[str, str | Path | None]
    ) -> Dict[str, Any]:
        result = validate_exports(exports=exports, out_dir=self.out_dir)
        self.layer_outputs["export_validation"] = result
        self.write_summary()
        return result

    def benchmark_truth(
        self,
        prediction_csv: str | Path,
        truth_csv: str | Path,
        batch_index: Optional[int] = None,
        match_radius_xy_nm: float = 50.0,
        match_radius_z_nm: float = 100.0,
    ) -> Dict[str, Any]:
        result = benchmark_truth_matching(
            prediction_csv=Path(prediction_csv).expanduser().resolve(),
            truth_csv=Path(truth_csv).expanduser().resolve(),
            out_dir=self.out_dir,
            batch_index=batch_index,
            match_radius_xy_nm=match_radius_xy_nm,
            match_radius_z_nm=match_radius_z_nm,
        )
        self.layer_outputs.setdefault("truth", []).append(result)
        self.write_summary()
        return result

    def add_quality_metrics_result(
        self,
        step: str,
        payload: Mapping[str, Any],
        batch_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        flags = payload.get("flags", [])
        if not isinstance(flags, list):
            flags = []
        output_paths = payload.get("output_paths", {})
        if not isinstance(output_paths, Mapping):
            output_paths = {}

        severities: Dict[str, int] = {}
        for flag in flags:
            if not isinstance(flag, Mapping):
                continue
            severity = str(flag.get("severity", "info"))
            severities[severity] = severities.get(severity, 0) + 1

        row = {
            "benchmark_layer": "quality_metrics",
            "step": step,
            "batch_index": batch_index,
            "status": payload.get("status", "unknown"),
            "n_flags": len(flags),
            "n_info": severities.get("info", 0),
            "n_warning": severities.get("warning", 0),
            "n_fail": severities.get("fail", 0),
            "n_error": severities.get("error", 0),
            "quality_metrics_json": output_paths.get("quality_metrics_json", ""),
            "quality_metrics_md": output_paths.get("quality_metrics_md", ""),
            "quality_summary_csv": output_paths.get("quality_summary_csv", ""),
            "quality_flags_csv": output_paths.get("quality_flags_csv", ""),
        }
        write_or_replace_rows_csv(
            [row],
            self.out_dir / "quality_metrics_benchmark.csv",
            {
                "benchmark_layer": "quality_metrics",
                "step": step,
                "batch_index": batch_index,
            },
        )
        self.layer_outputs.setdefault("quality_metrics", []).append(row)
        self.write_summary()
        return row

    def write_machine_specs(self) -> None:
        write_json(self.machine_specs, self.machine_specs_json_path)
        write_rows_csv([machine_specs_csv_row(self.machine_specs)], self.machine_specs_csv_path)

    def write_runtime(self) -> None:
        write_json(
            {
                "run_metadata": self.run_metadata,
                "machine_specs": self.machine_specs,
                "stages": self.rows,
            },
            self.runtime_json_path,
        )
        write_rows_csv(self.rows, self.runtime_csv_path)

    def summarize_runtime(self) -> Dict[str, Any]:
        total = 0.0
        by_stage: Dict[str, float] = {}
        for row in self.rows:
            elapsed = safe_float(row.get("elapsed_sec"), 0.0) or 0.0
            stage = str(row.get("stage", "unknown"))
            total += elapsed
            by_stage[stage] = by_stage.get(stage, 0.0) + elapsed

        if by_stage:
            maybe_plot_bar(
                list(by_stage.keys()),
                [float(v) for v in by_stage.values()],
                self.figures_dir / "stage_runtime_barplot.png",
                "Runtime by stage",
                "Seconds",
            )

        return {
            "runtime_csv": str(self.runtime_csv_path),
            "runtime_json": str(self.runtime_json_path),
            "n_timed_stages": len(self.rows),
            "total_timed_sec": round(total, 6),
            "time_by_stage_sec": {
                stage: round(value, 6) for stage, value in sorted(by_stage.items())
            },
        }

    def summarize_layers(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        input_rows = self.layer_outputs.get("input_qc", [])
        if input_rows:
            summary["input_qc"] = {
                "n_inputs_benchmarked": len(input_rows),
                "passed": sum(r.get("status") == "passed" for r in input_rows),
                "failed": sum(r.get("status") == "failed" for r in input_rows),
                "warnings": [
                    r.get("message", "")
                    for r in input_rows
                    if r.get("status") not in {"passed", "skipped"}
                ],
            }

        loc_rows = self.layer_outputs.get("localization_qc", [])
        if loc_rows:
            total_locs = sum(
                safe_int(r.get("n_localizations"), 0) or 0 for r in loc_rows
            )
            med_photons = [
                safe_float(r.get("median_photons"))
                for r in loc_rows
                if safe_float(r.get("median_photons")) is not None
            ]
            med_bg = [
                safe_float(r.get("median_background"))
                for r in loc_rows
                if safe_float(r.get("median_background")) is not None
            ]
            med_lpx = [
                safe_float(r.get("median_lpx"))
                for r in loc_rows
                if safe_float(r.get("median_lpx")) is not None
            ]
            summary["localization_qc"] = {
                "n_batches_benchmarked": len(loc_rows),
                "total_localizations": total_locs,
                "passed": sum(r.get("status") == "passed" for r in loc_rows),
                "failed": sum(r.get("status") == "failed" for r in loc_rows),
                "median_of_median_photons": numeric_percentile(med_photons, 50),
                "median_of_median_background": numeric_percentile(med_bg, 50),
                "median_of_median_lpx": numeric_percentile(med_lpx, 50),
            }

        resolution_results = self.layer_outputs.get("resolution", [])
        if resolution_results:
            metrics: List[Mapping[str, Any]] = []
            for item in resolution_results:
                if isinstance(item, Mapping):
                    item_metrics = item.get("metrics", [])
                    if isinstance(item_metrics, list):
                        metrics.extend(
                            metric for metric in item_metrics if isinstance(metric, Mapping)
                        )
            frc_values = [
                safe_float(metric.get("value"))
                for metric in metrics
                if metric.get("metric") == "frc_resolution_1_7"
                and safe_float(metric.get("value")) is not None
            ]
            sampling_values = [
                safe_float(metric.get("value"))
                for metric in metrics
                if metric.get("metric") == "sampling_limited_resolution_proxy"
                and safe_float(metric.get("value")) is not None
            ]
            precision_values = [
                safe_float(metric.get("value"))
                for metric in metrics
                if metric.get("metric") == "median_xy_localization_precision"
                and safe_float(metric.get("value")) is not None
            ]
            summary["resolution"] = {
                "n_resolution_benchmarks": len(resolution_results),
                "n_metrics": len(metrics),
                "n_frc_available": len(frc_values),
                "median_frc_resolution_nm": numeric_percentile(frc_values, 50),
                "median_sampling_limited_resolution_nm": numeric_percentile(
                    sampling_values, 50
                ),
                "median_xy_localization_precision": numeric_percentile(
                    precision_values, 50
                ),
            }

        drift_rows = self.layer_outputs.get("drift", [])
        if drift_rows:
            max_drift_values = [
                safe_float(row.get("max_radial_drift"))
                for row in drift_rows
                if isinstance(row, Mapping)
                and safe_float(row.get("max_radial_drift")) is not None
            ]
            p95_drift_values = [
                safe_float(row.get("p95_radial_drift"))
                for row in drift_rows
                if isinstance(row, Mapping)
                and safe_float(row.get("p95_radial_drift")) is not None
            ]
            summary["drift"] = {
                "n_drift_benchmarks": len(drift_rows),
                "passed": sum(r.get("status") == "passed" for r in drift_rows),
                "failed": sum(r.get("status") == "failed" for r in drift_rows),
                "median_max_radial_drift": numeric_percentile(max_drift_values, 50),
                "median_p95_radial_drift": numeric_percentile(p95_drift_values, 50),
            }

        if "export_validation" in self.layer_outputs:
            export_result = self.layer_outputs["export_validation"]
            exports = export_result.get("exports", [])
            summary["export_validation"] = {
                "status": export_result.get("status"),
                "passed": sum(r.get("status") == "passed" for r in exports),
                "failed": sum(r.get("status") == "failed" for r in exports),
                "skipped": sum(r.get("status") == "skipped" for r in exports),
                "csv": export_result.get("export_validation_csv", ""),
            }

        quality_rows = self.layer_outputs.get("quality_metrics", [])
        if quality_rows:
            summary["quality_metrics"] = {
                "n_quality_runs": len(quality_rows),
                "passed": sum(r.get("status") == "passed" for r in quality_rows),
                "warning": sum(r.get("status") == "warning" for r in quality_rows),
                "fail": sum(r.get("status") == "fail" for r in quality_rows),
                "error": sum(r.get("status") == "error" for r in quality_rows),
                "total_flags": sum(safe_int(r.get("n_flags"), 0) or 0 for r in quality_rows),
                "total_warning_flags": sum(
                    safe_int(r.get("n_warning"), 0) or 0 for r in quality_rows
                ),
                "total_fail_flags": sum(
                    safe_int(r.get("n_fail"), 0) or 0 for r in quality_rows
                ),
                "total_error_flags": sum(
                    safe_int(r.get("n_error"), 0) or 0 for r in quality_rows
                ),
            }

        truth_rows = self.layer_outputs.get("truth", [])
        if truth_rows:
            summary["truth"] = {
                "n_truth_benchmarks": len(truth_rows),
                "passed": sum(r.get("status") == "passed" for r in truth_rows),
                "failed": sum(r.get("status") == "failed" for r in truth_rows),
                "median_jaccard": numeric_percentile(
                    [
                        safe_float(r.get("jaccard"))
                        for r in truth_rows
                        if safe_float(r.get("jaccard")) is not None
                    ],
                    50,
                ),
                "median_f1": numeric_percentile(
                    [
                        safe_float(r.get("f1"))
                        for r in truth_rows
                        if safe_float(r.get("f1")) is not None
                    ],
                    50,
                ),
            }

        return summary

    def summarize(self) -> Dict[str, Any]:
        runtime = self.summarize_runtime()
        resources = self.sampler.summarize()
        layers = self.summarize_layers()
        status = "passed"
        warnings: List[str] = []

        for row in self.rows:
            if row.get("status") == "failed":
                status = "failed"
                warnings.append(f"Runtime stage failed: {row.get('stage')}")

        for layer_name, layer_data in layers.items():
            if isinstance(layer_data, dict) and layer_data.get("failed", 0):
                if status != "failed":
                    status = "warning"
                warnings.append(
                    f"{layer_name}: {layer_data.get('failed')} failed item(s)"
                )

        return {
            "status": status,
            "created_at": now_iso(),
            "benchmark_dir": str(self.out_dir),
            "run_metadata": self.run_metadata,
            "machine_specs": self.machine_specs,
            "machine_specs_summary": self.machine_specs_summary,
            "runtime": runtime,
            "resources": resources,
            "layers": layers,
            "files": {
                "runtime_csv": str(self.runtime_csv_path),
                "runtime_json": str(self.runtime_json_path),
                "resource_csv": str(self.sampler.csv_path),
                "resource_json": str(self.sampler.json_path),
                "machine_specs_json": str(self.machine_specs_json_path),
                "machine_specs_csv": str(self.machine_specs_csv_path),
                "comparison_ready_summary_csv": str(self.comparison_summary_csv_path),
                "comparison_ready_summary_json": str(self.comparison_summary_json_path),
                "quality_metrics_benchmark_csv": str(
                    self.out_dir / "quality_metrics_benchmark.csv"
                ),
                "input_qc_csv": str(self.out_dir / "input_qc_benchmark.csv"),
                "localization_qc_csv": str(
                    self.out_dir / "localization_qc_benchmark.csv"
                ),
                "resolution_csv": str(self.out_dir / "resolution_benchmark.csv"),
                "drift_csv": str(self.out_dir / "drift_benchmark.csv"),
                "truth_csv": str(self.out_dir / "truth_benchmark.csv"),
                "export_validation_csv": str(self.out_dir / "export_validation.csv"),
                "figures_dir": str(self.figures_dir),
                "benchmark_summary_json": str(self.summary_json_path),
            },
            "warnings": warnings,
        }

    def comparison_ready_row(self, summary: Mapping[str, Any]) -> Dict[str, Any]:
        layers = summary.get("layers", {})
        if not isinstance(layers, Mapping):
            layers = {}
        runtime = summary.get("runtime", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        resources = summary.get("resources", {})
        if not isinstance(resources, Mapping):
            resources = {}
        machine = summary.get("machine_specs_summary", {})
        if not isinstance(machine, Mapping):
            machine = {}

        input_qc = layers.get("input_qc", {})
        loc_qc = layers.get("localization_qc", {})
        resolution = layers.get("resolution", {})
        drift = layers.get("drift", {})
        truth = layers.get("truth", {})
        quality = layers.get("quality_metrics", {})
        export_validation = layers.get("export_validation", {})
        if not isinstance(input_qc, Mapping):
            input_qc = {}
        if not isinstance(loc_qc, Mapping):
            loc_qc = {}
        if not isinstance(resolution, Mapping):
            resolution = {}
        if not isinstance(drift, Mapping):
            drift = {}
        if not isinstance(truth, Mapping):
            truth = {}
        if not isinstance(quality, Mapping):
            quality = {}
        if not isinstance(export_validation, Mapping):
            export_validation = {}

        return {
            "created_at": summary.get("created_at", ""),
            "benchmark_status": summary.get("status", ""),
            "benchmark_dir": summary.get("benchmark_dir", ""),
            "hostname": machine.get("hostname", ""),
            "os": machine.get("os", ""),
            "cpu_model": machine.get("cpu_model", ""),
            "cpu_physical_cores": machine.get("cpu_physical_cores"),
            "cpu_logical_cores": machine.get("cpu_logical_cores"),
            "ram_total_gb": machine.get("ram_total_gb"),
            "gpu_count": machine.get("gpu_count"),
            "gpu_names": "; ".join(machine.get("gpu_names", []) or []),
            "gpu_total_memory_gb": "; ".join(
                str(v) for v in (machine.get("gpu_total_memory_gb", []) or [])
            ),
            "nvidia_driver_version": machine.get("nvidia_driver_version", ""),
            "cuda_runtime_version": machine.get("cuda_runtime_version", ""),
            "cudnn_version": machine.get("cudnn_version", ""),
            "torch_version": machine.get("torch_version", ""),
            "total_timed_sec": runtime.get("total_timed_sec"),
            "n_timed_stages": runtime.get("n_timed_stages"),
            "max_rss_mb": resources.get("max_rss_mb"),
            "max_gpu_total_mem_used_mb": resources.get("max_gpu_total_mem_used_mb"),
            "max_gpu_any_util_percent": resources.get("max_gpu_any_util_percent"),
            "n_inputs_benchmarked": input_qc.get("n_inputs_benchmarked"),
            "input_qc_failed": input_qc.get("failed"),
            "n_localization_batches": loc_qc.get("n_batches_benchmarked"),
            "total_localizations": loc_qc.get("total_localizations"),
            "median_photons": loc_qc.get("median_of_median_photons"),
            "median_background": loc_qc.get("median_of_median_background"),
            "median_lpx": loc_qc.get("median_of_median_lpx"),
            "median_xy_localization_precision": resolution.get(
                "median_xy_localization_precision"
            ),
            "median_frc_resolution_nm": resolution.get("median_frc_resolution_nm"),
            "n_frc_available": resolution.get("n_frc_available"),
            "median_sampling_limited_resolution_nm": resolution.get(
                "median_sampling_limited_resolution_nm"
            ),
            "median_max_radial_drift": drift.get("median_max_radial_drift"),
            "median_p95_radial_drift": drift.get("median_p95_radial_drift"),
            "truth_median_jaccard": truth.get("median_jaccard"),
            "truth_median_f1": truth.get("median_f1"),
            "quality_status_fail_count": quality.get("fail"),
            "quality_status_error_count": quality.get("error"),
            "quality_total_flags": quality.get("total_flags"),
            "quality_warning_flags": quality.get("total_warning_flags"),
            "quality_fail_flags": quality.get("total_fail_flags"),
            "quality_error_flags": quality.get("total_error_flags"),
            "export_validation_status": export_validation.get("status"),
        }

    def write_comparison_ready_summary(self, summary: Mapping[str, Any]) -> None:
        row = self.comparison_ready_row(summary)
        write_rows_csv([row], self.comparison_summary_csv_path)
        write_json(row, self.comparison_summary_json_path)

    def write_summary(self) -> Dict[str, Any]:
        summary = self.summarize()
        write_json(summary, self.summary_json_path)
        self.write_comparison_ready_summary(summary)
        return summary

    def finalize(self) -> Dict[str, Any]:
        self.write_machine_specs()
        self.write_runtime()
        self.sampler.write()
        return self.write_summary()


# =============================================================================
# CLI inspection / standalone use
# =============================================================================


def inspect_runtime_json(path: Path) -> None:
    data = read_json(path)
    stages = data.get("stages", [])
    print(f"Benchmark file: {path}")
    print(f"Timed stages: {len(stages)}")
    print()
    for row in stages:
        print(
            f"{row.get('stage', ''):28s} "
            f"batch={str(row.get('batch_index', '')):>4s} "
            f"time={row.get('elapsed_sec', '')} sec "
            f"status={row.get('status', '')}"
        )


def parse_export_args(values: Optional[Sequence[str]]) -> Dict[str, str]:
    exports: Dict[str, str] = {}
    if not values:
        return exports
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --export value. Expected name=path, got: {value}"
            )
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                f"Invalid --export value. Expected name=path, got: {value}"
            )
        exports[name] = path
    return exports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprehensive SMLM runtime/scientific benchmark helper."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to runtime_benchmark.json to inspect. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--bench-dir",
        default=None,
        help="Benchmark output folder for standalone benchmark generation.",
    )
    parser.add_argument(
        "--movie",
        default=None,
        help="Optional TIFF/OME-TIFF movie for input QC benchmark.",
    )
    parser.add_argument(
        "--canonical-csv",
        default=None,
        help="Optional canonical localization CSV for localization/resolution/drift benchmark.",
    )
    parser.add_argument(
        "--truth-csv",
        default=None,
        help="Optional ground-truth CSV for prediction-vs-truth benchmark.",
    )
    parser.add_argument(
        "--export",
        action="append",
        default=None,
        help="Optional export validation entry as name=path. Repeatable.",
    )
    parser.add_argument(
        "--coord-units",
        choices=["nm", "pixel", "auto"],
        default="nm",
        help="Coordinate units for localization QC.",
    )
    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        default=None,
        help="Pixel size in nm if coordinates are in pixels.",
    )
    parser.add_argument(
        "--match-radius-xy-nm",
        type=float,
        default=50.0,
        help="XY matching radius for truth benchmark.",
    )
    parser.add_argument(
        "--match-radius-z-nm",
        type=float,
        default=100.0,
        help="Z matching radius for truth benchmark.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.input and not any(
        [args.bench_dir, args.movie, args.canonical_csv, args.truth_csv, args.export]
    ):
        inspect_runtime_json(Path(args.input).expanduser().resolve())
        return

    bench_dir = Path(args.bench_dir or "benchmarks").expanduser().resolve()
    bench = RuntimeBenchmark(out_dir=bench_dir)

    if args.movie:
        bench.benchmark_input_movie(Path(args.movie).expanduser().resolve())

    if args.canonical_csv:
        bench.benchmark_localizations(
            canonical_csv=Path(args.canonical_csv).expanduser().resolve(),
            coordinate_units=args.coord_units,
            pixel_size_nm=args.pixel_size_nm,
        )

    if args.canonical_csv and args.truth_csv:
        bench.benchmark_truth(
            prediction_csv=Path(args.canonical_csv).expanduser().resolve(),
            truth_csv=Path(args.truth_csv).expanduser().resolve(),
            match_radius_xy_nm=args.match_radius_xy_nm,
            match_radius_z_nm=args.match_radius_z_nm,
        )

    exports = parse_export_args(args.export)
    if exports:
        bench.validate_exports(exports)

    summary = bench.finalize()
    print("Benchmark complete.")
    print(f"Benchmark dir: {bench_dir}")
    print(f"Summary JSON:  {summary['files']['benchmark_summary_json']}")
    print(f"Status:        {summary['status']}")


if __name__ == "__main__":
    main()
