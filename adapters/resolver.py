#!/usr/bin/env python3
"""
adapters/resolver.py

Runtime resolver for SMLM LabFlow.

Purpose:
    - read machine-specific LiteLoc paths from adapters/backend_paths.yml
      or config/local_paths.yaml
    - expose resolve_backend_runtime(...) for run_pipeline.py
    - return a plain dictionary, not dataclasses
    - keep CLI clean: no --model-path, no --calibration-file, no --backend-paths

Expected by run_pipeline.py:
    resolve_backend_runtime(
        step=...,
        profile=...,
        backend_name=...,
        backend_paths_file=...,
        project_root=...,
        results_dir=...,
        registry_dir=...,
        ...
    ) -> dict
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


class ResolverError(RuntimeError):
    """Raised when backend runtime resolution fails."""


# =============================================================================
# Small utilities
# =============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_nested(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)

    return default if current is None else current


def read_yaml(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.exists():
        raise ResolverError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ResolverError(f"YAML file must contain a dictionary: {path}")

    return data


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def as_path(value: Any, label: str, base_dir: Optional[Path] = None) -> Path:
    if value is None or str(value).strip() == "":
        raise ResolverError(f"Missing required path: {label}")

    path = Path(str(value)).expanduser()

    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    return path.resolve()


def optional_path(value: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None

    path = Path(str(value)).expanduser()

    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    return path.resolve()


def require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ResolverError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ResolverError(f"{label} is not a file: {path}")


def require_existing_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise ResolverError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ResolverError(f"{label} is not a directory: {path}")


# =============================================================================
# Config discovery
# =============================================================================


def find_backend_paths_file(
    backend_paths_file: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
) -> Path:
    """
    Search order:
        1. explicit backend_paths_file from run_pipeline.py
        2. LITELOC_BACKEND_PATHS env var
        3. adapters/backend_paths.yml
        4. config/local_paths.yaml
        5. ~/.liteloc_wrapper/local_paths.yaml
    """
    if backend_paths_file is not None:
        path = Path(backend_paths_file).expanduser().resolve()
        if path.exists():
            return path

    env_path = os.environ.get("LITELOC_BACKEND_PATHS") or os.environ.get(
        "LITELOC_WRAPPER_LOCAL_CONFIG"
    )
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        raise ResolverError(f"Backend paths env var points to missing file: {path}")

    root = Path(project_root or Path.cwd()).expanduser().resolve()

    candidates = [
        root / "adapters" / "backend_paths.yml",
        root / "config" / "local_paths.yaml",
        Path.home() / ".liteloc_wrapper" / "local_paths.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise ResolverError(f"Could not find backend paths file. Searched:\n{searched}")


# =============================================================================
# Registry helpers
# =============================================================================


def global_registry_dir(registry_dir: Optional[str | Path]) -> Optional[Path]:
    """
    Current run registry:
        results/3/registry

    Global sibling registry:
        results/registry

    This allows:
        calibrate run -> writes latest_calibration.json
        train run     -> resolver can find latest_calibration.json
        infer run     -> resolver can find latest_model.json
    """
    if registry_dir is None:
        return None

    local = Path(registry_dir).expanduser().resolve()

    if local.name == "registry":
        return local.parent.parent / "registry"

    return local / "registry"


def latest_artifact_from_registry(
    registry_dir: Optional[str | Path],
    artifact_name: str,
) -> Dict[str, Any]:
    gdir = global_registry_dir(registry_dir)

    if gdir is None:
        return {}

    return read_json_if_exists(gdir / artifact_name)


# =============================================================================
# Main resolver
# =============================================================================


def resolve_backend_runtime(
    step: str,
    profile: Dict[str, Any],
    backend_name: str = "liteloc",
    backend_paths_file: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    run_parent: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    benchmarks_dir: Optional[str | Path] = None,
    reports_dir: Optional[str | Path] = None,
    registry_dir: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """
    Resolve one backend runtime.

    Returns a dictionary because run_pipeline.py expects dict, not dataclass.
    """
    step = str(step).lower().strip()
    backend_name = str(backend_name).lower().strip()
    cli_overrides = cli_overrides or {}

    if backend_name != "liteloc":
        return {
            "status": "unsupported_backend",
            "backend_name": backend_name,
            "step": step,
            "message": "Only liteloc is currently supported by this resolver.",
        }

    paths_file = find_backend_paths_file(
        backend_paths_file=backend_paths_file,
        project_root=project_root,
    )

    config = read_yaml(paths_file)

    liteloc = config.get("liteloc")
    if not isinstance(liteloc, dict):
        raise ResolverError(f"Missing required section liteloc in: {paths_file}")

    root = as_path(liteloc.get("root"), "liteloc.root")
    require_existing_dir(root, "LiteLoc root")

    ports = liteloc.get("ports")
    if not isinstance(ports, dict):
        raise ResolverError(f"Missing required section liteloc.ports in: {paths_file}")

    calibrate_script = as_path(
        ports.get("calibrate"),
        "liteloc.ports.calibrate",
        base_dir=root,
    )
    train_script = as_path(
        ports.get("train"),
        "liteloc.ports.train",
        base_dir=root,
    )
    infer_script = as_path(
        ports.get("infer"),
        "liteloc.ports.infer",
        base_dir=root,
    )

    require_existing_file(calibrate_script, "LiteLoc calibration port")
    require_existing_file(train_script, "LiteLoc training port")
    require_existing_file(infer_script, "LiteLoc inference port")

    # Scientific identity comes from profile.
    psf_type = (
        cli_overrides.get("psf_type")
        or get_nested(profile, "experiment", "psf_type")
        or get_nested(profile, "psf", "type")
        or ""
    )

    psf_dimensionality = (
        cli_overrides.get("psf_dimensionality")
        or get_nested(profile, "experiment", "dimensionality")
        or get_nested(profile, "psf", "dimensionality")
        or ""
    )

    pixel_size_nm = (
        cli_overrides.get("pixel_size_nm")
        or get_nested(profile, "microscope", "pixel_size_nm")
        or get_nested(profile, "camera", "pixel_size_nm")
        or get_nested(profile, "smlm", "pixel_size_nm")
    )

    batch_size = (
        cli_overrides.get("batch_size")
        or get_nested(profile, "liteloc", "runtime", "batch_size")
        or get_nested(profile, "inference", "batch_size")
    )

    threshold = cli_overrides.get("threshold") or get_nested(
        profile, "inference", "threshold"
    )

    device = (
        cli_overrides.get("device")
        or get_nested(profile, "inference", "device")
        or "auto"
    )

    # Calibration/model are automatic through registry/profile.
    latest_calib = latest_artifact_from_registry(
        registry_dir, "latest_calibration.json"
    )
    latest_model = latest_artifact_from_registry(registry_dir, "latest_model.json")

    calibration_file = (
        cli_overrides.get("calibration_file")
        or get_nested(profile, "calibration", "file")
        or get_nested(profile, "psf", "calibration_file")
        or latest_calib.get("calibration_file")
        or ""
    )

    model_path = (
        cli_overrides.get("model_path")
        or get_nested(profile, "liteloc", "model_path")
        or get_nested(profile, "inference", "model_path")
        or latest_model.get("model_path")
        or ""
    )

    base_infer_yaml = (
        cli_overrides.get("base_infer_yaml")
        or get_nested(profile, "liteloc", "base_infer_yaml")
        or get_nested(profile, "liteloc", "infer_yaml")
        or get_nested(profile, "inference", "base_yaml")
        or "demo/demo1_astig_npc/infer_params_demo1.yaml"
    )

    base_infer_yaml_path = optional_path(base_infer_yaml, base_dir=root)

    resolved: Dict[str, Any] = {
        "status": "passed",
        "resolver_function": "adapters.resolver.resolve_backend_runtime",
        "resolved_at": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "backend_name": backend_name,
        "backend_paths_file": str(paths_file),
        "project_root": str(Path(project_root or Path.cwd()).resolve()),
        "run_parent": "" if run_parent is None else str(run_parent),
        "out_dir": "" if out_dir is None else str(out_dir),
        "results_dir": "" if results_dir is None else str(results_dir),
        "benchmarks_dir": "" if benchmarks_dir is None else str(benchmarks_dir),
        "reports_dir": "" if reports_dir is None else str(reports_dir),
        "registry_dir": "" if registry_dir is None else str(registry_dir),
        # LiteLoc root and ports
        "root": str(root),
        "liteloc_root": str(root),
        "ports": {
            "calibrate": str(calibrate_script),
            "train": str(train_script),
            "infer": str(infer_script),
        },
        "calibrate_script": str(calibrate_script),
        "calibration_script": str(calibrate_script),
        "train_script": str(train_script),
        "training_script": str(train_script),
        "infer_script": str(infer_script),
        "inference_script": str(infer_script),
        # Scientific/runtime identity
        "psf_type": psf_type,
        "psf_dimensionality": psf_dimensionality,
        "pixel_size_nm": pixel_size_nm,
        "device": device,
        "batch_size": batch_size,
        "threshold": threshold,
        # Artifacts
        "calibration_file": str(calibration_file) if calibration_file else "",
        "model_path": str(model_path) if model_path else "",
        "base_infer_yaml": str(base_infer_yaml_path) if base_infer_yaml_path else "",
        # Keep useful provenance
        "latest_calibration_registry": latest_calib,
        "latest_model_registry": latest_model,
        "cli_overrides": cli_overrides,
    }

    # Step-specific sanity.
    # Calibrate does not require existing calibration/model.
    if step == "train" and not resolved["calibration_file"]:
        resolved["status"] = "warning"
        resolved["message"] = (
            "No calibration_file resolved. Training may fail unless the training script has its own calibration config."
        )

    if step == "infer" and not resolved["model_path"]:
        resolved["status"] = "warning"
        resolved["message"] = (
            "No model_path resolved. Inference may fail unless the profile or LiteLoc script provides a model."
        )

    return resolved


# Supported aliases for run_pipeline.py
resolve_backend_config = resolve_backend_runtime
resolve_backend = resolve_backend_runtime
resolve_liteloc_runtime = resolve_backend_runtime
resolve_liteloc_paths = resolve_backend_runtime
resolve_paths = resolve_backend_runtime


# =============================================================================
# Doctor mode
# =============================================================================


def main() -> None:
    resolved = resolve_backend_runtime(
        step="doctor",
        profile={},
        backend_name="liteloc",
        project_root=Path.cwd(),
    )

    print("LiteLoc resolver OK")
    print(f"  root:      {resolved['liteloc_root']}")
    print(f"  calibrate: {resolved['calibrate_script']}")
    print(f"  train:     {resolved['train_script']}")
    print(f"  infer:     {resolved['infer_script']}")


if __name__ == "__main__":
    main()
