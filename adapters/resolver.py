#!/usr/bin/env python3
"""
Backend runtime resolver for SMLM LabFlow.

This module has one job: merge machine-specific backend wiring, scientific
profile settings, and registry artifacts into the backend_config dictionary
passed to adapters such as adapters/liteloc_adapter.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_BACKEND = "liteloc"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_backend_paths_file() -> Path:
    return project_root() / "adapters" / "backend_paths.yml"


def get_nested(data: Mapping[str, Any], keys: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to parse adapters/backend_paths.yml."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary: {path}")
    return data


def is_auto_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {
        "auto",
        "labflow:auto",
        "__auto__",
    }


def first_non_auto(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if is_auto_value(value):
            continue
        return value
    return None


def resolve_path(value: Any, base_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def profile_base_dir(profile: Mapping[str, Any], fallback: Path) -> Path:
    profile_path = profile.get("profile_path")
    if profile_path:
        try:
            return Path(str(profile_path)).expanduser().resolve().parent
        except Exception:
            pass
    return fallback


def global_registry_dir(
    run_parent: Optional[Path],
    registry_dir: Optional[Path],
    project_root_path: Path,
) -> Path:
    if run_parent is not None:
        return run_parent.expanduser().resolve().parent / "registry"
    if registry_dir is not None:
        return registry_dir.expanduser().resolve().parent.parent / "registry"
    return project_root_path / "outputs" / "registry"


def latest_artifact_path(
    registry_dir: Path,
    filename: str,
    artifact_key: str,
    warnings: List[str],
) -> str:
    artifact_json = registry_dir / filename
    artifact = load_json_if_exists(artifact_json, {})
    if not isinstance(artifact, Mapping):
        return ""

    status = str(artifact.get("status", "")).lower().strip()
    if status and status != "passed":
        warnings.append(
            f"Ignored {artifact_json} because its status is {artifact.get('status')!r}."
        )
        return ""

    value = artifact.get(artifact_key) or get_nested(
        artifact,
        ["step_result", artifact_key],
        None,
    )
    return str(value) if value else ""


def infer_pixel_size_nm(profile: Mapping[str, Any]) -> Optional[float]:
    for keys in (
        ["pixel_size_nm"],
        ["data", "pixel_size_nm"],
        ["input", "pixel_size_nm"],
        ["camera", "pixel_size_nm"],
        ["acquisition", "pixel_size_nm"],
        ["microscope", "pixel_size_nm"],
        ["smlm", "pixel_size_nm"],
    ):
        value = get_nested(profile, list(keys), None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def infer_coord_units(profile: Mapping[str, Any]) -> str:
    value = (
        get_nested(profile, ["inference", "coord_units"], None)
        or get_nested(profile, ["canonical", "coordinate_unit"], None)
        or get_nested(profile, ["outputs", "coord_units"], None)
        or "auto"
    )
    value = str(value).lower().strip()
    return value if value in {"auto", "nm", "pixel"} else "auto"


def profile_cli_overrides(
    profile: Mapping[str, Any],
    extra_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    values = {
        "psf_type": get_nested(profile, ["experiment", "psf_type"], None)
        or get_nested(profile, ["psf", "type"], None),
        "psf_dimensionality": get_nested(
            profile,
            ["experiment", "dimensionality"],
            None,
        )
        or get_nested(profile, ["psf", "dimensionality"], None),
        "calibration_mode": get_nested(profile, ["calibration", "mode"], None)
        or get_nested(profile, ["psf", "calibration_mode"], None),
        "calibration_file": get_nested(profile, ["calibration", "file"], None)
        or get_nested(profile, ["psf", "calibration_file"], None)
        or get_nested(profile, ["liteloc", "calibration_file"], None),
        "z_step_nm": get_nested(profile, ["calibration", "z_step_nm"], None)
        or get_nested(profile, ["psf", "z_step_nm"], None),
        "device": get_nested(profile, ["training", "device"], None)
        or get_nested(profile, ["inference", "device"], None),
        "batch_size": get_nested(profile, ["inference", "batch_size"], None)
        or get_nested(profile, ["liteloc", "runtime", "batch_size"], None),
        "threshold": get_nested(profile, ["inference", "threshold"], None),
        "time_block_gb": get_nested(profile, ["inference", "time_block_gb"], None)
        or get_nested(profile, ["liteloc", "runtime", "time_block_gb"], None),
        "sub_fov_size": get_nested(profile, ["inference", "sub_fov_size"], None)
        or get_nested(profile, ["liteloc", "runtime", "sub_fov_size"], None),
        "over_cut": get_nested(profile, ["inference", "over_cut"], None)
        or get_nested(profile, ["liteloc", "runtime", "over_cut"], None),
        "data_queue_size": get_nested(profile, ["inference", "data_queue_size"], None)
        or get_nested(profile, ["liteloc", "runtime", "data_queue_size"], None),
        "multi_gpu": get_nested(profile, ["inference", "multi_gpu"], None)
        or get_nested(profile, ["liteloc", "runtime", "multi_gpu"], None),
        "num_producers": get_nested(profile, ["inference", "num_producers"], None)
        or get_nested(profile, ["liteloc", "runtime", "num_producers"], None),
        "end_frame_num": get_nested(profile, ["inference", "end_frame_num"], None)
        or get_nested(profile, ["liteloc", "runtime", "end_frame_num"], None),
        "model_path": get_nested(profile, ["inference", "model_path"], None)
        or get_nested(profile, ["liteloc", "model_path"], None),
        "coord_units": infer_coord_units(profile),
        "pixel_size_nm": infer_pixel_size_nm(profile),
    }

    if extra_overrides:
        values.update(dict(extra_overrides))

    return {
        key: value
        for key, value in values.items()
        if value is not None and not is_auto_value(value)
    }


def resolve_backend_runtime(
    *,
    step: Optional[str] = None,
    command: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    backend_name: str = DEFAULT_BACKEND,
    backend_paths_file: Optional[str | Path] = None,
    backend_paths_path: Optional[str | Path] = None,
    cli_overrides: Optional[Mapping[str, Any]] = None,
    project_root: Optional[str | Path] = None,
    run_parent: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    benchmarks_dir: Optional[str | Path] = None,
    reports_dir: Optional[str | Path] = None,
    registry_dir: Optional[str | Path] = None,
    **_: Any,
) -> Dict[str, Any]:
    profile = dict(profile or {})
    step_name = str(step or command or "").strip().lower()
    backend_name = str(backend_name or DEFAULT_BACKEND).strip().lower()

    project_root_path = (
        Path(project_root).expanduser().resolve()
        if project_root
        else globals()["project_root"]()
    )
    backend_paths = Path(
        backend_paths_file or backend_paths_path or default_backend_paths_file()
    ).expanduser().resolve()

    warnings: List[str] = []
    backend_paths_data: Dict[str, Any] = {}
    if backend_paths.exists():
        backend_paths_data = read_yaml(backend_paths)
    else:
        warnings.append(f"Backend paths file does not exist yet: {backend_paths}")

    backend_block = backend_paths_data.get(backend_name, {}) or {}
    if not isinstance(backend_block, Mapping):
        raise ValueError(f"Backend block {backend_name!r} must be a YAML mapping.")

    profile_dir = profile_base_dir(profile, project_root_path)
    overrides = profile_cli_overrides(profile)
    overrides.update(dict(cli_overrides or {}))
    overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None and not is_auto_value(value)
    }

    run_parent_path = Path(run_parent).expanduser().resolve() if run_parent else None
    registry_path = Path(registry_dir).expanduser().resolve() if registry_dir else None
    shared_registry = global_registry_dir(
        run_parent=run_parent_path,
        registry_dir=registry_path,
        project_root_path=project_root_path,
    )

    root_value = first_non_auto(
        get_nested(profile, ["backend", "root"], None),
        get_nested(profile, ["backend", "liteloc_root"], None),
        get_nested(profile, ["liteloc", "root"], None),
        get_nested(profile, ["liteloc", "repo_dir"], None),
        backend_block.get("root"),
        backend_block.get("liteloc_root"),
    )

    explicit_calibration = first_non_auto(
        overrides.get("calibration_file"),
        get_nested(profile, ["calibration", "file"], None),
        get_nested(profile, ["psf", "calibration_file"], None),
        get_nested(profile, ["liteloc", "calibration_file"], None),
    )
    explicit_model = first_non_auto(
        overrides.get("model_path"),
        overrides.get("checkpoint_path"),
        get_nested(profile, ["inference", "model_path"], None),
        get_nested(profile, ["liteloc", "model_path"], None),
    )

    calibration_file = ""
    if explicit_calibration:
        calibration_file = resolve_path(explicit_calibration, profile_dir)
    elif step_name in {"train", "infer"}:
        calibration_file = latest_artifact_path(
            shared_registry,
            "latest_calibration.json",
            "calibration_file",
            warnings,
        )

    model_path = ""
    if explicit_model:
        model_path = resolve_path(explicit_model, profile_dir)
    elif step_name == "infer":
        model_path = latest_artifact_path(
            shared_registry,
            "latest_model.json",
            "model_path",
            warnings,
        )

    root = resolve_path(root_value, project_root_path) if root_value else ""
    result: Dict[str, Any] = {
        "status": "passed" if backend_block else "backend_paths_missing",
        "step": step_name,
        "backend_name": backend_name,
        "backend_paths_file": str(backend_paths),
        "backend_paths": backend_paths_data,
        "project_root": str(project_root_path),
        "run_parent": str(run_parent_path) if run_parent_path else "",
        "results_dir": str(results_dir or ""),
        "benchmarks_dir": str(benchmarks_dir or ""),
        "reports_dir": str(reports_dir or ""),
        "registry_dir": str(registry_path) if registry_path else "",
        "global_registry_dir": str(shared_registry),
        "resolver_function": "adapters.resolver.resolve_backend_runtime",
        "resolved_at": now_iso(),
        "cli_overrides": overrides,
        "root": root,
        "liteloc_root": root,
        "modules": dict(backend_block.get("modules", {}) or {}),
        "functions": dict(backend_block.get("functions", {}) or {}),
        "execution": dict(backend_block.get("execution", {}) or {}),
        "supported_calibration_modes": list(
            backend_block.get("supported_calibration_modes", []) or []
        ),
    }

    for key, value in backend_block.items():
        result.setdefault(key, value)
    for key, value in overrides.items():
        result[key] = value
    if calibration_file:
        result["calibration_file"] = calibration_file
    if model_path:
        result["model_path"] = model_path
    if warnings:
        result["warnings"] = warnings

    return result


resolve_backend_config = resolve_backend_runtime
resolve_backend = resolve_backend_runtime
resolve_liteloc_runtime = resolve_backend_runtime
resolve_liteloc_paths = resolve_backend_runtime
resolve_paths = resolve_backend_runtime
