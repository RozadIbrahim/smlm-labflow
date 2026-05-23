#!/usr/bin/env python3
"""
adapters/liteloc_adapter.py

LiteLoc backend adapter for SMLM LabFlow.

This file is intentionally limited to backend execution only:

    calibrate
    train
    infer

It does NOT run general quality metrics, generate reports, combine outputs, or
open napari/Locan. Those belong to quality_metrics.py, generate_run_report.py,
combine_run_outputs.py, post_inference.py, and napari_locan_review.py.

Expected architecture:

    run_pipeline.py
        -> adapters.resolver.resolve_backend_runtime(...)
        -> adapters.liteloc_adapter.run_liteloc_calibration / training / inference

Resolver/backend config should provide, when available:

    {
      "root": "/path/to/LiteLoc",
      "modules": {
        "vector_calibration": ".../utils/vectorpsf_fit.py",
        "spline_calibration_io": ".../spline_psf/calibration_io.py",
        "train": ".../network/loc_model.py",
        "infer": ".../network/multi_process.py"
      },
      "functions": {
        "vector_calibration": "beads_psf_calibrate",
        "spline_loader_class": "SMAPSplineCoefficient",
        "train_class": "LocModel",
        "infer_class": "CompetitiveSmlmDataAnalyzer_multi_producer"
      },
      "execution": {
        "vector_calibration": "function",
        "spline_calibration_io": "module",
        "train": "module",
        "infer": "module"
      },
      "calibration_mode": "vector_beads|spline_file|none|analytic",
      "calibration_file": "/optional/path/to/calibration.mat",
      "model_path": "/optional/path/to/checkpoint.pkl",
      "base_train_yaml": "/path/to/train_params.yaml",
      "base_infer_yaml": "/path/to/infer_params.yaml"
    }

Backward compatibility:
    The adapter still tolerates older backend_config dictionaries with:
        ports.train / ports.infer / ports.calibrate
        train_script / infer_script / calibrate_script
        liteloc_root / root

But the final design should use modules/functions/execution, not demo script ports.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("PyYAML is required by adapters/liteloc_adapter.py") from exc


# =============================================================================
# Generic utilities
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


def set_nested(data: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        existing = current.get(key)
        if not isinstance(existing, dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def deep_update(base: Dict[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def is_auto_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {
        "auto",
        "labflow:auto",
        "__auto__",
    }


def copied_mapping(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping/dictionary.")
    return copy.deepcopy(dict(value))


def profile_liteloc_runtime_yaml(profile: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    """
    Return an inline LiteLoc runtime YAML template from the LabFlow profile.

    Supported profile forms:
        liteloc.runtime_yaml.train
        liteloc.train_config
        training.runtime_yaml

    The same pattern is supported for calibration and infer.
    """
    candidates = [
        get_nested(profile, "liteloc", "runtime_yaml", stage),
        get_nested(profile, "liteloc", f"{stage}_config"),
    ]

    if stage == "calibration":
        candidates.append(get_nested(profile, "calibration", "runtime_yaml"))
    elif stage == "train":
        candidates.append(get_nested(profile, "training", "runtime_yaml"))
    elif stage == "infer":
        candidates.append(get_nested(profile, "inference", "runtime_yaml"))

    for candidate in candidates:
        if candidate:
            return copied_mapping(candidate, f"LiteLoc {stage} runtime YAML")
    return {}


def load_liteloc_runtime_yaml(
    *,
    stage: str,
    profile: Mapping[str, Any],
    base_yaml_value: Any,
    base_dir: Path,
    required_sections: Sequence[str],
) -> Tuple[Dict[str, Any], str]:
    """
    Load a LiteLoc YAML from either a base file or inline profile config.

    Base YAMLs remain supported for compatibility, but deployable LabFlow
    profiles can now contain the full LiteLoc YAML sections directly.
    """
    if base_yaml_value and not is_auto_value(base_yaml_value):
        base_yaml_path = resolve_path(base_yaml_value, base_dir=base_dir)
        return read_yaml(base_yaml_path), str(base_yaml_path)

    config = profile_liteloc_runtime_yaml(profile, stage)
    if not config:
        sections = ", ".join(required_sections)
        raise KeyError(
            f"Missing LiteLoc {stage} runtime YAML. Add liteloc.runtime_yaml.{stage} "
            f"with sections [{sections}], or provide a base_{stage}_yaml."
        )

    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError(
            f"LiteLoc {stage} runtime YAML is missing required section(s): {missing}"
        )

    return config, f"profile:liteloc.runtime_yaml.{stage}"


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary: {path}")
    return data


def resolve_path(value: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (base_dir / path).resolve()
    return path.resolve()


def optional_path(value: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_path(text, base_dir=base_dir)


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_mode(value: Any, default: str = "auto") -> str:
    if value is None or str(value).strip() == "":
        value = default
    return str(value).strip().lower().replace("-", "_")


def absolute_path(path: Path) -> Path:
    """Return an absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def snapshot_files(folder: Path) -> set[Path]:
    if not folder.exists():
        return set()
    return {p.resolve() for p in folder.rglob("*") if p.is_file()}


def add_liteloc_to_syspath_and_env(repo_dir: Path) -> Dict[str, str]:
    repo_dir = repo_dir.expanduser().resolve()
    repo_str = str(repo_dir)

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_str if not existing else repo_str + os.pathsep + existing
    return env


def path_to_module_name(path: Path, repo_dir: Path) -> str:
    """
    Convert /repo/network/multi_process.py -> network.multi_process.
    """
    path = path.expanduser().resolve()
    repo_dir = repo_dir.expanduser().resolve()

    try:
        relative = path.relative_to(repo_dir)
    except ValueError as exc:
        raise ValueError(f"Module path {path} is not inside LiteLoc root {repo_dir}") from exc

    if relative.suffix != ".py":
        raise ValueError(f"Expected a Python module file, got: {path}")

    return ".".join(relative.with_suffix("").parts)


def import_from_module_file(
    module_file: Path,
    repo_dir: Path,
    attribute_name: str,
) -> Any:
    """
    Import an attribute from a LiteLoc Python file using package-style imports.

    Example:
        module_file = /LiteLoc/network/multi_process.py
        attribute_name = CompetitiveSmlmDataAnalyzer_multi_producer
    """
    add_liteloc_to_syspath_and_env(repo_dir)
    module_name = path_to_module_name(module_file, repo_dir)
    module = importlib.import_module(module_name)
    obj = getattr(module, attribute_name, None)
    if obj is None:
        raise AttributeError(f"{attribute_name!r} not found in {module_name}")
    return obj


def get_attr_or_item(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def ensure_trailing_slash(path: Path) -> str:
    text = str(path.expanduser().resolve())
    return text if text.endswith(os.sep) else text + os.sep


def file_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


# =============================================================================
# Runtime config resolution inside adapter
# =============================================================================


@dataclass
class LiteLocRuntime:
    repo_dir: Path
    modules: Dict[str, Path]
    functions: Dict[str, str]
    execution: Dict[str, str]
    backend_config: Dict[str, Any]


def _candidate_root_values(profile: Mapping[str, Any], backend_config: Mapping[str, Any]) -> List[Any]:
    return [
        backend_config.get("liteloc_root"),
        backend_config.get("root"),
        get_nested(backend_config, "paths", "root"),
        get_nested(backend_config, "backend_paths", "liteloc", "root"),
        get_nested(backend_config, "backend_paths", "root"),
        get_nested(profile, "liteloc", "root"),
        get_nested(profile, "liteloc", "repo_dir"),
        get_nested(profile, "backend", "root"),
        get_nested(profile, "backend", "liteloc_root"),
    ]


def _first_existing(values: Iterable[Any], base_dir: Optional[Path], label: str, required: bool = False) -> Optional[Path]:
    attempted: List[str] = []
    for value in values:
        path = optional_path(value, base_dir=base_dir)
        if path is None:
            continue
        attempted.append(str(path))
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"Could not resolve {label}. Tried: {attempted if attempted else 'nothing'}")
    return None


def _get_module_candidate(
    backend_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    logical_name: str,
    legacy_names: Sequence[str],
) -> List[Any]:
    values: List[Any] = [
        get_nested(backend_config, "modules", logical_name),
        get_nested(backend_config, "backend_paths", "liteloc", "modules", logical_name),
        get_nested(profile, "liteloc", "modules", logical_name),
    ]

    for name in legacy_names:
        values.extend(
            [
                backend_config.get(name),
                backend_config.get(f"{name}_port"),
                get_nested(backend_config, "ports", name),
                get_nested(backend_config, "backend_paths", "liteloc", "ports", name),
                get_nested(profile, "liteloc", name),
                get_nested(profile, "liteloc", "ports", name),
                get_nested(profile, "backend", "ports", name),
            ]
        )

    return values


def resolve_liteloc_runtime(
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
) -> LiteLocRuntime:
    """
    Resolve LiteLoc root, backend modules, function names, and execution modes.

    Preferred input comes from adapters.resolver.py as backend_config.
    This function is deliberately tolerant because the project evolved from a
    legacy ports-based config to the final modules/functions/execution design.
    """
    backend_config = dict(backend_config or {})

    root = _first_existing(
        _candidate_root_values(profile, backend_config),
        base_dir=None,
        label="LiteLoc root",
        required=False,
    )

    if root is None:
        raise FileNotFoundError(
            "LiteLoc root is missing. The resolver must provide backend_config['root'] "
            "or backend_config['liteloc_root'], or the profile must contain liteloc.root."
        )

    repo_dir = root.resolve()

    modules: Dict[str, Path] = {}

    module_specs = {
        "vector_calibration": ("vector_calibration", ["calibrate", "calibration", "calibrate_script", "calibration_script"]),
        "spline_calibration_io": ("spline_calibration_io", ["spline_calibration_io", "spline_loader"]),
        "train": ("train", ["train", "training", "train_script", "training_script"]),
        "infer": ("infer", ["infer", "inference", "infer_script", "inference_script"]),
    }

    for logical_name, (_, legacy_names) in module_specs.items():
        candidate = _first_existing(
            _get_module_candidate(backend_config, profile, logical_name, legacy_names),
            base_dir=repo_dir,
            label=f"LiteLoc module {logical_name}",
            required=False,
        )
        if candidate is not None:
            modules[logical_name] = candidate.resolve()

    functions = {
        "vector_calibration": (
            get_nested(backend_config, "functions", "vector_calibration")
            or get_nested(profile, "liteloc", "functions", "vector_calibration")
            or "beads_psf_calibrate"
        ),
        "spline_loader_class": (
            get_nested(backend_config, "functions", "spline_loader_class")
            or get_nested(profile, "liteloc", "functions", "spline_loader_class")
            or "SMAPSplineCoefficient"
        ),
        "train_class": (
            get_nested(backend_config, "functions", "train_class")
            or get_nested(profile, "liteloc", "functions", "train_class")
            or "LocModel"
        ),
        "infer_class": (
            get_nested(backend_config, "functions", "infer_class")
            or get_nested(profile, "liteloc", "functions", "infer_class")
            or "CompetitiveSmlmDataAnalyzer_multi_producer"
        ),
    }

    execution = {
        "vector_calibration": normalize_mode(
            get_nested(backend_config, "execution", "vector_calibration")
            or get_nested(profile, "liteloc", "execution", "vector_calibration")
            or "function"
        ),
        "spline_calibration_io": normalize_mode(
            get_nested(backend_config, "execution", "spline_calibration_io")
            or get_nested(profile, "liteloc", "execution", "spline_calibration_io")
            or "module"
        ),
        "train": normalize_mode(
            get_nested(backend_config, "execution", "train")
            or get_nested(profile, "liteloc", "execution", "train")
            or "module"
        ),
        "infer": normalize_mode(
            get_nested(backend_config, "execution", "infer")
            or get_nested(profile, "liteloc", "execution", "infer")
            or get_nested(profile, "liteloc", "infer_execution")
            or "module"
        ),
    }

    resolved = dict(backend_config)
    resolved.setdefault("status", "resolved_by_liteloc_adapter")
    resolved["root"] = str(repo_dir)
    resolved["liteloc_root"] = str(repo_dir)
    resolved["modules"] = {k: str(v) for k, v in modules.items()}
    resolved["functions"] = dict(functions)
    resolved["execution"] = dict(execution)

    return LiteLocRuntime(
        repo_dir=repo_dir,
        modules=modules,
        functions=functions,
        execution=execution,
        backend_config=resolved,
    )


# =============================================================================
# Artifact detection
# =============================================================================


def artifact_fingerprint(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except Exception:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


def artifact_name_matches(
    path: Path,
    suffixes: set[str],
    name_tokens: Sequence[str] = (),
) -> bool:
    if path.suffix.lower() not in suffixes:
        return False
    if name_tokens and not any(token in path.name.lower() for token in name_tokens):
        return False
    return True


def snapshot_artifacts(
    folder: Path,
    *,
    suffixes: set[str],
    name_tokens: Sequence[str] = (),
) -> Dict[Path, Tuple[int, int]]:
    if not folder.exists():
        return {}

    snapshot: Dict[Path, Tuple[int, int]] = {}
    for path in folder.rglob("*"):
        if not path.is_file() or not artifact_name_matches(path, suffixes, name_tokens):
            continue
        fingerprint = artifact_fingerprint(path)
        if fingerprint is not None:
            snapshot[path.resolve()] = fingerprint
    return snapshot


def find_new_artifact(
    search_dirs: Sequence[Path],
    suffixes: set[str],
    before_files: Optional[set[Path]] = None,
    before_artifacts: Optional[Mapping[Path, Tuple[int, int]]] = None,
    name_tokens: Sequence[str] = (),
) -> Optional[Path]:
    candidates: List[Tuple[Path, Tuple[int, int]]] = []

    for folder in search_dirs:
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if not artifact_name_matches(path, suffixes, name_tokens):
                continue

            resolved = path.resolve()
            fingerprint = artifact_fingerprint(path)
            if fingerprint is None:
                continue

            if before_artifacts is not None:
                if before_artifacts.get(resolved) == fingerprint:
                    continue
            elif before_files is not None and resolved in before_files:
                continue

            candidates.append((resolved, fingerprint))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item[1][0], reverse=True)[0][0]


def copy_artifact_to_out_dir(
    artifact: Optional[Path],
    out_dir: Path,
    preferred_name: Optional[str] = None,
) -> str:
    if artifact is None:
        return ""

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (preferred_name or artifact.name)

    if artifact.resolve() != target.resolve():
        shutil.copy2(artifact, target)

    return str(target.resolve())


# =============================================================================
# Optional script runner, kept only for compatibility
# =============================================================================


def run_python_script(
    *,
    script_path: Path,
    repo_dir: Path,
    log_path: Path,
    env: Dict[str, str],
    extra_args: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess[str]:
    if not script_path.exists():
        raise FileNotFoundError(f"LiteLoc script not found: {script_path}")

    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write("Command:\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()

        return subprocess.run(
            command,
            cwd=str(repo_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            shell=False,
        )


# =============================================================================
# Calibration helpers
# =============================================================================


def infer_calibration_mode(input_path: Path, profile: Dict[str, Any], backend_config: Dict[str, Any]) -> str:
    explicit = (
        backend_config.get("calibration_mode")
        or get_nested(profile, "calibration", "mode")
        or get_nested(profile, "psf", "calibration_mode")
    )

    mode = normalize_mode(explicit, default="auto")
    if mode != "auto":
        return mode

    lower_name = input_path.name.lower()
    suffix = input_path.suffix.lower()

    if lower_name.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        return "vector_beads"

    if suffix in {".mat", ".h5", ".hdf5"}:
        return "spline_file"

    raise ValueError(
        "Could not infer calibration mode. Set calibration.mode in the profile "
        "to vector_beads, spline_file, none, or analytic."
    )


def calibration_input_for_vector_beads(input_path: Path, out_dir: Path, profile: Dict[str, Any]) -> Path:
    """
    LiteLoc beads_psf_calibrate writes the .mat next to beads_file_name.

    To avoid writing beside the original lab dataset, stage the input inside the
    run output folder and point LiteLoc to that staged path. Prefer a symlink,
    then a hardlink, then an optional copy. The returned path intentionally does
    not resolve symlinks because LiteLoc derives its output .mat path from this
    string.
    """
    use_symlink = as_bool(get_nested(profile, "calibration", "use_input_symlink", default=True), default=True)

    if not use_symlink:
        return input_path

    out_dir.mkdir(parents=True, exist_ok=True)
    link_path = out_dir / input_path.name
    staged_path = absolute_path(link_path)

    if link_path.exists() or link_path.is_symlink():
        return staged_path

    try:
        link_path.symlink_to(input_path)
        return staged_path
    except Exception:
        pass

    try:
        os.link(input_path, link_path)
        return staged_path
    except Exception:
        pass

    copy_on_failure = as_bool(
        get_nested(profile, "calibration", "copy_input_on_symlink_failure", default=True),
        default=True,
    )
    if copy_on_failure:
        shutil.copy2(input_path, link_path)
        return staged_path

    return input_path


def build_runtime_calibration_yaml(
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    repo_dir: Path,
    backend_config: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    base_yaml_value = (
        backend_config.get("base_calibration_yaml")
        or backend_config.get("calibration_base_yaml")
        or get_nested(profile, "calibration", "base_yaml")
        or get_nested(profile, "liteloc", "base_calibration_yaml")
    )

    config, config_source = load_liteloc_runtime_yaml(
        stage="calibration",
        profile=profile,
        base_yaml_value=base_yaml_value,
        base_dir=repo_dir,
        required_sections=(
            "psf_params_dict",
            "camera_params_dict",
            "calib_params_dict",
            "beads_file_name",
        ),
    )

    beads_input = calibration_input_for_vector_beads(input_path, out_dir, profile)

    # LiteLoc vectorpsf_fit.beads_psf_calibrate expects params_dict['beads_file_name'].
    config["beads_file_name"] = str(beads_input)

    # Useful metadata for your wrapper. LiteLoc ignores unknown top-level keys if it
    # only reads known nested fields from the loaded dict.
    config.setdefault("labflow", {})
    config["labflow"].update(
        {
            "created_at": now_iso(),
            "source_input_path": str(input_path),
            "runtime_beads_file_name": str(beads_input),
            "out_dir": str(out_dir),
            "config_source": config_source,
            "calibration_mode": "vector_beads",
        }
    )

    # Gentle optional patches. Do not guess too much; profile can override exactly.
    z_step_nm = (
        backend_config.get("z_step_nm")
        or get_nested(profile, "calibration", "z_step_nm")
        or get_nested(profile, "psf", "z_step_nm")
    )
    pixel_size_nm = (
        backend_config.get("pixel_size_nm")
        or get_nested(profile, "microscope", "pixel_size_nm")
        or get_nested(profile, "camera", "pixel_size_nm")
        or get_nested(profile, "smlm", "pixel_size_nm")
    )

    if z_step_nm is not None:
        config.setdefault("calib_params_dict", {})
        config["calib_params_dict"].setdefault("z_step", z_step_nm)
        config["calib_params_dict"].setdefault("z_step_nm", z_step_nm)
        config["labflow"]["z_step_nm"] = z_step_nm

    if pixel_size_nm is not None:
        config.setdefault("psf_params_dict", {})
        config["psf_params_dict"].setdefault(
            "pixel_size_xy",
            [pixel_size_nm, pixel_size_nm],
        )
        config["labflow"]["pixel_size_nm"] = pixel_size_nm

    patches = (
        backend_config.get("calibration_yaml_patches")
        or get_nested(profile, "calibration", "yaml_patches", default={})
    )
    if patches:
        if not isinstance(patches, Mapping):
            raise ValueError("calibration.yaml_patches must be a mapping/dictionary.")
        deep_update(config, patches)

    runtime_yaml_path = out_dir / "runtime_liteloc_calibration.yaml"
    write_yaml(config, runtime_yaml_path)

    return runtime_yaml_path, config


def run_vector_bead_calibration(
    *,
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
    runtime: LiteLocRuntime,
    log_path: Path,
    status_path: Path,
) -> Dict[str, Any]:
    start = time.time()

    module_path = runtime.modules.get("vector_calibration")
    if module_path is None:
        raise FileNotFoundError(
            "Missing vector_calibration module. Set liteloc.modules.vector_calibration "
            "in adapters/backend_paths.yml."
        )

    function_name = runtime.functions.get("vector_calibration", "beads_psf_calibrate")
    calibrate_fn = import_from_module_file(module_path, runtime.repo_dir, function_name)

    runtime_yaml_path, calib_config = build_runtime_calibration_yaml(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        repo_dir=runtime.repo_dir,
        backend_config=backend_config,
    )

    env = add_liteloc_to_syspath_and_env(runtime.repo_dir)
    env["LITELOC_CALIBRATION_INPUT"] = str(input_path)
    env["LITELOC_CALIBRATION_OUTPUT"] = str(out_dir)
    env["LITELOC_RUN_OUTPUT"] = str(out_dir)

    beads_file = absolute_path(Path(str(calib_config["beads_file_name"])))
    search_dirs = [
        out_dir,
        beads_file.parent,
        input_path.parent,
        runtime.repo_dir / "calibrate_mat",
        runtime.repo_dir / "results",
    ]

    artifact_suffixes = {".mat"}
    artifact_name_tokens = ("calib", "calibration", "psf", "spline")

    before_artifacts: Dict[Path, Tuple[int, int]] = {}
    for folder in search_dirs:
        before_artifacts.update(
            snapshot_artifacts(
                folder,
                suffixes=artifact_suffixes,
                name_tokens=artifact_name_tokens,
            )
        )

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc vector-bead calibration adapter reached.\n")
        log.write(f"Started at: {now_iso()}\n")
        log.write(f"LiteLoc root: {runtime.repo_dir}\n")
        log.write(f"Vector calibration module: {module_path}\n")
        log.write(f"Vector calibration function: {function_name}\n")
        log.write(f"Input path: {input_path}\n")
        log.write(f"Runtime YAML: {runtime_yaml_path}\n")
        log.write(f"Runtime beads_file_name: {beads_file}\n")
        if beads_file.is_symlink():
            log.write(f"Runtime beads_file_target: {beads_file.resolve()}\n")
        log.write(f"Output dir: {out_dir}\n")
        log.write(f"PYTHONPATH: {env.get('PYTHONPATH', '')}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                calibrate_fn(calib_config)
            return_code = 0
            error = ""
        except Exception as exc:
            return_code = 1
            error = repr(exc)

    artifact = find_new_artifact(
        search_dirs=search_dirs,
        suffixes=artifact_suffixes,
        before_artifacts=before_artifacts,
        name_tokens=artifact_name_tokens,
    )
    copied_artifact = copy_artifact_to_out_dir(artifact, out_dir)

    status = "passed" if return_code == 0 and copied_artifact else "failed"

    result = {
        "backend_name": "liteloc",
        "backend_status": status,
        "calibrate_status": status,
        "calibration_mode": "vector_beads",
        "elapsed_seconds": round(time.time() - start, 3),
        "returncode": return_code,
        "error": error,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "liteloc_root": str(runtime.repo_dir),
        "vector_calibration_module": str(module_path),
        "vector_calibration_function": function_name,
        "runtime_calibration_yaml": str(runtime_yaml_path),
        "runtime_beads_file_name": str(beads_file),
        "runtime_beads_file_target": str(beads_file.resolve()) if beads_file.is_symlink() else "",
        "calibration_file": copied_artifact,
        "raw_detected_calibration_file": str(artifact) if artifact else "",
        "artifact_detection": "new_or_modified_mat",
        "log_path": str(log_path),
        "status_json": str(status_path),
        "message": (
            "Vector bead calibration completed and a calibration .mat artifact was detected."
            if status == "passed"
            else "Vector bead calibration failed or no calibration .mat artifact was detected. Check log."
        ),
    }
    write_json(result, status_path)

    if error:
        raise RuntimeError(f"LiteLoc vector bead calibration failed. Check log: {log_path}")

    return result


def register_spline_calibration(
    *,
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
    runtime: LiteLocRuntime,
    log_path: Path,
    status_path: Path,
) -> Dict[str, Any]:
    start = time.time()

    calibration_value = (
        backend_config.get("calibration_file")
        or get_nested(profile, "calibration", "file")
        or get_nested(profile, "psf", "calibration_file")
        or str(input_path)
    )
    calibration_path = resolve_path(calibration_value, base_dir=runtime.repo_dir)

    if not calibration_path.exists():
        raise FileNotFoundError(f"Spline calibration file not found: {calibration_path}")

    if calibration_path.suffix.lower() not in {".mat", ".h5", ".hdf5"}:
        raise ValueError(
            f"spline_file calibration expects .mat/.h5/.hdf5, got: {calibration_path}"
        )

    module_path = runtime.modules.get("spline_calibration_io")
    if module_path is None:
        raise FileNotFoundError(
            "Missing spline_calibration_io module. Set liteloc.modules.spline_calibration_io "
            "in adapters/backend_paths.yml."
        )

    class_name = runtime.functions.get("spline_loader_class", "SMAPSplineCoefficient")
    loader_cls = import_from_module_file(module_path, runtime.repo_dir, class_name)

    validation_status = "not_run"
    validation_error = ""
    validation_object_type = ""

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc spline calibration registration reached.\n")
        log.write(f"Started at: {now_iso()}\n")
        log.write(f"LiteLoc root: {runtime.repo_dir}\n")
        log.write(f"Spline loader module: {module_path}\n")
        log.write(f"Spline loader class: {class_name}\n")
        log.write(f"Calibration file: {calibration_path}\n")
        log.write(f"Output dir: {out_dir}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                validation_object = loader_cls(str(calibration_path))
            validation_status = "passed"
            validation_object_type = type(validation_object).__name__
        except Exception as exc:
            validation_status = "failed"
            validation_error = repr(exc)

    copied_file = copy_artifact_to_out_dir(calibration_path, out_dir)

    status = "passed" if validation_status == "passed" and copied_file else "failed"

    result = {
        "backend_name": "liteloc",
        "backend_status": status,
        "calibrate_status": status,
        "calibration_mode": "spline_file",
        "elapsed_seconds": round(time.time() - start, 3),
        "returncode": 0 if status == "passed" else 1,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "liteloc_root": str(runtime.repo_dir),
        "spline_calibration_io_module": str(module_path),
        "spline_loader_class": class_name,
        "validation_status": validation_status,
        "validation_error": validation_error,
        "validation_object_type": validation_object_type,
        "calibration_file": copied_file,
        "raw_detected_calibration_file": str(calibration_path),
        "calibration_file_size_bytes": file_size_bytes(calibration_path),
        "log_path": str(log_path),
        "status_json": str(status_path),
        "message": (
            "Existing spline calibration file was validated and registered."
            if status == "passed"
            else "Spline calibration file could not be validated. Check log."
        ),
    }
    write_json(result, status_path)

    if validation_status == "failed":
        raise RuntimeError(
            f"Spline calibration validation failed for {calibration_path}. Check log: {log_path}"
        )

    return result


def register_no_external_calibration(
    *,
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
    runtime: LiteLocRuntime,
    log_path: Path,
    status_path: Path,
    mode: str,
) -> Dict[str, Any]:
    result = {
        "backend_name": "liteloc",
        "backend_status": "passed",
        "calibrate_status": "passed",
        "calibration_mode": mode,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "liteloc_root": str(runtime.repo_dir),
        "calibration_file": "",
        "raw_detected_calibration_file": "",
        "log_path": str(log_path),
        "status_json": str(status_path),
        "message": (
            "No external calibration file is required for this profile. "
            "Training should rely on analytic/profile-defined PSF configuration."
        ),
    }

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc no-external-calibration registration reached.\n")
        log.write(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        log.write("\n")

    write_json(result, status_path)
    return result


# =============================================================================
# Training helpers
# =============================================================================


def build_runtime_train_yaml(
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    repo_dir: Path,
    backend_config: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    base_yaml_value = (
        backend_config.get("base_train_yaml")
        or backend_config.get("train_yaml")
        or get_nested(profile, "liteloc", "base_train_yaml")
        or get_nested(profile, "training", "base_yaml")
    )

    config, config_source = load_liteloc_runtime_yaml(
        stage="train",
        profile=profile,
        base_yaml_value=base_yaml_value,
        base_dir=repo_dir,
        required_sections=("Camera", "PSF_model", "Training"),
    )

    # LiteLoc LocModel.save_model() commonly writes to params.Training.result_path.
    config.setdefault("Training", {})
    config["Training"]["result_path"] = ensure_trailing_slash(out_dir)
    if is_auto_value(config["Training"].get("infer_data")):
        config["Training"]["infer_data"] = str(input_path)

    # Store wrapper provenance.
    config.setdefault("LabFlow", {})
    config["LabFlow"].update(
        {
            "created_at": now_iso(),
            "source_input_path": str(input_path),
            "out_dir": str(out_dir),
            "config_source": config_source,
            "calibration_mode": (
                backend_config.get("calibration_mode")
                or get_nested(profile, "calibration", "mode")
                or ""
            ),
        }
    )

    calibration_file = (
        backend_config.get("calibration_file")
        or get_nested(profile, "calibration", "file")
        or get_nested(profile, "psf", "calibration_file")
        or ""
    )

    if calibration_file and not is_auto_value(calibration_file):
        calibration_path = resolve_path(calibration_file, base_dir=repo_dir)
        config["LabFlow"]["calibration_file"] = str(calibration_path)

        # Common safe patch locations. Profile-specific train_yaml_patches can override.
        config.setdefault("PSF_model", {})
        simulate_method = normalize_mode(
            get_nested(config, "PSF_model", "simulate_method"),
            default="",
        )
        calibration_mode = normalize_mode(
            backend_config.get("calibration_mode")
            or get_nested(profile, "calibration", "mode"),
            default="",
        )

        if not simulate_method:
            simulate_method = "spline" if calibration_mode == "spline_file" else "vector"
            config["PSF_model"]["simulate_method"] = simulate_method

        if simulate_method == "spline":
            config["PSF_model"].setdefault("spline_psf", {})
            config["PSF_model"]["spline_psf"]["calibration_file"] = str(calibration_path)
        elif simulate_method in {"vector", "uipsf"}:
            vector_key = "ui_psf" if simulate_method == "uipsf" else "vector_psf"
            config["PSF_model"].setdefault(vector_key, {})
            config["PSF_model"][vector_key]["zernikefit_file"] = str(calibration_path)
        else:
            config["PSF_model"]["calibration_file"] = str(calibration_path)

        for keys in (
            ("PSF_model", "spline_psf", "calibration_file"),
            ("PSF_model", "vector_psf", "zernikefit_file"),
            ("PSF_model", "ui_psf", "zernikefit_file"),
            ("PSF_model", "calibration_file"),
        ):
            if is_auto_value(get_nested(config, *keys)):
                set_nested(config, keys, str(calibration_path))
    else:
        auto_calibration_fields = [
            ("PSF_model", "spline_psf", "calibration_file"),
            ("PSF_model", "vector_psf", "zernikefit_file"),
            ("PSF_model", "ui_psf", "zernikefit_file"),
            ("PSF_model", "calibration_file"),
        ]
        unresolved = [
            ".".join(keys)
            for keys in auto_calibration_fields
            if is_auto_value(get_nested(config, *keys))
        ]
        if unresolved:
            raise KeyError(
                "LiteLoc training profile still contains auto calibration path(s): "
                f"{unresolved}. Run `calibrate` first, or set calibration.file / "
                "psf.calibration_file to a real calibration artifact."
            )

    patches = (
        backend_config.get("train_yaml_patches")
        or get_nested(profile, "liteloc", "train_yaml_patches", default={})
        or get_nested(profile, "training", "yaml_patches", default={})
    )
    if patches:
        if not isinstance(patches, Mapping):
            raise ValueError("train_yaml_patches / training.yaml_patches must be a mapping.")
        deep_update(config, patches)

    runtime_yaml_path = out_dir / "runtime_liteloc_train.yaml"
    write_yaml(config, runtime_yaml_path)

    return runtime_yaml_path, config


def find_checkpoint(out_dir: Path, before_files: Optional[set[Path]] = None) -> Optional[Path]:
    before_files = before_files or set()
    candidates: List[Path] = []

    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() in before_files:
            continue
        if path.suffix.lower() not in {".pkl", ".pickle", ".pt", ".pth", ".ckpt"}:
            continue
        if any(token in path.name.lower() for token in ("checkpoint", "model", "best", "epoch", "liteloc")):
            candidates.append(path.resolve())

    if not candidates:
        checkpoint = out_dir / "checkpoint.pkl"
        if checkpoint.exists():
            return checkpoint.resolve()
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# =============================================================================
# Inference helpers
# =============================================================================


def detect_infer_execution(runtime: LiteLocRuntime, profile: Dict[str, Any]) -> str:
    value = (
        get_nested(runtime.backend_config, "execution", "infer")
        or get_nested(profile, "liteloc", "infer_execution")
        or "module"
    )
    mode = normalize_mode(value, default="module")
    if mode not in {"module", "script"}:
        raise ValueError("Invalid inference execution mode. Expected module or script.")
    return mode


def build_runtime_liteloc_infer_yaml(
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    repo_dir: Path,
    backend_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    backend_config = dict(backend_config or {})
    out_dir.mkdir(parents=True, exist_ok=True)

    base_yaml_value = (
        backend_config.get("base_infer_yaml")
        or backend_config.get("infer_yaml")
        or get_nested(profile, "liteloc", "base_infer_yaml")
        or get_nested(profile, "liteloc", "infer_yaml")
        or get_nested(profile, "inference", "base_yaml")
    )

    config, config_source = load_liteloc_runtime_yaml(
        stage="infer",
        profile=profile,
        base_yaml_value=base_yaml_value,
        base_dir=repo_dir,
        required_sections=("Loc_Model", "Multi_Process"),
    )

    model_path_value = (
        backend_config.get("model_path")
        or backend_config.get("checkpoint_path")
        or get_nested(profile, "liteloc", "model_path")
        or get_nested(profile, "inference", "model_path")
        or get_nested(config, "Loc_Model", "model_path")
    )

    if not model_path_value or is_auto_value(model_path_value):
        raise KeyError(
            "Missing LiteLoc model checkpoint. Provide it through registry/latest_model.json, "
            "profile.inference.model_path, or profile.liteloc.model_path."
        )

    model_path = resolve_path(model_path_value, base_dir=repo_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"LiteLoc model checkpoint not found: {model_path}")

    raw_output_name = get_nested(
        profile,
        "output",
        "raw_output_name",
        default="liteloc_raw_output.csv",
    )
    runtime_yaml_name = get_nested(
        profile,
        "output",
        "runtime_infer_yaml_name",
        default=get_nested(
            profile,
            "output",
            "runtime_yaml_name",
            default="runtime_liteloc_infer.yaml",
        ),
    )

    raw_output_path = out_dir / str(raw_output_name)
    runtime_yaml_path = out_dir / str(runtime_yaml_name)

    config.setdefault("Loc_Model", {})
    config.setdefault("Multi_Process", {})

    config["Loc_Model"]["model_path"] = str(model_path)
    config["Multi_Process"]["image_path"] = str(input_path)
    config["Multi_Process"]["save_path"] = str(raw_output_path)

    runtime_sources = [
        ("time_block_gb", 1),
        ("batch_size", 64),
        ("sub_fov_size", 256),
        ("over_cut", 8),
        ("data_queue_size", 100),
        ("multi_gpu", False),
        ("end_frame_num", None),
        ("num_producers", 1),
    ]

    for key, default in runtime_sources:
        value = (
            backend_config.get(key)
            or get_nested(profile, "liteloc", "runtime", key)
            or get_nested(profile, "inference", key)
            or default
        )
        if value is not None:
            config["Multi_Process"][key] = value

    patches = (
        backend_config.get("infer_yaml_patches")
        or get_nested(profile, "liteloc", "infer_yaml_patches", default={})
        or get_nested(profile, "inference", "yaml_patches", default={})
    )
    if patches:
        if not isinstance(patches, Mapping):
            raise ValueError("infer_yaml_patches / inference.yaml_patches must be a mapping.")
        deep_update(config, patches)

    write_yaml(config, runtime_yaml_path)

    resolved_config = {
        "config_source": config_source,
        "model_path": str(model_path),
        "runtime_yaml_path": str(runtime_yaml_path),
        "raw_output_path": str(raw_output_path),
        "multi_process": config.get("Multi_Process", {}),
    }

    return runtime_yaml_path, raw_output_path, resolved_config


def run_liteloc_inference_script_mode(
    *,
    infer_module: Path,
    runtime_yaml_path: Path,
    repo_dir: Path,
    env: Dict[str, str],
    log_path: Path,
) -> int:
    completed = run_python_script(
        script_path=infer_module,
        repo_dir=repo_dir,
        log_path=log_path,
        env=env,
        extra_args=["--infer_params_path", str(runtime_yaml_path)],
    )
    return int(completed.returncode)


def run_liteloc_inference_module_mode(
    *,
    infer_module: Path,
    infer_class_name: str,
    runtime_yaml_path: Path,
    repo_dir: Path,
    log_path: Path,
) -> int:
    add_liteloc_to_syspath_and_env(repo_dir)
    infer_cls = import_from_module_file(infer_module, repo_dir, infer_class_name)

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRunning LiteLoc inference in MODULE mode.\n")
        log.write(f"Infer module: {infer_module}\n")
        log.write(f"Infer class: {infer_class_name}\n")
        log.write(f"Runtime YAML: {runtime_yaml_path}\n\n")
        log.flush()

    try:
        import torch  # type: ignore
        from utils.help_utils import load_yaml_infer  # type: ignore
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Failed to import torch or utils.help_utils.load_yaml_infer.\n")
            log.write(repr(exc) + "\n")
        raise

    infer_params = load_yaml_infer(str(runtime_yaml_path))
    loc_model = torch.load(infer_params.Loc_Model.model_path, weights_only=False)
    mp_params = infer_params.Multi_Process

    t0 = time.time()

    kwargs = {
        "loc_model": loc_model,
        "tiff_path": mp_params.image_path,
        "output_path": mp_params.save_path,
        "time_block_gb": mp_params.time_block_gb,
        "batch_size": mp_params.batch_size,
        "sub_fov_size": mp_params.sub_fov_size,
        "over_cut": mp_params.over_cut,
        "data_queue_size": mp_params.data_queue_size,
        "multi_GPU": mp_params.multi_gpu,
        "num_producers": mp_params.num_producers,
    }

    end_frame_num = get_attr_or_item(mp_params, "end_frame_num", None)
    if end_frame_num is not None:
        kwargs["end_frame_num"] = end_frame_num

    with log_path.open("a", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                analyzer = infer_cls(**kwargs)
                t1 = time.time()
                print(f"LiteLoc module init time: {t1 - t0:.3f} seconds")
                analyzer.start()
                t2 = time.time()
                print(f"LiteLoc module analyze time: {t2 - t1:.3f} seconds")
        except Exception:
            raise

    return 0


# =============================================================================
# Public API: calibration
# =============================================================================


def run_liteloc_calibration(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    backend_config = dict(backend_config or {})
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_liteloc_runtime(profile, backend_config)
    mode = infer_calibration_mode(input_path, profile, runtime.backend_config)

    log_path = out_dir / "liteloc_calibration.log"
    status_path = out_dir / "liteloc_calibration_adapter_status.json"

    if mode == "vector_beads":
        return run_vector_bead_calibration(
            input_path=input_path,
            out_dir=out_dir,
            profile=profile,
            backend_config=runtime.backend_config,
            runtime=runtime,
            log_path=log_path,
            status_path=status_path,
        )

    if mode == "spline_file":
        return register_spline_calibration(
            input_path=input_path,
            out_dir=out_dir,
            profile=profile,
            backend_config=runtime.backend_config,
            runtime=runtime,
            log_path=log_path,
            status_path=status_path,
        )

    if mode in {"none", "analytic"}:
        return register_no_external_calibration(
            input_path=input_path,
            out_dir=out_dir,
            profile=profile,
            backend_config=runtime.backend_config,
            runtime=runtime,
            log_path=log_path,
            status_path=status_path,
            mode=mode,
        )

    raise ValueError(
        f"Unsupported calibration mode: {mode}. Expected vector_beads, spline_file, none, or analytic."
    )


# =============================================================================
# Public API: training
# =============================================================================


def run_liteloc_training(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    start = time.time()
    backend_config = dict(backend_config or {})
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_liteloc_runtime(profile, backend_config)
    train_module = runtime.modules.get("train")

    if train_module is None:
        raise FileNotFoundError(
            "Missing train module. Set liteloc.modules.train in adapters/backend_paths.yml."
        )

    train_class_name = runtime.functions.get("train_class", "LocModel")
    train_execution = normalize_mode(runtime.execution.get("train", "module"), default="module")

    log_path = out_dir / "liteloc_training.log"
    status_path = out_dir / "liteloc_training_adapter_status.json"

    runtime_yaml_path, train_config = build_runtime_train_yaml(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        repo_dir=runtime.repo_dir,
        backend_config=runtime.backend_config,
    )

    before_files = snapshot_files(out_dir)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc training adapter reached.\n")
        log.write(f"Started at: {now_iso()}\n")
        log.write(f"LiteLoc root: {runtime.repo_dir}\n")
        log.write(f"Training module: {train_module}\n")
        log.write(f"Training class: {train_class_name}\n")
        log.write(f"Training execution: {train_execution}\n")
        log.write(f"Runtime train YAML: {runtime_yaml_path}\n")
        log.write(f"Input path: {input_path}\n")
        log.write(f"Output dir: {out_dir}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        try:
            if train_execution == "module":
                add_liteloc_to_syspath_and_env(runtime.repo_dir)
                train_cls = import_from_module_file(train_module, runtime.repo_dir, train_class_name)
                from utils.help_utils import load_yaml  # type: ignore

                params = load_yaml(str(runtime_yaml_path))
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    model = train_cls(params)
                    model.train()
                return_code = 0
                error = ""
            elif train_execution == "script":
                env = add_liteloc_to_syspath_and_env(runtime.repo_dir)
                completed = run_python_script(
                    script_path=train_module,
                    repo_dir=runtime.repo_dir,
                    log_path=log_path,
                    env=env,
                    extra_args=["--train_params_path", str(runtime_yaml_path)],
                )
                return_code = int(completed.returncode)
                error = "" if return_code == 0 else f"script_returncode_{return_code}"
            else:
                raise ValueError(f"Unsupported train execution mode: {train_execution}")

        except Exception as exc:
            return_code = 1
            error = repr(exc)

    checkpoint = find_checkpoint(out_dir, before_files=before_files)
    copied_model = str(checkpoint) if checkpoint else ""
    status = "passed" if return_code == 0 and copied_model else "failed"

    result = {
        "backend_name": "liteloc",
        "backend_status": status,
        "train_status": status,
        "elapsed_seconds": round(time.time() - start, 3),
        "returncode": return_code,
        "error": error,
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "liteloc_root": str(runtime.repo_dir),
        "train_module": str(train_module),
        "train_class": train_class_name,
        "train_execution": train_execution,
        "runtime_train_yaml": str(runtime_yaml_path),
        "model_path": copied_model,
        "log_path": str(log_path),
        "status_json": str(status_path),
        "message": (
            "LiteLoc training completed and a checkpoint was detected."
            if status == "passed"
            else "LiteLoc training failed or no checkpoint was detected. Check liteloc_training.log."
        ),
    }
    write_json(result, status_path)

    if error:
        raise RuntimeError(f"LiteLoc training failed. Check log: {log_path}")

    return result


# =============================================================================
# Public API: inference
# =============================================================================


def run_liteloc_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Optional[str]:
    start = time.time()
    backend_config = dict(backend_config or {})
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_liteloc_runtime(profile, backend_config)
    infer_module = runtime.modules.get("infer")

    if infer_module is None:
        raise FileNotFoundError(
            "Missing inference module. Set liteloc.modules.infer in adapters/backend_paths.yml."
        )

    infer_class_name = runtime.functions.get(
        "infer_class",
        "CompetitiveSmlmDataAnalyzer_multi_producer",
    )

    env = add_liteloc_to_syspath_and_env(runtime.repo_dir)

    runtime_yaml_path, raw_output_path, resolved_runtime = build_runtime_liteloc_infer_yaml(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        repo_dir=runtime.repo_dir,
        backend_config=runtime.backend_config,
    )

    log_name = get_nested(profile, "output", "log_name", default="liteloc.log")
    log_path = out_dir / str(log_name)
    status_path = out_dir / "liteloc_adapter_status.json"
    infer_execution = detect_infer_execution(runtime, profile)

    status: Dict[str, Any] = {
        "created_at": now_iso(),
        "backend": "liteloc",
        "adapter": "liteloc_adapter",
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "repo_dir": str(runtime.repo_dir),
        "infer_module": str(infer_module),
        "infer_class": infer_class_name,
        "infer_execution": infer_execution,
        "runtime_yaml_path": str(runtime_yaml_path),
        "raw_output_path": str(raw_output_path),
        "log_path": str(log_path),
        "resolved_runtime": resolved_runtime,
        "status": "started",
    }
    write_json(status, status_path)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc inference adapter reached.\n")
        log.write(f"Started at: {now_iso()}\n")
        log.write(f"LiteLoc root: {runtime.repo_dir}\n")
        log.write(f"Infer module: {infer_module}\n")
        log.write(f"Infer class: {infer_class_name}\n")
        log.write(f"Inference execution: {infer_execution}\n")
        log.write(f"Runtime YAML: {runtime_yaml_path}\n")
        log.write(f"Raw output path: {raw_output_path}\n")
        log.write(f"PYTHONPATH: {env.get('PYTHONPATH', '')}\n")
        log.write("=" * 80 + "\n")
        log.flush()

    try:
        if infer_execution == "module":
            return_code = run_liteloc_inference_module_mode(
                infer_module=infer_module,
                infer_class_name=infer_class_name,
                runtime_yaml_path=runtime_yaml_path,
                repo_dir=runtime.repo_dir,
                log_path=log_path,
            )
        elif infer_execution == "script":
            return_code = run_liteloc_inference_script_mode(
                infer_module=infer_module,
                runtime_yaml_path=runtime_yaml_path,
                repo_dir=runtime.repo_dir,
                env=env,
                log_path=log_path,
            )
        else:
            raise RuntimeError(f"Unsupported LiteLoc inference execution mode: {infer_execution}")

    except Exception as exc:
        elapsed = round(time.time() - start, 3)
        status.update(
            {
                "status": "failed_exception",
                "return_code": None,
                "elapsed_seconds": elapsed,
                "message": repr(exc),
            }
        )
        write_json(status, status_path)
        raise

    elapsed = round(time.time() - start, 3)
    status.update({"return_code": return_code, "elapsed_seconds": elapsed})

    if return_code != 0:
        status.update(
            {
                "status": "failed",
                "message": "LiteLoc inference failed. Check liteloc.log.",
            }
        )
        write_json(status, status_path)
        raise RuntimeError(f"LiteLoc inference failed. Check log: {log_path}")

    if not raw_output_path.exists():
        status.update(
            {
                "status": "failed_no_csv",
                "message": "LiteLoc finished, but expected raw CSV was not found.",
            }
        )
        write_json(status, status_path)
        raise FileNotFoundError(
            f"LiteLoc finished but did not create expected CSV: {raw_output_path}"
        )

    status.update(
        {
            "status": "passed",
            "message": "Real LiteLoc inference completed successfully.",
            "raw_output_path": str(raw_output_path),
        }
    )
    write_json(status, status_path)

    return str(raw_output_path)


# =============================================================================
# Aliases expected by run_pipeline.py
# =============================================================================


# Calibration aliases

def run_calibration(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_calibration(input_path, out_dir, profile, backend_config, **kwargs)


def calibrate_liteloc(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_calibration(input_path, out_dir, profile, backend_config, **kwargs)


def run_calibrate(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_calibration(input_path, out_dir, profile, backend_config, **kwargs)


def calibrate(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_calibration(input_path, out_dir, profile, backend_config, **kwargs)


# Training aliases

def run_training(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_training(input_path, out_dir, profile, backend_config, **kwargs)


def train_liteloc(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_training(input_path, out_dir, profile, backend_config, **kwargs)


def run_train(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_training(input_path, out_dir, profile, backend_config, **kwargs)


def train(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_liteloc_training(input_path, out_dir, profile, backend_config, **kwargs)


# Inference aliases

def run_liteloc_inference(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    return run_liteloc_one_movie(input_path, out_dir, profile, backend_config, **kwargs)


def run_inference_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    return run_liteloc_one_movie(input_path, out_dir, profile, backend_config, **kwargs)


def run_inference(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    return run_liteloc_one_movie(input_path, out_dir, profile, backend_config, **kwargs)


def run_liteloc(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    return run_liteloc_one_movie(input_path, out_dir, profile, backend_config, **kwargs)


def infer(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
    backend_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    return run_liteloc_one_movie(input_path, out_dir, profile, backend_config, **kwargs)
