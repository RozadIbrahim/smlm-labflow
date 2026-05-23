#!/usr/bin/env python3
"""
run_pipeline.py

Main SMLM LabFlow wrapper pipeline.

Scientist-facing CLI:

    python run_pipeline.py calibrate -i data/beads  -p profiles/dna_paint_standard.yaml
    python run_pipeline.py train     -i data/train  -p profiles/dna_paint_standard.yaml
    python run_pipeline.py infer     -i data/movies -p profiles/dna_paint_standard.yaml

Optional named parent run folder:

    python run_pipeline.py infer \
        -i data/movies \
        -p profiles/dna_paint_standard.yaml \
        -o outputs/npc_condition_A

Backend override, only when needed:

    python run_pipeline.py infer \
        -i data/movies \
        -p profiles/dna_paint_standard.yaml \
        -b liteloc

Output layout for every run:

    parent_run_folder/
    ├── results/
    ├── benchmarks/
    ├── reports/
    ├── registry/
    └── README_RUN.txt

Architecture:
    calibrate/train/infer subcommand
    → profile loading
    → automatic backend resolution
    → backend adapter
    → benchmark pack
    → registry/artifact snapshot
    → supervisor-friendly report

Important:
    This script does NOT open napari and does NOT run interactive Locan analysis.
    napari/Locan review should be run separately in napari_locan_env using
    napari_locan_review.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from benchmark import RuntimeBenchmark
except Exception:  # pragma: no cover - compatibility fallback
    from runtime_benchmark import RuntimeBenchmark  # type: ignore

try:
    from run_folders import RunFolders, prepare_parent_run_folder, write_run_status
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import run_folders.py. Put run_folders.py in the project root "
        "before using this upgraded run_pipeline.py."
    ) from exc


TIFF_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
)

VALID_STEPS = {"calibrate", "train", "infer"}
DEFAULT_BACKEND = "liteloc"


# =============================================================================
# Basic utilities
# =============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def project_root() -> Path:
    return Path(__file__).resolve().parent


def is_tiff(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and name.endswith(TIFF_EXTENSIONS)


def safe_stem(path: Path) -> str:
    name = path.name

    for ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._-")
    return name or "movie"


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "run"


def make_batch_id(path: Path, index: int) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{index:04d}_{safe_stem(path)}_{digest}"


def display_path(path: Path | str | None, base: Optional[Path] = None) -> str:
    if path is None:
        return ""

    if isinstance(path, str) and path.strip() == "":
        return ""

    path = Path(path)
    if str(path).strip() in {"", "."}:
        return ""

    try:
        path = path.expanduser().resolve()
    except Exception:
        return str(path)

    if base is None:
        base = Path.cwd().resolve()
    else:
        base = Path(base).expanduser().resolve()

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_yaml_if_possible(data: Any, path: Path) -> None:
    try:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception:
        json_path = path.with_suffix(".json")
        write_json(data, json_path)


def flatten_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            clean[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            clean[key] = value
    return clean


def write_manifest_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def discover_tiff_movies(
    input_path: Path, max_files: Optional[int] = None
) -> List[Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if not is_tiff(input_path):
            raise ValueError(f"Input file is not TIFF/OME-TIFF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    movies = [p.resolve() for p in input_path.rglob("*") if is_tiff(p)]
    movies = sorted(movies, key=lambda p: str(p).lower())

    if max_files is not None:
        movies = movies[:max_files]

    return movies


# =============================================================================
# Profile loading and profile access
# =============================================================================


def read_profile_yaml(profile_path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to parse profile YAML files. Install it with: pip install PyYAML"
        ) from exc

    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    if not isinstance(profile, dict):
        raise ValueError(f"Profile is not a YAML dictionary: {profile_path}")

    return profile


def deep_merge_profile(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_profile(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_profile_with_extends(
    profile_path: Path,
    stack: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()
    stack = stack or []

    if profile_path in stack:
        chain = " -> ".join(str(path) for path in [*stack, profile_path])
        raise ValueError(f"Profile extends cycle detected: {chain}")
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    profile = read_profile_yaml(profile_path)
    extends_value = profile.get("extends")
    if not extends_value:
        return profile

    parents = extends_value if isinstance(extends_value, list) else [extends_value]
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = profile_path.parent / parent_path
        parent_profile = load_profile_with_extends(parent_path, [*stack, profile_path])
        merged = deep_merge_profile(merged, parent_profile)

    return deep_merge_profile(merged, profile)


def load_profile(profile_path: Path) -> Dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()
    profile = load_profile_with_extends(profile_path)
    profile["profile_path"] = str(profile_path)
    profile["profile_loaded"] = True
    return profile


def get_nested(
    data: Mapping[str, Any], keys: Sequence[str], default: Any = None
) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def set_nested(data: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def profile_name(profile: Mapping[str, Any], profile_path: Path) -> str:
    return safe_name(str(profile.get("profile_name") or profile_path.stem))


def get_backend_name(
    profile: Mapping[str, Any], backend_override: Optional[str]
) -> str:
    if backend_override:
        return backend_override.strip().lower()

    backend_block = profile.get("backend", {})
    if isinstance(backend_block, Mapping):
        return str(backend_block.get("name", DEFAULT_BACKEND)).strip().lower()

    return DEFAULT_BACKEND


def infer_pixel_size_nm(profile: Mapping[str, Any]) -> Optional[float]:
    candidate_paths = [
        ["pixel_size_nm"],
        ["data", "pixel_size_nm"],
        ["input", "pixel_size_nm"],
        ["camera", "pixel_size_nm"],
        ["acquisition", "pixel_size_nm"],
        ["microscope", "pixel_size_nm"],
        ["smlm", "pixel_size_nm"],
    ]

    for keys in candidate_paths:
        value = get_nested(profile, keys, default=None)
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


def infer_default_locprec_nm(profile: Mapping[str, Any]) -> float:
    value = (
        get_nested(profile, ["post_inference", "default_locprec_nm"], None)
        or get_nested(profile, ["downstream", "default_locprec_nm"], None)
        or 20.0
    )
    try:
        return float(value)
    except Exception:
        return 20.0


def infer_default_lpx_px(profile: Mapping[str, Any]) -> float:
    value = (
        get_nested(profile, ["post_inference", "default_lpx_px"], None)
        or get_nested(profile, ["downstream", "default_lpx_px"], None)
        or 1.0
    )
    try:
        return float(value)
    except Exception:
        return 1.0


def infer_export_setting(profile: Mapping[str, Any], name: str) -> Optional[bool]:
    """
    Return explicit export setting from profile, or None to let post_inference decide.
    Supports both downstream.export_picasso and outputs.export_picasso.
    """
    candidates = [
        get_nested(profile, ["downstream", f"export_{name}"], None),
        get_nested(profile, ["outputs", f"export_{name}"], None),
        get_nested(profile, ["exports", name], None),
    ]
    for value in candidates:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in {"true", "yes", "1", "on"}:
                return True
            if lowered in {"false", "no", "0", "off"}:
                return False
    return None


def profile_cli_overrides(
    profile: Mapping[str, Any],
    extra_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runtime overrides come from profile fields, not public CLI paths.
    This keeps the user interface friendly while still giving the resolver what it needs.
    """
    values = {
        "psf_type": get_nested(profile, ["experiment", "psf_type"], None)
        or get_nested(profile, ["psf", "type"], None),
        "psf_dimensionality": get_nested(
            profile, ["experiment", "dimensionality"], None
        )
        or get_nested(profile, ["psf", "dimensionality"], None),
        "calibration_mode": get_nested(profile, ["calibration", "mode"], None)
        or get_nested(profile, ["psf", "calibration_mode"], None),
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
        "coord_units": infer_coord_units(profile),
        "pixel_size_nm": infer_pixel_size_nm(profile),
    }
    if extra_overrides:
        values.update(dict(extra_overrides))
    return {k: v for k, v in values.items() if v is not None}


# =============================================================================
# Dynamic call helpers
# =============================================================================


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Call a function with only the keyword arguments it supports.
    Allows old and new adapters to coexist.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    parameters = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )

    if accepts_kwargs:
        return fn(**kwargs)

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return fn(**supported_kwargs)


def import_optional_module(module_name: str) -> Tuple[Optional[Any], str]:
    try:
        return importlib.import_module(module_name), f"imported {module_name}"
    except ModuleNotFoundError:
        return None, f"{module_name} not found"
    except Exception as exc:
        return None, f"{module_name} import failed: {repr(exc)}"


# =============================================================================
# Resolver bridge
# =============================================================================


def default_backend_paths_file() -> Path:
    return project_root() / "adapters" / "backend_paths.yml"


def load_backend_paths_fallback() -> Dict[str, Any]:
    path = default_backend_paths_file().resolve()

    if not path.exists():
        return {
            "status": "missing_backend_paths_file",
            "backend_paths_file": str(path),
            "backend_paths": {},
        }

    try:
        import yaml
    except ImportError:
        return {
            "status": "pyyaml_missing",
            "backend_paths_file": str(path),
            "backend_paths": {},
        }

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"backend_paths.yml is not a YAML dictionary: {path}")

    return {
        "status": "fallback_loaded_backend_paths",
        "backend_paths_file": str(path),
        "backend_paths": data,
    }


def resolve_backend_runtime_config(
    step: str,
    profile: Dict[str, Any],
    backend_name: str,
    folders: RunFolders,
    extra_cli_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve backend runtime automatically.

    Public CLI does not expose backend_paths, calibration_file, or model_path.
    The resolver should use:
        profile YAML
        adapters/backend_paths.yml
        global registry under outputs/registry
        current run folders
    """
    cli_overrides = profile_cli_overrides(profile, extra_cli_overrides)
    backend_paths_file = default_backend_paths_file().resolve()

    resolver, import_message = import_optional_module("adapters.resolver")
    if resolver is None:
        fallback = load_backend_paths_fallback()
        return {
            "status": "resolver_import_failed",
            "message": import_message,
            "step": step,
            "backend_name": backend_name,
            "backend_paths_file": str(backend_paths_file),
            "project_root": str(project_root()),
            "run_parent": str(folders.parent),
            "results_dir": str(folders.results),
            "benchmarks_dir": str(folders.benchmarks),
            "reports_dir": str(folders.reports),
            "registry_dir": str(folders.registry),
            "cli_overrides": cli_overrides,
            **fallback,
        }

    resolver_function_names = [
        "resolve_backend_runtime",
        "resolve_backend_config",
        "resolve_backend",
        "resolve_liteloc_runtime",
        "resolve_liteloc_paths",
        "resolve_paths",
    ]

    resolver_fn: Optional[Callable[..., Any]] = None
    resolver_fn_name = ""
    for name in resolver_function_names:
        candidate = getattr(resolver, name, None)
        if callable(candidate):
            resolver_fn = candidate
            resolver_fn_name = name
            break

    if resolver_fn is None:
        fallback = load_backend_paths_fallback()
        return {
            "status": "resolver_function_missing",
            "message": (
                "adapters.resolver was found, but no supported public resolver "
                "function was found. Add resolve_backend_runtime()."
            ),
            "step": step,
            "backend_name": backend_name,
            "backend_paths_file": str(backend_paths_file),
            "project_root": str(project_root()),
            "run_parent": str(folders.parent),
            "results_dir": str(folders.results),
            "benchmarks_dir": str(folders.benchmarks),
            "reports_dir": str(folders.reports),
            "registry_dir": str(folders.registry),
            "cli_overrides": cli_overrides,
            **fallback,
        }

    resolved = call_with_supported_kwargs(
        resolver_fn,
        step=step,
        command=step,
        profile=profile,
        backend_name=backend_name,
        backend_paths_file=backend_paths_file,
        backend_paths_path=backend_paths_file,
        cli_overrides=cli_overrides,
        project_root=project_root(),
        out_dir=folders.results,
        run_parent=folders.parent,
        results_dir=folders.results,
        benchmarks_dir=folders.benchmarks,
        reports_dir=folders.reports,
        registry_dir=folders.registry,
    )

    if not isinstance(resolved, dict):
        raise TypeError(
            f"adapters.resolver.{resolver_fn_name}() must return a dictionary, "
            f"got {type(resolved).__name__}"
        )

    # Profile-derived overrides win over fallback defaults, but real resolver may also set richer keys.
    for key, value in cli_overrides.items():
        resolved.setdefault(key, value)

    resolved.setdefault("status", "passed")
    resolved.setdefault("step", step)
    resolved.setdefault("backend_name", backend_name)
    resolved.setdefault("backend_paths_file", str(backend_paths_file))
    resolved.setdefault("project_root", str(project_root()))
    resolved.setdefault("run_parent", str(folders.parent))
    resolved.setdefault("results_dir", str(folders.results))
    resolved.setdefault("benchmarks_dir", str(folders.benchmarks))
    resolved.setdefault("reports_dir", str(folders.reports))
    resolved.setdefault("registry_dir", str(folders.registry))
    resolved.setdefault("cli_overrides", cli_overrides)
    resolved.setdefault("resolver_function", f"adapters.resolver.{resolver_fn_name}")
    resolved["resolved_at"] = now_iso()
    return resolved


def validate_backend_connection(
    step: str,
    input_path: Path,
    backend_name: str,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
) -> None:
    """
    Fail early when the configured backend cannot be reached.

    This deliberately checks the real adapter runtime before starting a run, so
    a missing LiteLoc root/module/dependency does not become a vague
    "Pipeline complete / Status: failed" footer.
    """
    if backend_name.lower().strip() != "liteloc":
        return

    try:
        from adapters import liteloc_adapter
    except Exception as exc:
        raise RuntimeError(
            f"Could not import the LiteLoc adapter: {repr(exc)}"
        ) from exc

    try:
        runtime = liteloc_adapter.resolve_liteloc_runtime(profile, backend_config)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to LiteLoc backend. Check adapters/backend_paths.yml "
            f"and the LiteLoc installation. Resolver root was "
            f"{backend_config.get('root') or backend_config.get('liteloc_root') or '<empty>'}. "
            f"Original error: {repr(exc)}"
        ) from exc

    required_module = ""
    required_attr = ""
    if step == "calibrate":
        mode = liteloc_adapter.infer_calibration_mode(
            input_path,
            profile,
            runtime.backend_config,
        )
        if mode == "vector_beads":
            required_module = "vector_calibration"
            required_attr = runtime.functions.get(
                "vector_calibration",
                "beads_psf_calibrate",
            )
        elif mode == "spline_file":
            required_module = "spline_calibration_io"
            required_attr = runtime.functions.get(
                "spline_loader_class",
                "SMAPSplineCoefficient",
            )
    elif step == "train":
        required_module = "train"
        required_attr = runtime.functions.get("train_class", "LocModel")
    elif step == "infer":
        required_module = "infer"
        required_attr = runtime.functions.get(
            "infer_class",
            "CompetitiveSmlmDataAnalyzer_multi_producer",
        )

    if not required_module:
        return

    module_path = runtime.modules.get(required_module)
    if module_path is None:
        raise RuntimeError(
            "Could not connect to LiteLoc backend. Required module "
            f"{required_module!r} was not found under {runtime.repo_dir}. "
            "Check adapters/backend_paths.yml module paths."
        )

    try:
        liteloc_adapter.import_from_module_file(
            module_path,
            runtime.repo_dir,
            required_attr,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the required LiteLoc backend entry point "
            f"{required_attr!r} from {module_path}. Original error: {repr(exc)}"
        ) from exc


# =============================================================================
# Adapter discovery and execution
# =============================================================================


def backend_module_name(backend_name: str) -> str:
    backend_name = backend_name.lower().strip()
    if backend_name == "liteloc":
        return "adapters.liteloc_adapter"
    return f"adapters.{backend_name}_adapter"


def get_backend_step_function(
    backend_name: str,
    step: str,
) -> Tuple[Optional[Callable[..., Any]], str]:
    module_name = backend_module_name(backend_name)
    module, import_message = import_optional_module(module_name)

    if module is None:
        return None, import_message

    candidates_by_step = {
        "calibrate": [
            "run_liteloc_calibration",
            "run_calibration",
            "calibrate_liteloc",
            "run_calibrate",
            "calibrate",
        ],
        "train": [
            "run_liteloc_training",
            "run_training",
            "train_liteloc",
            "run_train",
            "train",
        ],
        "infer": [
            "run_liteloc_one_movie",
            "run_inference_one_movie",
            "run_liteloc_inference",
            "run_inference",
            "run_liteloc",
            "infer",
        ],
    }

    for name in candidates_by_step.get(step, []):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, f"using {module_name}.{name}"

    return None, f"No supported {step} function found in {module_name}"


def call_backend_function(
    fn: Callable[..., Any],
    step: str,
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
    batch_index: Optional[int] = None,
) -> Any:
    try:
        return call_with_supported_kwargs(
            fn,
            step=step,
            command=step,
            input_path=input_path,
            movie_path=input_path,
            train_path=input_path,
            calibration_path=input_path,
            out_dir=out_dir,
            output_dir=out_dir,
            results_dir=out_dir,
            profile=profile,
            backend_config=backend_config,
            runtime_config=backend_config,
            batch_index=batch_index,
        )
    except TypeError:
        return fn(input_path, out_dir, profile)


def normalize_backend_result(
    result: Any,
    backend_name: str,
    step: str,
    message: str,
) -> Dict[str, Any]:
    if isinstance(result, dict):
        clean = dict(result)
    elif result is None:
        clean = {}
    else:
        clean = {"output_path": str(result)}

    status_key = f"{step}_status"
    clean.setdefault(
        status_key, "passed" if result is not None else "pending_no_output"
    )
    clean.setdefault("backend_status", clean.get(status_key, "passed"))
    clean.setdefault("backend_name", backend_name)
    clean.setdefault("backend_message", message)

    if step == "calibrate":
        artifact = (
            clean.get("calibration_file")
            or clean.get("calibration_path")
            or clean.get("calibration_model")
            or clean.get("output_path")
        )
        clean.setdefault("calibration_file", artifact or "")

    elif step == "train":
        artifact = (
            clean.get("model_path")
            or clean.get("checkpoint_path")
            or clean.get("checkpoint")
            or clean.get("output_path")
        )
        clean.setdefault("model_path", artifact or "")

    elif step == "infer":
        artifact = (
            clean.get("raw_output_path")
            or clean.get("raw_output")
            or clean.get("localization_csv")
            or clean.get("output_path")
        )
        clean.setdefault("raw_output_path", artifact or "")

    return clean


def default_backend_log_path(step: str, out_dir: Path, profile: Mapping[str, Any]) -> Path:
    if step == "calibrate":
        return out_dir / "liteloc_calibration.log"
    if step == "train":
        return out_dir / "liteloc_training.log"
    log_name = get_nested(profile, ["output", "log_name"], "liteloc.log")
    return out_dir / str(log_name)


def run_backend_step(
    step: str,
    backend_name: str,
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    backend_config: Dict[str, Any],
    batch_index: Optional[int] = None,
) -> Dict[str, Any]:
    fn, message = get_backend_step_function(backend_name=backend_name, step=step)

    if fn is None:
        return {
            "backend_status": "pending_adapter_missing",
            f"{step}_status": "pending_adapter_missing",
            "backend_name": backend_name,
            "backend_message": message,
        }

    try:
        result = call_backend_function(
            fn=fn,
            step=step,
            input_path=input_path,
            out_dir=out_dir,
            profile=profile,
            backend_config=backend_config,
            batch_index=batch_index,
        )
        return normalize_backend_result(
            result=result,
            backend_name=backend_name,
            step=step,
            message=message,
        )
    except Exception as exc:
        log_path = default_backend_log_path(step, out_dir, profile)
        return {
            "backend_status": "failed",
            f"{step}_status": "failed",
            "backend_name": backend_name,
            "backend_message": repr(exc),
            "backend_log_path": str(log_path),
        }


# =============================================================================
# QC, post-inference, combine, report
# =============================================================================


def get_qc_function() -> Callable[..., Dict[str, Any]]:
    try:
        from qc_input import qc_one_movie
    except Exception as exc:
        raise ImportError(
            "Could not import qc_one_movie from qc_input.py. Make sure qc_input.py exists."
        ) from exc
    return qc_one_movie


def run_qc_safely(
    qc_one_movie: Callable[..., Dict[str, Any]],
    movie_path: Path,
    movie_out_dir: Path,
) -> Dict[str, Any]:
    try:
        return call_with_supported_kwargs(
            qc_one_movie,
            input_path=movie_path,
            movie_path=movie_path,
            out_dir=movie_out_dir,
            output_dir=movie_out_dir,
        )
    except TypeError:
        try:
            return qc_one_movie(movie_path, movie_out_dir)
        except Exception as exc:
            return {"qc_status": "failed", "qc_error": repr(exc)}
    except Exception as exc:
        return {"qc_status": "failed", "qc_error": repr(exc)}


def get_post_inference_function() -> Callable[..., Dict[str, Any]]:
    try:
        from post_inference import run_post_inference
    except Exception as exc:
        raise ImportError(
            "Could not import run_post_inference from post_inference.py. "
            "Make sure post_inference.py exists."
        ) from exc
    return run_post_inference


def run_post_inference_safely(
    run_post_inference: Callable[..., Dict[str, Any]],
    raw_output_path: str | Path,
    movie_out_dir: Path,
    profile: Dict[str, Any],
    backend_name: str,
    source_file: Path,
    coord_units: str,
    pixel_size_nm: Optional[float],
    default_locprec_nm: float,
    default_lpx_px: float,
    napari_units: str,
    locan_units: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    try:
        post_summary = call_with_supported_kwargs(
            run_post_inference,
            input_path=raw_output_path,
            raw_output_path=raw_output_path,
            out_dir=movie_out_dir,
            output_dir=movie_out_dir,
            profile=profile,
            backend_name=backend_name,
            source_file=str(source_file),
            coord_units=coord_units,
            pixel_size_nm=pixel_size_nm,
            default_locprec_nm=default_locprec_nm,
            default_lpx_px=default_lpx_px,
            napari_units=napari_units,
            locan_units=locan_units,
            export_smap_enabled=infer_export_setting(profile, "smap"),
            export_picasso_enabled=infer_export_setting(profile, "picasso"),
            export_napari_enabled=infer_export_setting(profile, "napari"),
            export_locan_enabled=infer_export_setting(profile, "locan"),
        )

        canonical_output_path = post_summary.get("canonical_csv", "")
        canonical_result = {
            "canonical_status": "passed" if canonical_output_path else "failed",
            "canonical_message": "post_inference.run_post_inference completed",
            "canonical_output_path": canonical_output_path,
            "post_inference_summary": post_summary.get(
                "post_inference_summary",
                str(movie_out_dir / "post_inference_summary.json"),
            ),
            "localization_qc": post_summary.get("localization_qc", ""),
        }
        export_result = {
            "status": post_summary.get("status", "unknown"),
            "exports": post_summary.get("exports", {}),
            "plots": post_summary.get("plots", {}),
            "quality_flags": post_summary.get("quality_flags", []),
            "coord_units_detected": post_summary.get(
                "coord_units_detected", coord_units
            ),
            "pixel_size_nm": post_summary.get("pixel_size_nm", pixel_size_nm),
        }
        return post_summary, canonical_result, export_result

    except Exception as exc:
        canonical_result = {
            "canonical_status": "failed",
            "canonical_message": repr(exc),
            "canonical_output_path": "",
            "post_inference_summary": "",
            "localization_qc": "",
        }
        export_result = {"status": "failed", "error": repr(exc)}
        return {}, canonical_result, export_result


def build_export_validation_map(
    canonical_path: str | Path | None,
    export_result: Mapping[str, Any],
) -> Dict[str, str | Path | None]:
    exports: Dict[str, str | Path | None] = {"canonical": canonical_path}

    raw_exports = export_result.get("exports", {})
    if isinstance(raw_exports, Mapping):
        for key, value in raw_exports.items():
            if isinstance(value, Mapping):
                path = value.get("path") or value.get("file") or value.get("output")
                exports[str(key)] = path
            else:
                exports[str(key)] = value  # type: ignore[assignment]

    known_aliases = {
        "picasso": ["picasso_csv", "picasso_output", "picasso_localizations"],
        "smap": ["smap_csv", "smap_output", "smap_localizations"],
        "napari": ["napari_csv", "napari_output", "napari_points"],
        "locan": ["locan_csv", "locan_output", "locan_localizations"],
    }
    for export_name, keys in known_aliases.items():
        if export_name in exports and exports[export_name]:
            continue
        for key in keys:
            value = export_result.get(key)
            if value:
                exports[export_name] = value
                break

    return exports


def run_quality_metrics_safely(
    *,
    step: str,
    bench: RuntimeBenchmark,
    folders: RunFolders,
    profile: Mapping[str, Any],
    paths: Mapping[str, Any],
    batch_index: Optional[int] = None,
    reports_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from quality_metrics import run_quality
    except Exception as exc:
        payload = {
            "step": step,
            "status": "not_available",
            "flags": [
                {
                    "severity": "warning",
                    "code": "quality_metrics_import_failed",
                    "message": repr(exc),
                }
            ],
            "output_paths": {},
        }
        bench.add_quality_metrics_result(step, payload, batch_index=batch_index)
        return payload

    out_dir = folders.reports
    quality_benchmarks_dir = folders.benchmarks
    if reports_subdir:
        safe_subdir = safe_name(reports_subdir)
        out_dir = folders.reports / safe_subdir
        quality_benchmarks_dir = folders.benchmarks / safe_subdir

    quality_paths: Dict[str, Any] = dict(paths)
    quality_paths.setdefault("out_dir", out_dir)
    quality_paths.setdefault("reports_dir", out_dir)
    quality_paths.setdefault("benchmarks_dir", quality_benchmarks_dir)

    try:
        payload = run_quality(
            step,
            paths=quality_paths,
            profile=profile,
            out_dir=out_dir,
        )
    except Exception as exc:
        payload = {
            "step": step,
            "status": "error",
            "flags": [
                {
                    "severity": "error",
                    "code": "quality_metrics_failed",
                    "message": repr(exc),
                }
            ],
            "output_paths": {},
        }

    bench.add_quality_metrics_result(step, payload, batch_index=batch_index)
    return payload


def resolve_optional_profile_path(value: Any, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"auto", "none", "null", "false"}:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def infer_truth_csv_for_movie(
    profile: Mapping[str, Any],
    movie_path: Path,
    batch_index: int,
) -> Optional[Path]:
    truth_block = profile.get("truth", {})
    benchmark_block = profile.get("benchmark", {})

    candidates: List[Any] = [
        get_nested(profile, ["truth", "csv"], None),
        get_nested(profile, ["truth", "path"], None),
        get_nested(profile, ["ground_truth", "csv"], None),
        get_nested(profile, ["ground_truth", "path"], None),
        get_nested(profile, ["benchmark", "truth_csv"], None),
    ]

    if isinstance(truth_block, Mapping):
        by_file = truth_block.get("by_file")
        if isinstance(by_file, Mapping):
            candidates.insert(0, by_file.get(movie_path.name))
            candidates.insert(0, by_file.get(movie_path.stem))
        by_index = truth_block.get("by_index")
        if isinstance(by_index, Mapping):
            candidates.insert(0, by_index.get(str(batch_index)))
            candidates.insert(0, by_index.get(batch_index))
    if isinstance(benchmark_block, Mapping):
        by_file = benchmark_block.get("truth_by_file")
        if isinstance(by_file, Mapping):
            candidates.insert(0, by_file.get(movie_path.name))
            candidates.insert(0, by_file.get(movie_path.stem))

    base_dir = project_root()
    profile_path = profile.get("profile_path")
    if profile_path:
        try:
            base_dir = Path(str(profile_path)).expanduser().resolve().parent
        except Exception:
            base_dir = project_root()

    for candidate in candidates:
        path = resolve_optional_profile_path(candidate, base_dir)
        if path is not None and path.exists():
            return path
    return None


def truth_match_radius_xy_nm(profile: Mapping[str, Any]) -> float:
    value = (
        get_nested(profile, ["truth", "match_radius_xy_nm"], None)
        or get_nested(profile, ["benchmark", "match_radius_xy_nm"], None)
        or 50.0
    )
    try:
        return float(value)
    except Exception:
        return 50.0


def truth_match_radius_z_nm(profile: Mapping[str, Any]) -> float:
    value = (
        get_nested(profile, ["truth", "match_radius_z_nm"], None)
        or get_nested(profile, ["benchmark", "match_radius_z_nm"], None)
        or 100.0
    )
    try:
        return float(value)
    except Exception:
        return 100.0


def combine_outputs_safely(folders: RunFolders) -> Dict[str, Any]:
    try:
        from combine_run_outputs import combine_run_outputs
    except Exception as exc:
        return {
            "status": "not_available",
            "error": repr(exc),
            "combined_dir": "",
            "outputs": {},
        }

    for candidate in [folders.results, folders.parent]:
        try:
            result = combine_run_outputs(candidate)
            if isinstance(result, dict):
                return result
            return {"status": "passed", "combined_dir": str(result), "outputs": {}}
        except Exception as exc:
            last_error = repr(exc)

    return {
        "status": "failed",
        "error": last_error,
        "combined_dir": "",
        "outputs": {},
    }


def generate_report_safely(folders: RunFolders) -> Dict[str, Any]:
    try:
        from generate_run_report import generate_run_report
    except Exception as exc:
        return {"status": "not_available", "error": repr(exc)}

    for candidate in [folders.parent, folders.results]:
        try:
            outputs = generate_run_report(candidate)
            if isinstance(outputs, dict):
                return {
                    "status": "passed",
                    "markdown_report": outputs.get("markdown_report", ""),
                    "html_report": outputs.get("html_report", ""),
                    "assets_dir": outputs.get("assets_dir", ""),
                }
            return {"status": "passed", "html_report": str(outputs)}
        except Exception as exc:
            last_error = repr(exc)

    return {"status": "failed", "error": last_error}


# =============================================================================
# Registry and artifacts
# =============================================================================


def global_registry_dir(folders: RunFolders) -> Path:
    """
    Global registry lives under outputs/registry if the run folder is outputs/<run>.
    Otherwise it lives next to the chosen parent folder as <parent_parent>/registry.
    """
    return folders.parent.parent / "registry"


def artifact_id(step: str, folders: RunFolders) -> str:
    return safe_name(f"{step}_{folders.parent.name}")


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_artifact_snapshot(
    step: str,
    folders: RunFolders,
    profile: Mapping[str, Any],
    backend_name: str,
    backend_config: Mapping[str, Any],
    step_result: Mapping[str, Any],
    status: str,
) -> Dict[str, Any]:
    profile_path = str(profile.get("profile_path", ""))
    profile_label = profile_name(profile, Path(profile_path or "profile.yaml"))
    art_id = artifact_id(step, folders)

    artifact: Dict[str, Any] = {
        "id": art_id,
        "step": step,
        "status": status,
        "created_at": now_iso(),
        "profile_name": profile_label,
        "profile_path": profile_path,
        "backend": backend_name,
        "run_parent": str(folders.parent),
        "results_dir": str(folders.results),
        "benchmarks_dir": str(folders.benchmarks),
        "reports_dir": str(folders.reports),
        "registry_dir": str(folders.registry),
        "psf_type": backend_config.get("psf_type")
        or get_nested(profile, ["psf", "type"], None),
        "dimensionality": backend_config.get("psf_dimensionality")
        or get_nested(profile, ["psf", "dimensionality"], None),
        "backend_config": dict(backend_config),
        "step_result": dict(step_result),
    }

    if step == "calibrate":
        artifact["calibration_file"] = step_result.get("calibration_file", "")
    elif step == "train":
        artifact["model_path"] = step_result.get("model_path", "")
        artifact["used_calibration"] = backend_config.get("calibration_file", "")
    elif step == "infer":
        artifact["used_model"] = backend_config.get("model_path", "")
        artifact["used_calibration"] = backend_config.get("calibration_file", "")

    local_artifact_path = folders.registry / "artifact.json"
    write_json(artifact, local_artifact_path)

    global_dir = global_registry_dir(folders)
    global_dir.mkdir(parents=True, exist_ok=True)
    global_artifacts_path = global_dir / "artifacts.json"
    registry = load_json_if_exists(global_artifacts_path, {"artifacts": []})
    if not isinstance(registry, dict):
        registry = {"artifacts": []}
    artifacts = registry.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []

    artifacts = [
        item
        for item in artifacts
        if not (isinstance(item, dict) and item.get("id") == art_id)
    ]
    artifacts.append(artifact)
    registry["artifacts"] = artifacts
    registry["updated_at"] = now_iso()
    write_json(registry, global_artifacts_path)

    if step == "calibrate" and status == "passed" and artifact.get("calibration_file"):
        write_json(artifact, global_dir / "latest_calibration.json")
    elif step == "train" and status == "passed" and artifact.get("model_path"):
        write_json(artifact, global_dir / "latest_model.json")
    elif step == "infer" and status in {"passed", "warning"}:
        write_json(artifact, global_dir / "latest_results.json")

    return artifact


# =============================================================================
# Command implementations
# =============================================================================


def prepare_run_context(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str, RunFolders, Dict[str, Any]]:
    step = args.command
    input_path = Path(args.i).expanduser().resolve()
    profile_path = Path(args.p).expanduser().resolve()
    profile = load_profile(profile_path)
    backend_name = get_backend_name(profile, getattr(args, "b", None))

    folders = prepare_parent_run_folder(
        step=step,
        input_path=input_path,
        profile_path=profile_path,
        output_arg=Path(args.o) if getattr(args, "o", None) else None,
        name=getattr(args, "name", None),
        overwrite=bool(getattr(args, "overwrite", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        command=sys.argv,
        extra_manifest={
            "backend_name": backend_name,
            "profile_name": profile_name(profile, profile_path),
        },
    )

    extra_cli_overrides: Dict[str, Any] = {}
    if getattr(args, "calib_mode", None):
        extra_cli_overrides["calibration_mode"] = args.calib_mode

    backend_config = resolve_backend_runtime_config(
        step=step,
        profile=profile,
        backend_name=backend_name,
        folders=folders,
        extra_cli_overrides=extra_cli_overrides,
    )

    if not getattr(args, "dry_run", False):
        validate_backend_connection(
            step=step,
            input_path=input_path,
            backend_name=backend_name,
            profile=profile,
            backend_config=backend_config,
        )

    profile["_runtime"] = {"backend": backend_config}
    profile["resolved_backend"] = backend_config

    if not getattr(args, "dry_run", False):
        write_json(backend_config, folders.registry / "resolved_config.json")
        write_yaml_if_possible(
            backend_config, folders.registry / "resolved_config.yaml"
        )

    return profile, backend_name, folders, backend_config


def print_header(
    step: str,
    input_path: Path,
    profile_path: Path,
    folders: RunFolders,
    backend_name: str,
    backend_config: Mapping[str, Any],
    n_movies: Optional[int] = None,
) -> None:
    print("=" * 70)
    print(f"SMLM LabFlow pipeline — {step}")
    print("=" * 70)
    print(f"Input:        {display_path(input_path)}")
    print(f"Run folder:   {display_path(folders.parent)}")
    print(f"Results:      {display_path(folders.results)}")
    print(f"Benchmarks:   {display_path(folders.benchmarks)}")
    print(f"Reports:      {display_path(folders.reports)}")
    print(f"Registry:     {display_path(folders.registry)}")
    print(f"Profile:      {display_path(profile_path)}")
    print(f"Backend:      {backend_name}")
    backend_root = backend_config.get("root") or backend_config.get("liteloc_root", "")
    if backend_root:
        print(f"Backend root: {display_path(backend_root)}")
    print(
        f"Resolver:     {backend_config.get('resolver_function', backend_config.get('status', ''))}"
    )
    print(f"PSF type:     {backend_config.get('psf_type', '')}")
    print(f"Calibration:  {display_path(backend_config.get('calibration_file', ''))}")
    print(f"Model:        {display_path(backend_config.get('model_path', ''))}")
    print(f"Device:       {backend_config.get('device', '')}")
    if n_movies is not None:
        print(f"Movies:       {n_movies}")
    print("=" * 70)
    print()


def dry_run_result(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    backend_name: str,
    folders: RunFolders,
    backend_config: Mapping[str, Any],
    movies: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    result = {
        "status": "dry_run",
        "step": args.command,
        "message": "No files were written and no backend stages were executed.",
        "input": str(Path(args.i).expanduser().resolve()),
        "profile": str(Path(args.p).expanduser().resolve()),
        "backend_name": backend_name,
        "run_folder": folders.as_dict(),
        "resolved_backend_config": dict(backend_config),
        "profile_name": profile.get("profile_name", ""),
        "n_movies_detected": len(movies) if movies is not None else None,
        "movies": [str(movie) for movie in movies] if movies is not None else [],
    }
    print_header(
        step=args.command,
        input_path=Path(args.i).expanduser().resolve(),
        profile_path=Path(args.p).expanduser().resolve(),
        folders=folders,
        backend_name=backend_name,
        backend_config=backend_config,
        n_movies=len(movies) if movies is not None else None,
    )
    print("Dry run enabled. Nothing was executed.")
    print("Planned parent run folder:")
    print(display_path(folders.parent))
    print("=" * 70)
    return result


def run_calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    profile, backend_name, folders, backend_config = prepare_run_context(args)
    input_path = Path(args.i).expanduser().resolve()
    profile_path = Path(args.p).expanduser().resolve()

    if args.dry_run:
        return dry_run_result(args, profile, backend_name, folders, backend_config)

    print_header(
        "calibrate", input_path, profile_path, folders, backend_name, backend_config
    )
    write_run_status(folders, status="running", message="Calibration started.")

    bench = RuntimeBenchmark(out_dir=folders.benchmarks)

    movies = discover_tiff_movies(input_path)
    for index, movie_path in enumerate(movies, start=1):
        bench.benchmark_input_movie(movie_path, batch_index=index)

    with bench.stage(
        "backend_calibrate", input_path=input_path, out_dir=folders.results
    ):
        backend_result = run_backend_step(
            step="calibrate",
            backend_name=backend_name,
            input_path=input_path,
            out_dir=folders.results,
            profile=profile,
            backend_config=backend_config,
        )

    quality_result = run_quality_metrics_safely(
        step="calibrate",
        bench=bench,
        folders=folders,
        profile=profile,
        paths={
            "run_dir": folders.parent,
            "calibration_file": backend_result.get("calibration_file", ""),
            "out_dir": folders.reports,
            "reports_dir": folders.reports,
        },
    )

    benchmark_summary = bench.finalize()
    status = (
        "passed"
        if backend_result.get("calibrate_status") == "passed"
        or backend_result.get("backend_status") == "passed"
        else "failed"
    )
    if status == "passed" and quality_result.get("status") in {"fail", "error"}:
        status = "warning"

    artifact = write_artifact_snapshot(
        step="calibrate",
        folders=folders,
        profile=profile,
        backend_name=backend_name,
        backend_config=backend_config,
        step_result=backend_result,
        status=status,
    )

    summary = {
        "created_at": now_iso(),
        "step": "calibrate",
        "status": status,
        "input": str(input_path),
        "profile_path": str(profile_path),
        "backend_name": backend_name,
        "run_parent": str(folders.parent),
        "results_dir": str(folders.results),
        "benchmarks_dir": str(folders.benchmarks),
        "reports_dir": str(folders.reports),
        "registry_dir": str(folders.registry),
        "n_input_movies": len(movies),
        "backend_result": backend_result,
        "quality_metrics": quality_result,
        "benchmark": benchmark_summary,
        "artifact": artifact,
    }
    write_json(summary, folders.results / "calibration_summary.json")
    write_json(summary, folders.registry / "run_summary.json")

    report = generate_report_safely(folders)
    summary["report"] = report
    write_json(summary, folders.registry / "run_summary.json")
    write_run_status(
        folders,
        status=status,
        message="Calibration completed.",
        extra={"artifact_id": artifact.get("id")},
    )

    print_footer(folders, summary)
    return summary


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    profile, backend_name, folders, backend_config = prepare_run_context(args)
    input_path = Path(args.i).expanduser().resolve()
    profile_path = Path(args.p).expanduser().resolve()

    if args.dry_run:
        return dry_run_result(args, profile, backend_name, folders, backend_config)

    print_header(
        "train", input_path, profile_path, folders, backend_name, backend_config
    )
    write_run_status(folders, status="running", message="Training started.")

    bench = RuntimeBenchmark(out_dir=folders.benchmarks)

    movies = discover_tiff_movies(input_path)
    for index, movie_path in enumerate(movies, start=1):
        bench.benchmark_input_movie(movie_path, batch_index=index)

    with bench.stage("backend_train", input_path=input_path, out_dir=folders.results):
        backend_result = run_backend_step(
            step="train",
            backend_name=backend_name,
            input_path=input_path,
            out_dir=folders.results,
            profile=profile,
            backend_config=backend_config,
        )

    quality_result = run_quality_metrics_safely(
        step="train",
        bench=bench,
        folders=folders,
        profile=profile,
        paths={
            "run_dir": folders.parent,
            "checkpoint": backend_result.get("model_path", ""),
            "model_path": backend_result.get("model_path", ""),
            "out_dir": folders.reports,
            "reports_dir": folders.reports,
        },
    )

    benchmark_summary = bench.finalize()
    status = (
        "passed"
        if backend_result.get("train_status") == "passed"
        or backend_result.get("backend_status") == "passed"
        else "failed"
    )
    if status == "passed" and quality_result.get("status") in {"fail", "error"}:
        status = "warning"

    artifact = write_artifact_snapshot(
        step="train",
        folders=folders,
        profile=profile,
        backend_name=backend_name,
        backend_config=backend_config,
        step_result=backend_result,
        status=status,
    )

    summary = {
        "created_at": now_iso(),
        "step": "train",
        "status": status,
        "input": str(input_path),
        "profile_path": str(profile_path),
        "backend_name": backend_name,
        "run_parent": str(folders.parent),
        "results_dir": str(folders.results),
        "benchmarks_dir": str(folders.benchmarks),
        "reports_dir": str(folders.reports),
        "registry_dir": str(folders.registry),
        "n_input_movies": len(movies),
        "backend_result": backend_result,
        "quality_metrics": quality_result,
        "benchmark": benchmark_summary,
        "artifact": artifact,
    }
    write_json(summary, folders.results / "training_summary.json")
    write_json(summary, folders.registry / "run_summary.json")

    report = generate_report_safely(folders)
    summary["report"] = report
    write_json(summary, folders.registry / "run_summary.json")
    write_run_status(
        folders,
        status=status,
        message="Training completed.",
        extra={"artifact_id": artifact.get("id")},
    )

    print_footer(folders, summary)
    return summary


def run_infer(args: argparse.Namespace) -> Dict[str, Any]:
    profile, backend_name, folders, backend_config = prepare_run_context(args)
    input_path = Path(args.i).expanduser().resolve()
    profile_path = Path(args.p).expanduser().resolve()

    movies = discover_tiff_movies(input_path, max_files=args.max_files)
    if not movies:
        raise RuntimeError(f"No TIFF/OME-TIFF files found in: {input_path}")

    if args.dry_run:
        return dry_run_result(
            args, profile, backend_name, folders, backend_config, movies=movies
        )

    print_header(
        "infer",
        input_path,
        profile_path,
        folders,
        backend_name,
        backend_config,
        n_movies=len(movies),
    )
    write_run_status(folders, status="running", message="Inference started.")

    bench = RuntimeBenchmark(out_dir=folders.benchmarks)
    qc_one_movie = get_qc_function()
    run_post_inference = get_post_inference_function()

    coord_units = infer_coord_units(profile)
    pixel_size_nm = infer_pixel_size_nm(profile)
    default_locprec_nm = infer_default_locprec_nm(profile)
    default_lpx_px = infer_default_lpx_px(profile)
    napari_units = str(get_nested(profile, ["downstream", "napari_units"], "nm"))
    locan_units = str(get_nested(profile, ["downstream", "locan_units"], "nm"))

    batches_dir = folders.results / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for index, movie_path in enumerate(movies, start=1):
        batch_id = make_batch_id(movie_path, index)
        batch_out_dir = batches_dir / batch_id
        batch_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(movies)}] {movie_path.name}")

        base_row: Dict[str, Any] = {
            "batch_index": index,
            "batch_id": batch_id,
            "run_id": batch_id,
            "input_path": str(movie_path),
            "input_name": movie_path.name,
            "input_parent": str(movie_path.parent),
            "batch_dir": str(batch_out_dir),
            "run_dir": str(batch_out_dir),
            "profile_path": str(profile_path),
            "backend_name": backend_name,
            "coord_units_requested": coord_units,
            "pixel_size_nm": pixel_size_nm,
            "review_mode": "external_manual",
            "created_at": now_iso(),
            "resolver_status": backend_config.get("status", ""),
            "resolver_function": backend_config.get("resolver_function", ""),
            "psf_type": backend_config.get("psf_type", ""),
            "psf_dimensionality": backend_config.get("psf_dimensionality", ""),
            "calibration_file": backend_config.get("calibration_file", ""),
            "model_path": backend_config.get("model_path", ""),
            "device": backend_config.get("device", ""),
            "batch_size": backend_config.get("batch_size", ""),
            "threshold": backend_config.get("threshold", ""),
        }
        quality_result: Dict[str, Any] = {}
        truth_result: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Input QC
        # ------------------------------------------------------------------
        with bench.stage(
            "input_qc", batch_index=index, input_path=movie_path, out_dir=batch_out_dir
        ):
            qc_result = run_qc_safely(qc_one_movie, movie_path, batch_out_dir)

        bench.benchmark_input_movie(movie_path, batch_index=index)
        qc_status = qc_result.get("qc_status", "unknown")
        print(f"    QC: {qc_status}")

        if qc_status != "passed":
            backend_result = {
                "backend_status": "skipped_qc_failed",
                "backend_name": backend_name,
                "backend_message": "QC failed; backend skipped.",
                "raw_output_path": "",
            }
            canonical_result = {
                "canonical_status": "skipped_qc_failed",
                "canonical_message": "QC failed; post-inference skipped.",
                "canonical_output_path": "",
                "post_inference_summary": "",
                "localization_qc": "",
            }
            export_result = {"status": "skipped_qc_failed"}
            print("    Backend: skipped_qc_failed")
            print("    Post-inference: skipped_qc_failed")

        else:
            # --------------------------------------------------------------
            # Backend inference
            # --------------------------------------------------------------
            with bench.stage(
                "backend_inference",
                batch_index=index,
                input_path=movie_path,
                out_dir=batch_out_dir,
            ):
                backend_result = run_backend_step(
                    step="infer",
                    backend_name=backend_name,
                    input_path=movie_path,
                    out_dir=batch_out_dir,
                    profile=profile,
                    backend_config=backend_config,
                    batch_index=index,
                )

            print(f"    Backend: {backend_result.get('backend_status')}")
            raw_output_path = backend_result.get("raw_output_path", "")

            if raw_output_path:
                # ----------------------------------------------------------
                # Post-inference conversion + exports
                # ----------------------------------------------------------
                with bench.stage(
                    "post_inference",
                    batch_index=index,
                    input_path=raw_output_path,
                    out_dir=batch_out_dir,
                ):
                    post_summary, canonical_result, export_result = (
                        run_post_inference_safely(
                            run_post_inference=run_post_inference,
                            raw_output_path=raw_output_path,
                            movie_out_dir=batch_out_dir,
                            profile=profile,
                            backend_name=backend_name,
                            source_file=movie_path,
                            coord_units=coord_units,
                            pixel_size_nm=pixel_size_nm,
                            default_locprec_nm=default_locprec_nm,
                            default_lpx_px=default_lpx_px,
                            napari_units=napari_units,
                            locan_units=locan_units,
                        )
                    )

                canonical_path = canonical_result.get("canonical_output_path", "")
                if canonical_path:
                    bench.benchmark_localizations(
                        canonical_csv=canonical_path,
                        batch_index=index,
                        coordinate_units=export_result.get(
                            "coord_units_detected", coord_units
                        ),
                        pixel_size_nm=export_result.get("pixel_size_nm", pixel_size_nm),
                    )

                    export_validation_map = build_export_validation_map(
                        canonical_path, export_result
                    )
                    bench.validate_exports(export_validation_map)

                    quality_result = run_quality_metrics_safely(
                        step="infer",
                        bench=bench,
                        folders=folders,
                        profile=profile,
                        batch_index=index,
                        reports_subdir=f"batch_{index:03d}_{safe_stem(movie_path)}",
                        paths={
                            "run_dir": batch_out_dir,
                            "canonical_csv": canonical_path,
                            "input_qc_json": qc_result.get(
                                "qc_json", str(batch_out_dir / "input_qc.json")
                            ),
                        },
                    )

                    truth_csv = infer_truth_csv_for_movie(profile, movie_path, index)
                    if truth_csv is not None:
                        truth_result = bench.benchmark_truth(
                            prediction_csv=canonical_path,
                            truth_csv=truth_csv,
                            batch_index=index,
                            match_radius_xy_nm=truth_match_radius_xy_nm(profile),
                            match_radius_z_nm=truth_match_radius_z_nm(profile),
                        )
            else:
                canonical_result = {
                    "canonical_status": "skipped_no_raw_output",
                    "canonical_message": "No raw backend output available for post-inference.",
                    "canonical_output_path": "",
                    "post_inference_summary": "",
                    "localization_qc": "",
                }
                export_result = {"status": "skipped_no_raw_output"}

            print(f"    Post-inference: {export_result.get('status')}")

        row: Dict[str, Any] = {}
        row.update(base_row)
        row["qc_status"] = qc_result.get("qc_status", "")
        row["qc_json"] = qc_result.get("qc_json", str(batch_out_dir / "input_qc.json"))
        row["qc_preview"] = qc_result.get(
            "preview_png", str(batch_out_dir / "input_preview.png")
        )
        row["qc_histogram"] = qc_result.get(
            "histogram_png", str(batch_out_dir / "input_histogram.png")
        )
        row["shape"] = qc_result.get("shape", "")
        row["axes"] = qc_result.get("axes", "")
        row["dtype"] = qc_result.get("dtype", "")
        row["n_frames_guess"] = qc_result.get("n_frames_guess", "")
        row["frame_guess_confidence"] = qc_result.get("frame_guess_confidence", "")
        row["qc_full_result"] = qc_result
        row.update(backend_result)
        row.update(canonical_result)
        row["quality_metrics_status"] = quality_result.get("status", "")
        row["quality_metrics_result"] = quality_result
        row["truth_benchmark_status"] = truth_result.get("status", "")
        row["truth_benchmark_result"] = truth_result
        row["post_inference_status"] = export_result.get("status", "")
        row["downstream_export_status"] = export_result.get("status", "")
        row["downstream_export_result"] = export_result
        row["review_status"] = "external_manual"
        row["review_result"] = {
            "status": "external_manual",
            "message": (
                "napari/Locan review is not run inside run_pipeline.py. "
                "Use napari_locan_review.py separately in napari_locan_env."
            ),
        }
        rows.append(row)
        print()

    manifest_csv = folders.results / "batch_manifest.csv"
    manifest_json = folders.results / "batch_manifest.json"
    summary_json = folders.results / "run_summary.json"
    write_manifest_csv(rows, manifest_csv)
    write_json(rows, manifest_json)

    benchmark_summary = bench.finalize()
    combined_exports = combine_outputs_safely(folders)

    summary = {
        "created_at": now_iso(),
        "step": "infer",
        "input": str(input_path),
        "run_parent": str(folders.parent),
        "results_dir": str(folders.results),
        "benchmarks_dir": str(folders.benchmarks),
        "reports_dir": str(folders.reports),
        "registry_dir": str(folders.registry),
        "profile_path": str(profile_path),
        "backend_name": backend_name,
        "coord_units_requested": coord_units,
        "pixel_size_nm": pixel_size_nm,
        "review_mode": "external_manual",
        "n_movies": len(rows),
        "qc_passed": sum(row.get("qc_status") == "passed" for row in rows),
        "qc_failed": sum(row.get("qc_status") == "failed" for row in rows),
        "backend_passed": sum(row.get("backend_status") == "passed" for row in rows),
        "backend_failed": sum(row.get("backend_status") == "failed" for row in rows),
        "canonical_passed": sum(
            row.get("canonical_status") == "passed" for row in rows
        ),
        "canonical_failed": sum(
            row.get("canonical_status") == "failed" for row in rows
        ),
        "post_inference_passed": sum(
            row.get("post_inference_status") in {"passed", "warning"} for row in rows
        ),
        "post_inference_failed": sum(
            row.get("post_inference_status") == "failed" for row in rows
        ),
        "quality_passed": sum(
            row.get("quality_metrics_status") == "passed" for row in rows
        ),
        "quality_warning": sum(
            row.get("quality_metrics_status") == "warning" for row in rows
        ),
        "quality_failed": sum(
            row.get("quality_metrics_status") in {"fail", "error"} for row in rows
        ),
        "truth_benchmarks": sum(
            bool(row.get("truth_benchmark_status")) for row in rows
        ),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "summary_json": str(summary_json),
        "benchmark": benchmark_summary,
        "combined_exports": combined_exports,
        "resolved_backend_config": backend_config,
    }

    status = "passed"
    if (
        summary["qc_failed"]
        or summary["backend_failed"]
        or summary["canonical_failed"]
        or summary["post_inference_failed"]
        or summary["quality_failed"]
    ):
        status = "warning"
    summary["status"] = status

    artifact = write_artifact_snapshot(
        step="infer",
        folders=folders,
        profile=profile,
        backend_name=backend_name,
        backend_config=backend_config,
        step_result=summary,
        status=status,
    )
    summary["artifact"] = artifact

    write_json(summary, summary_json)
    write_json(summary, folders.registry / "run_summary.json")

    report = generate_report_safely(folders)
    summary["report"] = report
    write_json(summary, summary_json)
    write_json(summary, folders.registry / "run_summary.json")
    write_run_status(
        folders,
        status=status,
        message="Inference completed.",
        extra={"artifact_id": artifact.get("id")},
    )

    print_footer(folders, summary)
    return summary


# =============================================================================
# Terminal footer
# =============================================================================


def print_footer(folders: RunFolders, summary: Mapping[str, Any]) -> None:
    status = str(summary.get("status", ""))
    print("=" * 70)
    print("Pipeline failed" if status == "failed" else "Pipeline complete")
    print("=" * 70)
    print(f"Status:        {status}")
    print(f"Run folder:    {display_path(folders.parent)}")
    print(f"Results:       {display_path(folders.results)}")
    print(f"Benchmarks:    {display_path(folders.benchmarks)}")
    print(f"Reports:       {display_path(folders.reports)}")
    print(f"Registry:      {display_path(folders.registry)}")

    backend_result = summary.get("backend_result", {})
    if isinstance(backend_result, Mapping) and backend_result.get("backend_status") == "failed":
        message = backend_result.get("backend_message", "")
        log_path = backend_result.get("log_path") or backend_result.get("backend_log_path")
        if message:
            print(f"Backend error: {message}")
        if log_path:
            print(f"Backend log:   {display_path(log_path)}")

    benchmark = summary.get("benchmark", {})
    if isinstance(benchmark, Mapping):
        files = (
            benchmark.get("files", {})
            if isinstance(benchmark.get("files", {}), Mapping)
            else {}
        )
        runtime_csv = (
            files.get("runtime_csv")
            or benchmark.get("benchmark_csv")
            or benchmark.get("runtime", {}).get("runtime_csv", "")
            if isinstance(benchmark.get("runtime", {}), Mapping)
            else ""
        )
        summary_json = files.get("benchmark_summary_json") or ""
        if runtime_csv:
            print(f"Runtime CSV:   {display_path(runtime_csv)}")
        if summary_json:
            print(f"Benchmark JSON:{display_path(summary_json)}")

    report = summary.get("report", {})
    if isinstance(report, Mapping):
        if report.get("html_report"):
            print(f"HTML report:   {display_path(report.get('html_report'))}")
        elif report.get("status") not in {"passed", None}:
            print(f"Report:        {report.get('status')} {report.get('error', '')}")

    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        required=True,
        help="Input TIFF/OME-TIFF file or folder.",
    )
    parser.add_argument(
        "-p",
        required=True,
        help="Profile YAML path.",
    )
    parser.add_argument(
        "-o",
        default=None,
        help="Parent run folder. The pipeline creates results/, benchmarks/, reports/, and registry/ inside it.",
    )
    parser.add_argument(
        "-b",
        default=None,
        help="Optional backend override. Default comes from profile, usually liteloc.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional friendly run name used only when -o is not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without creating folders or running backend stages.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reuse of a non-empty -o folder. Use carefully.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "SMLM LabFlow pipeline. Use one of: calibrate, train, infer. "
            "The public CLI is intentionally small for lab users."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="SMLM LabFlow pipeline 0.2",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{calibrate,train,infer}",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Create/update PSF calibration artifacts from bead/calibration data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(calibrate_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a backend model using the latest compatible calibration from the registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(train_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on raw SMLM movies using the latest compatible model from the registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(infer_parser)
    infer_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional quick test limit for inference.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        summary: Optional[Dict[str, Any]] = None
        if args.command == "calibrate":
            summary = run_calibrate(args)
        elif args.command == "train":
            summary = run_train(args)
        elif args.command == "infer":
            summary = run_infer(args)
        else:
            parser.error(f"Unknown command: {args.command}")

        if isinstance(summary, Mapping) and summary.get("status") == "failed":
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print("\nPipeline failed.")
        print(f"Error: {repr(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
