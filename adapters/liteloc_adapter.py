"""
adapters/liteloc_adapter.py

General LiteLoc adapter for the SMLM wrapper pipeline.

Purpose:
    - resolve LiteLoc backend ports from config/local_paths.yaml via config_resolver.py
    - generate a run-specific LiteLoc inference YAML
    - run LiteLoc inference through either:
        1. script mode:
            calls a LiteLoc inference script with --infer_params_path
        2. module mode:
            imports LiteLoc internals and calls network.multi_process directly
    - produce liteloc_raw_output.csv
    - return that raw CSV path to run_pipeline.py

Important separation:
    config/local_paths.yaml:
        machine-specific LiteLoc root + backend ports

    profile.yaml:
        scientific/runtime configuration:
            - base_infer_yaml
            - model_path
            - PSF/model choice
            - runtime inference parameters
            - output filenames

Assumption:
    You already activated the LiteLoc-compatible environment before running:

        conda activate liteloc_env
        python run_pipeline.py ...

This adapter therefore uses sys.executable implicitly through the current process
environment and does not require a python_executable path in local_paths.yaml.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


# =============================================================================
# Small utilities
# =============================================================================


def get_nested(profile: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely read nested values from the YAML profile.

    Example:
        get_nested(profile, "output", "raw_output_name", default="x.csv")
    """
    current: Any = profile

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path.

    If absolute:
        return absolute resolved path.

    If relative and base_dir is provided:
        resolve relative to base_dir.

    Otherwise:
        resolve relative to current working directory.
    """
    p = Path(path).expanduser()

    if p.is_absolute():
        return p.resolve()

    if base_dir is not None:
        return (base_dir / p).resolve()

    return p.resolve()


def add_liteloc_to_syspath_and_env(repo_dir: Path) -> Dict[str, str]:
    """
    Make LiteLoc importable for subprocesses and optional in-process module mode.

    Returns:
        Environment dictionary suitable for subprocess.run(env=...).
    """
    repo_dir = repo_dir.expanduser().resolve()

    repo_str = str(repo_dir)

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")

    pythonpath_parts = [repo_str]

    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return env


def load_liteloc_backend_config_from_resolver() -> Any:
    """
    Load LiteLoc backend config using config_resolver.py.

    Expected resolver function:
        load_liteloc_backend_config()

    Expected returned object:
        backend_config.root
        backend_config.ports.infer
        backend_config.ports.train
        backend_config.ports.calibrate
    """
    try:
        from config_resolver import load_liteloc_backend_config
    except Exception as exc:
        raise ImportError(
            "Could not import load_liteloc_backend_config from config_resolver.py. "
            "Make sure config_resolver.py exists in the project root and that "
            "config/local_paths.yaml is configured."
        ) from exc

    return load_liteloc_backend_config()


def load_base_liteloc_yaml(base_yaml_path: Path) -> Dict[str, Any]:
    """
    Load a LiteLoc base YAML and ensure minimum expected sections exist.
    """
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Base LiteLoc inference YAML not found: {base_yaml_path}")

    with base_yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError(f"LiteLoc YAML must be a dictionary: {base_yaml_path}")

    config.setdefault("Loc_Model", {})
    config.setdefault("Multi_Process", {})

    return config


def detect_infer_execution(profile: Dict[str, Any], infer_port: Path) -> str:
    """
    Decide whether to run inference in script mode or module mode.

    Profile override:
        liteloc.infer_execution: "script" | "module" | "auto"

    Auto behavior:
        - if infer port is network/multi_process.py, use module mode
        - otherwise use script mode
    """
    value = str(
        get_nested(profile, "liteloc", "infer_execution", default="auto")
    ).lower().strip()

    if value not in {"auto", "script", "module"}:
        raise ValueError(
            "Invalid profile value liteloc.infer_execution. "
            "Expected one of: auto, script, module."
        )

    if value in {"script", "module"}:
        return value

    infer_name = infer_port.name.lower()
    infer_parent = infer_port.parent.name.lower()

    if infer_name == "multi_process.py" or infer_parent == "network":
        return "module"

    return "script"


# =============================================================================
# Runtime YAML generation
# =============================================================================


def build_runtime_liteloc_infer_yaml(
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
    repo_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Build a run-specific LiteLoc inference YAML.

    It starts from a LiteLoc-native base YAML, then overwrites:
        - Loc_Model.model_path
        - Multi_Process.image_path
        - Multi_Process.save_path
        - runtime parameters

    Returns:
        runtime_yaml_path, raw_output_path, resolved_config
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Base LiteLoc inference YAML
    # -------------------------------------------------------------------------
    base_yaml_value = get_nested(
        profile,
        "liteloc",
        "base_infer_yaml",
        default=get_nested(
            profile,
            "liteloc",
            "infer_yaml",
            default="demo/demo1_astig_npc/infer_params_demo1.yaml",
        ),
    )

    base_yaml_path = resolve_path(base_yaml_value, base_dir=repo_dir)

    # -------------------------------------------------------------------------
    # Model checkpoint
    # -------------------------------------------------------------------------
    model_path_value = get_nested(profile, "liteloc", "model_path")

    if model_path_value is None:
        model_path_value = get_nested(profile, "inference", "model_path")

    if model_path_value is None:
        raise KeyError(
            "Missing model path. Add one of these to your profile:\n"
            "  liteloc.model_path: /path/to/checkpoint.pkl\n"
            "or:\n"
            "  inference.model_path: /path/to/checkpoint.pkl"
        )

    model_path = resolve_path(model_path_value, base_dir=repo_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"LiteLoc model checkpoint not found: {model_path}")

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    raw_output_name = get_nested(
        profile,
        "output",
        "raw_output_name",
        default="liteloc_raw_output.csv",
    )

    runtime_yaml_name = get_nested(
        profile,
        "output",
        "runtime_yaml_name",
        default="runtime_liteloc_infer.yaml",
    )

    raw_output_path = out_dir / str(raw_output_name)
    runtime_yaml_path = out_dir / str(runtime_yaml_name)

    # -------------------------------------------------------------------------
    # Patch LiteLoc-native YAML
    # -------------------------------------------------------------------------
    config = load_base_liteloc_yaml(base_yaml_path)

    config["Loc_Model"]["model_path"] = str(model_path)

    config["Multi_Process"]["image_path"] = str(input_path)
    config["Multi_Process"]["save_path"] = str(raw_output_path)

    # Runtime parameters.
    # Priority:
    #   1. liteloc.runtime.*
    #   2. older top-level inference.*
    #   3. conservative defaults
    config["Multi_Process"]["time_block_gb"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "time_block_gb",
        default=get_nested(profile, "inference", "time_block_gb", default=1),
    )

    config["Multi_Process"]["batch_size"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "batch_size",
        default=get_nested(profile, "inference", "batch_size", default=64),
    )

    config["Multi_Process"]["sub_fov_size"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "sub_fov_size",
        default=get_nested(profile, "inference", "sub_fov_size", default=256),
    )

    config["Multi_Process"]["over_cut"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "over_cut",
        default=get_nested(profile, "inference", "over_cut", default=8),
    )

    config["Multi_Process"]["data_queue_size"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "data_queue_size",
        default=get_nested(profile, "inference", "data_queue_size", default=100),
    )

    config["Multi_Process"]["multi_gpu"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "multi_gpu",
        default=get_nested(profile, "inference", "multi_gpu", default=False),
    )

    config["Multi_Process"]["num_producers"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "num_producers",
        default=get_nested(profile, "inference", "num_producers", default=1),
    )

    with runtime_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    resolved_config = {
        "base_yaml_path": str(base_yaml_path),
        "model_path": str(model_path),
        "runtime_yaml_path": str(runtime_yaml_path),
        "raw_output_path": str(raw_output_path),
        "multi_process": config.get("Multi_Process", {}),
    }

    return runtime_yaml_path, raw_output_path, resolved_config


# =============================================================================
# Inference execution modes
# =============================================================================


def run_liteloc_inference_script_mode(
    *,
    infer_port: Path,
    runtime_yaml_path: Path,
    repo_dir: Path,
    env: Dict[str, str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    """
    Run a LiteLoc inference script that accepts:

        --infer_params_path runtime_liteloc_infer.yaml

    This is compatible with LiteLoc demo inference scripts such as:
        demo/demo1_astig_npc/liteloc_infer_demo1.py
    """
    if not infer_port.exists():
        raise FileNotFoundError(f"LiteLoc inference script not found: {infer_port}")

    cmd = [
        sys.executable,
        str(infer_port),
        "--infer_params_path",
        str(runtime_yaml_path),
    ]

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRunning LiteLoc inference in SCRIPT mode.\n")
        log.write("Command:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        result = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return result


def run_liteloc_inference_module_mode(
    *,
    runtime_yaml_path: Path,
    repo_dir: Path,
    log_path: Path,
) -> int:
    """
    Run LiteLoc inference by importing LiteLoc modules directly.

    This mirrors the official inference demo logic:
        - load YAML using utils.help_utils.load_yaml_infer
        - torch.load(model_path)
        - call network.multi_process.CompetitiveSmlmDataAnalyzer_multi_producer
        - start analyzer
    """
    add_liteloc_to_syspath_and_env(repo_dir)

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRunning LiteLoc inference in MODULE mode.\n")
        log.write("Runtime YAML:\n")
        log.write(str(runtime_yaml_path) + "\n\n")
        log.flush()

    try:
        import torch  # type: ignore
        from utils.help_utils import load_yaml_infer  # type: ignore
        from network import multi_process  # type: ignore
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Failed to import LiteLoc modules.\n")
            log.write(repr(exc) + "\n")
        raise

    infer_params = load_yaml_infer(str(runtime_yaml_path))

    # weights_only=False follows the current LiteLoc demo behavior.
    liteloc_model = torch.load(infer_params.Loc_Model.model_path, weights_only=False)

    multi_process_params = infer_params.Multi_Process

    t0 = time.time()

    analyzer = multi_process.CompetitiveSmlmDataAnalyzer_multi_producer(
        loc_model=liteloc_model,
        tiff_path=multi_process_params.image_path,
        output_path=multi_process_params.save_path,
        time_block_gb=multi_process_params.time_block_gb,
        batch_size=multi_process_params.batch_size,
        sub_fov_size=multi_process_params.sub_fov_size,
        over_cut=multi_process_params.over_cut,
        data_queue_size=multi_process_params.data_queue_size,
        multi_GPU=multi_process_params.multi_gpu,
        num_producers=multi_process_params.num_producers,
    )

    t1 = time.time()

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"LiteLoc module init time: {t1 - t0:.3f} seconds\n")
        log.flush()

    analyzer.start()

    t2 = time.time()

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"LiteLoc module analyze time: {t2 - t1:.3f} seconds\n")
        log.flush()

    return 0


# =============================================================================
# Main inference function called by run_pipeline.py
# =============================================================================


def run_liteloc_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    """
    Function called automatically by run_pipeline.py.

    Runs real LiteLoc inference using backend ports resolved from local_paths.yaml.

    Args:
        input_path:
            TIFF file or folder to analyze.

        out_dir:
            Pipeline run output directory.

        profile:
            Loaded YAML profile.

    Returns:
        Path to liteloc_raw_output.csv if produced.

    Raises:
        RuntimeError if LiteLoc execution fails.
        FileNotFoundError if expected output CSV is not created.
    """
    start_time = time.time()

    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_config = load_liteloc_backend_config_from_resolver()

    repo_dir = Path(backend_config.root).expanduser().resolve()
    infer_port = Path(backend_config.ports.infer).expanduser().resolve()

    env = add_liteloc_to_syspath_and_env(repo_dir)

    runtime_yaml_path, raw_output_path, resolved_runtime = build_runtime_liteloc_infer_yaml(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        repo_dir=repo_dir,
    )

    log_name = get_nested(
        profile,
        "output",
        "log_name",
        default="liteloc.log",
    )

    log_path = out_dir / str(log_name)
    status_path = out_dir / "liteloc_adapter_status.json"

    infer_execution = detect_infer_execution(profile, infer_port)

    status: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "liteloc",
        "adapter": "general_liteloc_adapter",
        "variant": get_nested(
            profile,
            "backend",
            "variant",
            default=get_nested(profile, "psf", "family", default="unspecified"),
        ),
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "repo_dir": str(repo_dir),
        "infer_port": str(infer_port),
        "infer_execution": infer_execution,
        "runtime_yaml_path": str(runtime_yaml_path),
        "raw_output_path": str(raw_output_path),
        "log_path": str(log_path),
        "resolved_runtime": resolved_runtime,
        "status": "started",
    }

    write_json(status, status_path)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("LiteLoc adapter reached.\n")
        log.write("Running REAL LiteLoc inference through general adapter.\n\n")
        log.write("LiteLoc root:\n")
        log.write(str(repo_dir) + "\n\n")
        log.write("LiteLoc infer port:\n")
        log.write(str(infer_port) + "\n\n")
        log.write("Inference execution mode:\n")
        log.write(infer_execution + "\n\n")
        log.write("Runtime YAML:\n")
        log.write(str(runtime_yaml_path) + "\n\n")
        log.write("Raw output path:\n")
        log.write(str(raw_output_path) + "\n\n")
        log.write("PYTHONPATH used for LiteLoc:\n")
        log.write(env.get("PYTHONPATH", "") + "\n\n")
        log.write("=" * 80 + "\n")
        log.flush()

    try:
        if infer_execution == "script":
            result = run_liteloc_inference_script_mode(
                infer_port=infer_port,
                runtime_yaml_path=runtime_yaml_path,
                repo_dir=repo_dir,
                env=env,
                log_path=log_path,
            )
            return_code = int(result.returncode)

        elif infer_execution == "module":
            return_code = run_liteloc_inference_module_mode(
                runtime_yaml_path=runtime_yaml_path,
                repo_dir=repo_dir,
                log_path=log_path,
            )

        else:
            raise RuntimeError(f"Unsupported LiteLoc infer execution mode: {infer_execution}")

    except Exception as exc:
        elapsed_seconds = round(time.time() - start_time, 2)

        status.update(
            {
                "status": "failed_exception",
                "return_code": None,
                "elapsed_seconds": elapsed_seconds,
                "message": repr(exc),
            }
        )
        write_json(status, status_path)
        raise

    elapsed_seconds = round(time.time() - start_time, 2)

    status.update(
        {
            "return_code": return_code,
            "elapsed_seconds": elapsed_seconds,
        }
    )

    if return_code != 0:
        status.update(
            {
                "status": "failed",
                "message": "LiteLoc execution failed. Check liteloc.log.",
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
            "status": "real_liteloc_output_created",
            "message": "Real LiteLoc inference completed successfully.",
        }
    )
    write_json(status, status_path)

    return str(raw_output_path)


# =============================================================================
# Optional aliases expected by run_pipeline.py
# =============================================================================


def run_liteloc_inference(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    """
    Alias for clarity.
    """
    return run_liteloc_one_movie(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
    )


def run_inference_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    """
    Alias compatible with run_pipeline.py dynamic adapter discovery.
    """
    return run_liteloc_one_movie(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
    )


def run_liteloc(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    """
    Alias compatible with run_pipeline.py dynamic adapter discovery.
    """
    return run_liteloc_one_movie(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
    )