"""
adapters/liteloc_adapter.py

Real LiteLoc adapter for the wrapper pipeline.

Current purpose:
    - generate a run-specific LiteLoc inference YAML
    - call LiteLoc's official inference script
    - produce liteloc_raw_output.csv
    - return that raw CSV path to run_pipeline.py

This replaces the previous mock adapter.

Current backend:
    LiteLoc astigmatic pretrained inference
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import yaml


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path.

    If the path is absolute:
        return it directly.

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


def load_base_liteloc_yaml(base_yaml_path: Path) -> Dict[str, Any]:
    """
    Load the LiteLoc base inference YAML.

    If something is missing, create the minimum expected structure.
    """
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Base LiteLoc inference YAML not found: {base_yaml_path}")

    with open(base_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    config.setdefault("Loc_Model", {})
    config.setdefault("Multi_Process", {})

    return config


def build_runtime_liteloc_yaml(
    input_path: Path,
    out_dir: Path,
    profile: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Build a run-specific LiteLoc inference YAML.

    It starts from LiteLoc's base YAML, then overwrites:
        - model_path
        - image_path
        - save_path
        - batch/runtime parameters

    Returns:
        runtime_yaml_path, raw_output_path
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir_value = get_nested(profile, "liteloc", "repo_dir")
    if repo_dir_value is None:
        raise KeyError("Missing profile field: liteloc.repo_dir")

    repo_dir = resolve_path(repo_dir_value)

    base_yaml_value = get_nested(
        profile,
        "liteloc",
        "base_infer_yaml",
        default="demo/demo1_astig_npc/infer_params_demo1.yaml",
    )
    base_yaml_path = resolve_path(base_yaml_value, base_dir=repo_dir)

    model_path_value = get_nested(profile, "liteloc", "model_path")
    if model_path_value is None:
        raise KeyError("Missing profile field: liteloc.model_path")

    model_path = resolve_path(model_path_value, base_dir=repo_dir)

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

    config = load_base_liteloc_yaml(base_yaml_path)

    # ------------------------------------------------------------
    # Required LiteLoc fields
    # ------------------------------------------------------------
    config["Loc_Model"]["model_path"] = str(model_path)

    config["Multi_Process"]["image_path"] = str(input_path)
    config["Multi_Process"]["save_path"] = str(raw_output_path)

    # ------------------------------------------------------------
    # Runtime parameters
    # Priority:
    #   1. liteloc.runtime.*
    #   2. older top-level inference.*
    #   3. safe defaults
    # ------------------------------------------------------------
    config["Multi_Process"]["time_block_gb"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "time_block_gb",
        default=1,
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
        default=256,
    )

    config["Multi_Process"]["over_cut"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "over_cut",
        default=8,
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
        default=False,
    )

    config["Multi_Process"]["num_producers"] = get_nested(
        profile,
        "liteloc",
        "runtime",
        "num_producers",
        default=1,
    )

    with open(runtime_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return runtime_yaml_path, raw_output_path


def run_liteloc_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    """
    Function called automatically by run_pipeline.py.

    Runs real LiteLoc inference.

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
        RuntimeError if LiteLoc command fails.
        FileNotFoundError if expected output CSV is not created.
    """
    start_time = time.time()

    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir_value = get_nested(profile, "liteloc", "repo_dir")
    if repo_dir_value is None:
        raise KeyError("Missing profile field: liteloc.repo_dir")

    repo_dir = resolve_path(repo_dir_value)

    infer_script_value = get_nested(
        profile,
        "liteloc",
        "infer_script",
        default="demo/demo1_astig_npc/liteloc_infer_demo1.py",
    )

    infer_script = resolve_path(infer_script_value, base_dir=repo_dir)

    if not infer_script.exists():
        raise FileNotFoundError(f"LiteLoc inference script not found: {infer_script}")

    runtime_yaml_path, raw_output_path = build_runtime_liteloc_yaml(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
    )

    log_name = get_nested(
        profile,
        "output",
        "log_name",
        default="liteloc.log",
    )

    log_path = out_dir / str(log_name)
    status_path = out_dir / "liteloc_adapter_status.json"

    cmd = [
        "python",
        str(infer_script),
        "--infer_params_path",
        str(runtime_yaml_path),
    ]

    status = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "liteloc",
        "variant": get_nested(
            profile,
            "backend",
            "variant",
            default="liteloc_astig_pretrained",
        ),
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "repo_dir": str(repo_dir),
        "infer_script": str(infer_script),
        "runtime_yaml_path": str(runtime_yaml_path),
        "raw_output_path": str(raw_output_path),
        "log_path": str(log_path),
        "command": " ".join(cmd),
        "status": "started",
    }

    write_json(status, status_path)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("LiteLoc adapter reached.\n")
        log.write("Running REAL LiteLoc inference.\n\n")
        log.write("Command:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.write("Working directory:\n")
        log.write(str(repo_dir) + "\n\n")
        log.write("Runtime YAML:\n")
        log.write(str(runtime_yaml_path) + "\n\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        env = os.environ.copy()

        existing_pythonpath = env.get("PYTHONPATH", "")

        pythonpath_parts = [
            str(repo_dir),
        ]

        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        log.write("\nPYTHONPATH used for LiteLoc:\n")
        log.write(env["PYTHONPATH"] + "\n\n")
        log.flush()

        result = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed_seconds = round(time.time() - start_time, 2)

    status.update(
        {
            "return_code": result.returncode,
            "elapsed_seconds": elapsed_seconds,
        }
    )

    if result.returncode != 0:
        status.update(
            {
                "status": "failed",
                "message": "LiteLoc subprocess failed. Check liteloc.log.",
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


# Optional alias if later you prefer a clearer function name.
def run_liteloc_inference(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Dict[str, Any],
) -> Optional[str]:
    return run_liteloc_one_movie(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
    )