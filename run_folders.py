#!/usr/bin/env python3
"""
run_folders.py

Helper for creating and managing one parent run folder.

Design:
    parent_run_folder/
    ├── results/
    ├── benchmarks/
    ├── reports/
    ├── registry/
    └── README_RUN.txt

Public meaning:
    -o PATH means: use PATH as the parent run folder.

Example:
    python run_pipeline.py infer \
        -i data/movies \
        -p profiles/dna_paint_standard.yaml \
        -o outputs/npc_condition_A

Creates:
    outputs/npc_condition_A/
    ├── results/
    ├── benchmarks/
    ├── reports/
    ├── registry/
    └── README_RUN.txt
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


VALID_STEPS = {"calibrate", "train", "infer"}
RUN_SUBDIRS = ("results", "benchmarks", "reports", "registry")


@dataclass
class RunFolders:
    """
    Structured paths for one pipeline run.
    """

    step: str
    parent: Path
    results: Path
    benchmarks: Path
    reports: Path
    registry: Path
    readme: Path
    run_manifest: Path
    run_status: Path
    profile_snapshot: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "step": self.step,
            "parent": str(self.parent),
            "results": str(self.results),
            "benchmarks": str(self.benchmarks),
            "reports": str(self.reports),
            "registry": str(self.registry),
            "readme": str(self.readme),
            "run_manifest": str(self.run_manifest),
            "run_status": str(self.run_status),
            "profile_snapshot": str(self.profile_snapshot),
        }


# =============================================================================
# Basic utilities
# =============================================================================


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(text: str) -> str:
    """
    Convert user/profile names into filesystem-safe folder names.
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "run"


def path_is_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and not any(path.iterdir())


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def read_profile_name(profile_path: Path) -> str:
    """
    Prefer profile_name from YAML if PyYAML is installed.
    Fallback: filename stem.
    """
    profile_path = profile_path.expanduser()

    if not profile_path.exists():
        return safe_name(profile_path.stem)

    try:
        import yaml
    except ImportError:
        return safe_name(profile_path.stem)

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}

        if isinstance(profile, dict):
            value = profile.get("profile_name")
            if value:
                return safe_name(str(value))

    except Exception:
        pass

    return safe_name(profile_path.stem)


def choose_numbered_sibling(path: Path) -> Path:
    """
    If path exists and is non-empty, return path_001, path_002, etc.

    Example:
        outputs/run_A      exists and non-empty
        outputs/run_A_001  returned
    """
    path = path.expanduser()
    parent = path.parent
    base = path.name

    match = re.match(r"^(.*?)(?:_(\d{3,}))$", base)

    if match:
        prefix = match.group(1)
        start = int(match.group(2)) + 1
    else:
        prefix = base
        start = 1

    for number in range(start, 10000):
        candidate = parent / f"{prefix}_{number:03d}"

        if not candidate.exists():
            return candidate

        if path_is_empty_dir(candidate):
            return candidate

    raise RuntimeError(f"Could not create numbered run folder near: {path}")


# =============================================================================
# Run folder naming and creation
# =============================================================================


def default_parent_run_folder(
    step: str,
    profile_path: Path,
    outputs_root: Path = Path("outputs"),
    name: Optional[str] = None,
) -> Path:
    """
    Build automatic parent folder when user did not provide -o.

    Example:
        outputs/infer_20260518_170500_dna_paint_standard
        outputs/infer_20260518_170500_npc_condition_A
    """
    step = step.lower().strip()

    if step not in VALID_STEPS:
        raise ValueError(f"Invalid step: {step}. Expected one of {sorted(VALID_STEPS)}")

    label = safe_name(name) if name else read_profile_name(profile_path)

    return outputs_root / f"{step}_{now_stamp()}_{label}"


def resolve_parent_run_folder(
    step: str,
    profile_path: Path,
    output_arg: Optional[Path] = None,
    outputs_root: Path = Path("outputs"),
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Decide the final parent run folder.

    Rules:
        - If -o is given: use it as the parent folder.
        - If -o is absent: create automatic folder under outputs/.
        - If folder exists and is empty: use it.
        - If folder exists and is non-empty:
            - overwrite=True: reuse it.
            - overwrite=False: create numbered sibling.
        - If path exists as a file: fail.
    """
    if output_arg is not None:
        target = Path(output_arg).expanduser()
    else:
        target = default_parent_run_folder(
            step=step,
            profile_path=profile_path,
            outputs_root=outputs_root,
            name=name,
        )

    target = target.resolve()

    if target.exists() and target.is_file():
        raise FileExistsError(f"Output parent path exists as a file: {target}")

    if overwrite:
        return target

    if not target.exists():
        return target

    if path_is_empty_dir(target):
        return target

    return choose_numbered_sibling(target).resolve()


def build_run_folders(parent: Path, step: str) -> RunFolders:
    """
    Return the standard paths inside one parent run folder.
    """
    parent = parent.expanduser().resolve()

    return RunFolders(
        step=step,
        parent=parent,
        results=parent / "results",
        benchmarks=parent / "benchmarks",
        reports=parent / "reports",
        registry=parent / "registry",
        readme=parent / "README_RUN.txt",
        run_manifest=parent / "registry" / "run_manifest.json",
        run_status=parent / "registry" / "run_status.json",
        profile_snapshot=parent / "registry" / "profile_snapshot.yaml",
    )


def create_run_tree(folders: RunFolders) -> None:
    """
    Actually create parent folder and standard child folders.
    """
    folders.parent.mkdir(parents=True, exist_ok=True)

    for subdir in RUN_SUBDIRS:
        (folders.parent / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Metadata, README, snapshots
# =============================================================================


def copy_profile_snapshot(profile_path: Path, folders: RunFolders) -> Optional[Path]:
    """
    Copy the profile used for this run into registry/profile_snapshot.yaml.
    """
    profile_path = profile_path.expanduser()

    if not profile_path.exists():
        return None

    folders.registry.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile_path, folders.profile_snapshot)

    return folders.profile_snapshot


def make_readme_text(
    step: str,
    input_path: Path,
    profile_path: Path,
    folders: RunFolders,
    created_at: str,
) -> str:
    return f"""SMLM LabFlow Run
================

Step
----
{step}

Created at
----------
{created_at}

Input
-----
{input_path}

Profile
-------
{profile_path}

Main folders
------------
results/     Scientific outputs: calibration files, models, canonical CSVs, exports, QC outputs.
benchmarks/  Runtime and performance files.
reports/     Human-readable reports and figures.
registry/    Run metadata, profile snapshot, provenance, and artifact links.

Recommended file to open first
------------------------------
reports/run_report.html

Notes
-----
This folder is self-contained for this run.
You can copy this parent folder to another location and keep the run evidence together.
"""


def write_readme(
    step: str,
    input_path: Path,
    profile_path: Path,
    folders: RunFolders,
    created_at: str,
) -> None:
    folders.readme.write_text(
        make_readme_text(
            step=step,
            input_path=input_path,
            profile_path=profile_path,
            folders=folders,
            created_at=created_at,
        ),
        encoding="utf-8",
    )


def write_run_manifest(
    step: str,
    input_path: Path,
    profile_path: Path,
    folders: RunFolders,
    created_at: str,
    command: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "status": "created",
        "step": step,
        "created_at": created_at,
        "input_path": str(input_path),
        "profile_path": str(profile_path),
        "parent_run_folder": str(folders.parent),
        "folders": folders.as_dict(),
        "command": list(command) if command is not None else list(sys.argv),
    }

    if extra:
        manifest.update(extra)

    write_json(manifest, folders.run_manifest)
    return manifest


def write_run_status(
    folders: RunFolders,
    status: str,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status_data: Dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "parent_run_folder": str(folders.parent),
    }

    if extra:
        status_data.update(extra)

    write_json(status_data, folders.run_status)
    return status_data


# =============================================================================
# Main public helper
# =============================================================================


def prepare_parent_run_folder(
    step: str,
    input_path: Path,
    profile_path: Path,
    output_arg: Optional[Path] = None,
    outputs_root: Path = Path("outputs"),
    name: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    command: Optional[Sequence[str]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> RunFolders:
    """
    Main function to use from run_pipeline.py.

    Parameters
    ----------
    step:
        One of: calibrate, train, infer.

    input_path:
        User input file/folder.

    profile_path:
        Profile YAML.

    output_arg:
        Value from -o. If None, automatic folder is created under outputs/.

    outputs_root:
        Default root used only when -o is not provided.

    name:
        Optional human-friendly label used only for automatic folder names.

    overwrite:
        If False, non-empty folders are protected by automatic numbering.

    dry_run:
        If True, resolve folder paths but do not create anything.

    command:
        Usually sys.argv.

    extra_manifest:
        Optional additional metadata.

    Returns
    -------
    RunFolders
        Structured paths for this run.
    """
    step = step.lower().strip()

    if step not in VALID_STEPS:
        raise ValueError(f"Invalid step: {step}. Expected one of {sorted(VALID_STEPS)}")

    input_path = Path(input_path).expanduser()
    profile_path = Path(profile_path).expanduser()

    parent = resolve_parent_run_folder(
        step=step,
        profile_path=profile_path,
        output_arg=output_arg,
        outputs_root=outputs_root,
        name=name,
        overwrite=overwrite,
    )

    folders = build_run_folders(parent=parent, step=step)

    if dry_run:
        return folders

    created_at = datetime.now().isoformat(timespec="seconds")

    create_run_tree(folders)
    copy_profile_snapshot(profile_path=profile_path, folders=folders)

    write_readme(
        step=step,
        input_path=input_path,
        profile_path=profile_path,
        folders=folders,
        created_at=created_at,
    )

    write_run_manifest(
        step=step,
        input_path=input_path,
        profile_path=profile_path,
        folders=folders,
        created_at=created_at,
        command=command,
        extra=extra_manifest,
    )

    write_run_status(
        folders=folders,
        status="created",
        message="Parent run folder created successfully.",
    )

    return folders


# =============================================================================
# Existing run validation
# =============================================================================


def validate_parent_run_folder(parent: Path) -> Dict[str, Any]:
    """
    Check whether a parent run folder has the expected structure.
    """
    parent = Path(parent).expanduser().resolve()

    expected = {
        "results": parent / "results",
        "benchmarks": parent / "benchmarks",
        "reports": parent / "reports",
        "registry": parent / "registry",
        "readme": parent / "README_RUN.txt",
    }

    checks = {name: path.exists() for name, path in expected.items()}

    return {
        "parent": str(parent),
        "exists": parent.exists(),
        "is_dir": parent.is_dir(),
        "checks": checks,
        "valid": parent.exists() and parent.is_dir() and all(checks.values()),
    }


def load_existing_run_folders(parent: Path, step: str = "infer") -> RunFolders:
    """
    Rebuild RunFolders object for an already-created run folder.
    """
    parent = Path(parent).expanduser().resolve()
    validation = validate_parent_run_folder(parent)

    if not validation["valid"]:
        raise FileNotFoundError(
            f"Invalid or incomplete run folder: {parent}\n" f"Validation: {validation}"
        )

    return build_run_folders(parent=parent, step=step)
