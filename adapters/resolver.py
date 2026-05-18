#!/usr/bin/env python3
"""
resolver.py

Resolve machine-specific LiteLoc backend ports from config/local_paths.yaml.

Scope:
- LiteLoc repository root
- LiteLoc backend ports:
    - calibrate
    - train
    - infer

No scientific parameters.
No Python executable field.
Assumption:
    The wrapper is launched from the correct environment, e.g.

        conda activate liteloc_env
        python run_pipeline.py ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_CANDIDATES = (
    Path("config/local_paths.yaml"),
    Path.home() / ".liteloc_wrapper" / "local_paths.yaml",
)


class ConfigResolverError(RuntimeError):
    """Raised when LiteLoc backend path resolution fails."""


@dataclass(frozen=True)
class LiteLocPorts:
    calibrate: Path
    train: Path
    infer: Path


@dataclass(frozen=True)
class LiteLocBackendConfig:
    root: Path
    ports: LiteLocPorts


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file safely as a dictionary."""
    if not path.exists():
        raise ConfigResolverError(f"Local config not found: {path}")

    if not path.is_file():
        raise ConfigResolverError(f"Local config path is not a file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise ConfigResolverError(f"Local config is empty: {path}")

    if not isinstance(data, dict):
        raise ConfigResolverError(f"Local config must be a YAML dictionary: {path}")

    return data


def _get_required(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Get a required key and reject null/empty values."""
    value = mapping.get(key)

    if value in (None, ""):
        raise ConfigResolverError(f"Missing required key: {context}.{key}")

    return value


def _as_path(value: Any, label: str) -> Path:
    """Convert a YAML value to a pathlib.Path."""
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigResolverError(
            f"{label} must be a path string, got {type(value).__name__}"
        )

    return Path(value).expanduser()


def _resolve_port(port_value: Any, root: Path, label: str) -> Path:
    """
    Resolve a backend port.

    If port_value is relative, it is interpreted relative to LiteLoc root.
    If port_value is absolute, it is used as-is.
    """
    port_path = _as_path(port_value, label)

    if not port_path.is_absolute():
        port_path = root / port_path

    return port_path.resolve()


def find_local_config(explicit_path: str | Path | None = None) -> Path:
    """
    Find local_paths.yaml.

    Search order:
    1. explicit_path argument
    2. LITELOC_WRAPPER_LOCAL_CONFIG environment variable
    3. ./config/local_paths.yaml
    4. ~/.liteloc_wrapper/local_paths.yaml
    """
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()

        if not path.exists():
            raise ConfigResolverError(f"Explicit local config not found: {path}")

        return path

    env_path = os.environ.get("LITELOC_WRAPPER_LOCAL_CONFIG")
    if env_path:
        path = Path(env_path).expanduser().resolve()

        if not path.exists():
            raise ConfigResolverError(
                "LITELOC_WRAPPER_LOCAL_CONFIG points to a missing file: "
                f"{path}"
            )

        return path

    for candidate in DEFAULT_CONFIG_CANDIDATES:
        path = candidate.expanduser().resolve()

        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in DEFAULT_CONFIG_CANDIDATES)
    raise ConfigResolverError(
        "Could not find local_paths.yaml.\n"
        "Searched:\n"
        f"{searched}\n"
        "Create config/local_paths.yaml or set LITELOC_WRAPPER_LOCAL_CONFIG."
    )


def load_liteloc_backend_config(
    local_config_path: str | Path | None = None,
    *,
    validate_exists: bool = True,
) -> LiteLocBackendConfig:
    """
    Load and resolve LiteLoc backend ports.

    Expected YAML:

        liteloc:
          root: /path/to/LiteLoc
          ports:
            calibrate: demo/demo3_calibrate_psf/demo3_psf_calibration.py
            train: network/loc_model.py
            infer: network/multi_process.py
    """
    config_path = find_local_config(local_config_path)
    data = _load_yaml(config_path)

    liteloc = data.get("liteloc")
    if not isinstance(liteloc, dict):
        raise ConfigResolverError("Missing required section: liteloc")

    root = _as_path(
        _get_required(liteloc, "root", "liteloc"),
        "liteloc.root",
    ).resolve()

    ports_data = liteloc.get("ports")
    if not isinstance(ports_data, dict):
        raise ConfigResolverError("Missing required section: liteloc.ports")

    ports = LiteLocPorts(
        calibrate=_resolve_port(
            _get_required(ports_data, "calibrate", "liteloc.ports"),
            root,
            "liteloc.ports.calibrate",
        ),
        train=_resolve_port(
            _get_required(ports_data, "train", "liteloc.ports"),
            root,
            "liteloc.ports.train",
        ),
        infer=_resolve_port(
            _get_required(ports_data, "infer", "liteloc.ports"),
            root,
            "liteloc.ports.infer",
        ),
    )

    config = LiteLocBackendConfig(root=root, ports=ports)

    if validate_exists:
        validate_liteloc_backend_config(config)

    return config


def validate_liteloc_backend_config(config: LiteLocBackendConfig) -> None:
    """Validate that resolved LiteLoc backend paths exist."""
    if not config.root.exists():
        raise ConfigResolverError(f"LiteLoc root does not exist: {config.root}")

    if not config.root.is_dir():
        raise ConfigResolverError(f"LiteLoc root is not a directory: {config.root}")

    for name, path in (
        ("calibrate", config.ports.calibrate),
        ("train", config.ports.train),
        ("infer", config.ports.infer),
    ):
        if not path.exists():
            raise ConfigResolverError(f"LiteLoc {name} port does not exist: {path}")

        if not path.is_file():
            raise ConfigResolverError(f"LiteLoc {name} port is not a file: {path}")


def print_liteloc_backend_config(config: LiteLocBackendConfig) -> None:
    """Print resolved backend paths for doctor/debug mode."""
    print("LiteLoc backend ports:")
    print(f"  root:      {config.root}")
    print(f"  calibrate: {config.ports.calibrate}")
    print(f"  train:     {config.ports.train}")
    print(f"  infer:     {config.ports.infer}")


if __name__ == "__main__":
    try:
        backend_config = load_liteloc_backend_config()
        print_liteloc_backend_config(backend_config)
        print("\nOK: LiteLoc backend ports resolved successfully.")
    except ConfigResolverError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)