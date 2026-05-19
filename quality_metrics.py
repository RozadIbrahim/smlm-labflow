#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_metrics.py

Automatic scientific/data quality control for an SMLM/LiteLoc wrapper pipeline.

Purpose
-------
This module is deliberately separate from benchmark.py.
- benchmark.py = runtime/resource timing
- quality_metrics.py = scientific/data quality diagnostics

It can run automatically after:
- calibrate
- train
- infer

Primary outputs
---------------
<reports_dir>/quality_metrics.json
<reports_dir>/quality_metrics.md
<reports_dir>/quality_summary.csv
<reports_dir>/quality_flags.csv
<reports_dir>/quality_assets/*.png

Typical integration from run_pipeline.py
----------------------------------------
from quality_metrics import run_quality_after_infer

quality = run_quality_after_infer(
    paths={
        "run_dir": run_dir,
        "canonical_csv": canonical_csv,
        "input_qc_json": input_qc_json,
        "out_dir": run_dir / "reports",
    },
    profile=profile,
)

CLI examples
------------
python quality_metrics.py infer \
  --canonical results/run_001/results/canonical_localizations.csv \
  --input-qc results/run_001/results/input_qc.json \
  --out results/run_001/reports

python quality_metrics.py train \
  --run-dir results/train_001 \
  --checkpoint results/train_001/results/checkpoint.pkl \
  --out results/train_001/reports

python quality_metrics.py calibrate \
  --run-dir results/calib_001 \
  --calibration-file results/calib_001/results/psf_calibration.mat \
  --out results/calib_001/reports

Design notes
------------
- Safe with missing optional dependencies: scipy/tifffile are optional.
- Safe with imperfect CSVs: missing columns produce flags rather than hard crashes.
- Compatible with Python 3.9+.
- Avoids the common bug: the historical unhashable-list missing-value check because lists are unhashable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Matplotlib is used in non-interactive mode because this script may run on a lab server.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None

try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "1.0.0"

REQUIRED_INFER_COLUMNS = ["frame", "x", "y"]
OPTIONAL_LOCALIZATION_COLUMNS = [
    "z",
    "photons",
    "background",
    "confidence",
    "sigma_x",
    "sigma_y",
    "crlb_x",
    "crlb_y",
    "crlb_z",
    "rmse_x",
    "rmse_y",
    "rmse_z",
    "backend",
]

COLUMN_ALIASES = {
    "frame": ["frame", "frames", "t", "time", "frame_id", "frame_idx", "frame_index"],
    "x": ["x", "x_nm", "xnm", "x [nm]", "x_pix", "x_pixel", "x_px", "xrec", "x_pred"],
    "y": ["y", "y_nm", "ynm", "y [nm]", "y_pix", "y_pixel", "y_px", "yrec", "y_pred"],
    "z": ["z", "z_nm", "znm", "z [nm]", "zrec", "z_pred"],
    "photons": ["photons", "photon", "nphotons", "intensity", "signal", "n", "phot"],
    "background": ["background", "bg", "bkg", "offset", "baseline"],
    "confidence": ["confidence", "prob", "probability", "score", "p", "likelihood"],
    "sigma_x": ["sigma_x", "sigmax", "sx"],
    "sigma_y": ["sigma_y", "sigmay", "sy"],
    "crlb_x": ["crlb_x", "x_crlb", "crlbx"],
    "crlb_y": ["crlb_y", "y_crlb", "crlby"],
    "crlb_z": ["crlb_z", "z_crlb", "crlbz"],
    "rmse_x": ["rmse_x", "x_rmse", "rmsex"],
    "rmse_y": ["rmse_y", "y_rmse", "rmsey"],
    "rmse_z": ["rmse_z", "z_rmse", "rmsez"],
    "backend": ["backend", "method", "model", "software"],
}

DEFAULT_THRESHOLDS = {
    # General
    "max_nan_fraction_key_columns": 0.001,
    "max_inf_fraction_key_columns": 0.0,
    "min_localizations": 1,
    # Infer/localization specific
    "duplicate_exact_warn_fraction": 0.001,
    "close_pair_radius_xy": 30.0,  # Units follow canonical x/y. Usually nm if your canonical schema is nm.
    "close_pair_warn_fraction": 0.05,
    "max_negative_photon_fraction": 0.0,
    "max_negative_background_fraction": 0.05,
    "confidence_min": 0.0,
    "confidence_max": 1.0,
    "grid_fft_warn_score": 20.0,
    "frame_count_cv_warn": 2.5,
    "center_drift_warn_fraction_of_fov": 0.15,
    # Plots
    "density_bins": 256,
    "max_points_for_close_pair_qc": 250000,
    "max_points_per_frame_for_close_pair_qc": 5000,
}

SEVERITY_ORDER = {"pass": 0, "info": 1, "warning": 2, "fail": 3, "error": 4}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_missing_value(value: Any) -> bool:
    """Robust missing-value test that does not put lists/dicts inside a set."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        # pandas/numpy scalar NaN handling.
        return bool(pd.isna(value))
    except Exception:
        return False


def safe_float(value: Any) -> Optional[float]:
    try:
        if is_missing_value(value):
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if is_missing_value(value):
            return None
        return int(value)
    except Exception:
        return None


def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas/path objects into JSON-safe Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [to_builtin(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(data: Mapping[str, Any], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_builtin(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_builtin(row.get(k, "")) for k in keys})
    return p


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, new_key))
        elif isinstance(value, list):
            # Keep compact summaries readable. Full list remains in JSON.
            if len(value) <= 8 and all(not isinstance(v, (dict, list, tuple, set)) for v in value):
                out[new_key] = ";".join(str(v) for v in value)
            else:
                out[new_key] = f"<list:{len(value)}>"
        else:
            out[new_key] = value
    return out


def normalize_col_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("µ", "u")
    text = re.sub(r"\s+", " ", text)
    return text


def find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    normalized_to_original = {normalize_col_name(c): c for c in df.columns}
    for alias in aliases:
        norm = normalize_col_name(alias)
        if norm in normalized_to_original:
            return normalized_to_original[norm]
    # Last-resort fuzzy match after removing separators.
    compressed = {re.sub(r"[^a-z0-9]", "", normalize_col_name(c)): c for c in df.columns}
    for alias in aliases:
        norm = re.sub(r"[^a-z0-9]", "", normalize_col_name(alias))
        if norm in compressed:
            return compressed[norm]
    return None


def canonicalize_localization_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    rename: Dict[str, str] = {}
    for canonical_name in list(COLUMN_ALIASES.keys()):
        original = find_column(df, canonical_name)
        if original is not None:
            mapping[canonical_name] = original
            if original != canonical_name and canonical_name not in df.columns:
                rename[original] = canonical_name
    out = df.rename(columns=rename).copy()
    return out, mapping


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def robust_quantiles(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return {
            "count": int(s.shape[0]),
            "finite_count": 0,
            "nan_count": int(s.isna().sum()),
            "min": None,
            "q01": None,
            "q05": None,
            "median": None,
            "q95": None,
            "q99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    q = finite.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "count": int(s.shape[0]),
        "finite_count": int(finite.shape[0]),
        "nan_count": int(s.isna().sum()),
        "min": float(finite.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "median": float(q.loc[0.5]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.shape[0] > 1 else 0.0,
    }


def get_nested(mapping: Optional[Mapping[str, Any]], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping or {}
    for part in path:
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def merge_thresholds(profile: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    quality_cfg = get_nested(profile, ["quality"], {}) if profile else {}
    if isinstance(quality_cfg, Mapping):
        user_thresholds = quality_cfg.get("thresholds", {})
        if isinstance(user_thresholds, Mapping):
            thresholds.update(user_thresholds)
        # Also allow profile.quality.close_pair_radius_xy directly.
        for key in DEFAULT_THRESHOLDS:
            if key in quality_cfg:
                thresholds[key] = quality_cfg[key]
    return thresholds


def best_existing_path(paths: Mapping[str, Any], keys: Sequence[str]) -> Optional[Path]:
    for key in keys:
        p = as_path(paths.get(key))
        if p is not None and p.exists():
            return p
    for key in keys:
        p = as_path(paths.get(key))
        if p is not None:
            return p
    return None


def infer_out_dir(paths: Mapping[str, Any], explicit_out_dir: Optional[Union[str, Path]] = None) -> Path:
    if explicit_out_dir is not None:
        return ensure_dir(explicit_out_dir)
    for key in ["out_dir", "reports_dir", "quality_dir"]:
        p = as_path(paths.get(key))
        if p is not None:
            return ensure_dir(p)
    run_dir = as_path(paths.get("run_dir"))
    if run_dir is not None:
        return ensure_dir(run_dir / "reports")
    return ensure_dir("reports")


def add_flag(
    flags: List[Dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    metric: Optional[str] = None,
    value: Any = None,
    threshold: Any = None,
    recommendation: Optional[str] = None,
) -> None:
    flags.append(
        {
            "severity": severity,
            "code": code,
            "metric": metric,
            "value": to_builtin(value),
            "threshold": to_builtin(threshold),
            "message": message,
            "recommendation": recommendation,
        }
    )


def overall_status(flags: Sequence[Mapping[str, Any]]) -> str:
    if not flags:
        return "passed"
    max_sev = "pass"
    for flag in flags:
        sev = str(flag.get("severity", "info"))
        if SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(max_sev, 0):
            max_sev = sev
    if max_sev == "error":
        return "error"
    if max_sev == "fail":
        return "fail"
    if max_sev == "warning":
        return "warning"
    return "passed"


def file_inventory(root: Optional[Path], max_files: int = 300) -> List[Dict[str, Any]]:
    if root is None or not root.exists():
        return []
    if root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*") if p.is_file()]
    rows = []
    for p in sorted(files)[:max_files]:
        try:
            stat = p.stat()
            rows.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "suffix": p.suffix.lower(),
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
                }
            )
        except Exception:
            rows.append({"path": str(p), "name": p.name, "suffix": p.suffix.lower(), "size_bytes": None})
    return rows


def find_files_by_patterns(root: Optional[Path], patterns: Sequence[str], max_files: int = 50) -> List[Path]:
    if root is None or not root.exists():
        return []
    if root.is_file():
        candidates = [root]
    else:
        candidates = [p for p in root.rglob("*") if p.is_file()]
    out: List[Path] = []
    for p in candidates:
        text = str(p).lower()
        if any(re.search(pattern, text) for pattern in patterns):
            out.append(p)
    return sorted(out)[:max_files]


def save_simple_line_plot(series: pd.Series, path: Path, title: str, ylabel: str) -> Optional[Path]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 4.5))
        plt.plot(series.index, series.values, linewidth=1)
        plt.title(title)
        plt.xlabel("index")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
    except Exception:
        plt.close("all")
        return None


def save_histogram(series: pd.Series, path: Path, title: str, xlabel: str, bins: int = 80) -> Optional[Path]:
    try:
        values = pd.to_numeric(series, errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.hist(values.values, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
    except Exception:
        plt.close("all")
        return None


# -----------------------------------------------------------------------------
# Infer/localization QC
# -----------------------------------------------------------------------------

def load_localization_csv(path: Path) -> pd.DataFrame:
    # Pandas auto-detects headers poorly when files are malformed; explicit default is fine for canonical CSV.
    return pd.read_csv(path)


def schema_quality(df: pd.DataFrame, flags: List[Dict[str, Any]]) -> Dict[str, Any]:
    present = list(df.columns)
    missing_required = [c for c in REQUIRED_INFER_COLUMNS if c not in df.columns]
    present_optional = [c for c in OPTIONAL_LOCALIZATION_COLUMNS if c in df.columns]

    if missing_required:
        add_flag(
            flags,
            "fail",
            "missing_required_columns",
            f"Canonical localization CSV is missing required columns: {missing_required}",
            metric="schema.missing_required",
            value=missing_required,
            threshold=REQUIRED_INFER_COLUMNS,
            recommendation="Fix post_inference.py/schema.py mapping before trusting downstream exports.",
        )
    else:
        add_flag(flags, "info", "required_columns_present", "Required canonical columns are present.")

    return {
        "present_columns": present,
        "required_columns": REQUIRED_INFER_COLUMNS,
        "missing_required_columns": missing_required,
        "present_optional_columns": present_optional,
        "n_columns": int(len(present)),
    }


def numeric_column_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    numeric_candidates = [
        c
        for c in [
            "frame",
            "x",
            "y",
            "z",
            "photons",
            "background",
            "confidence",
            "sigma_x",
            "sigma_y",
            "crlb_x",
            "crlb_y",
            "crlb_z",
            "rmse_x",
            "rmse_y",
            "rmse_z",
        ]
        if c in df.columns
    ]
    metrics: Dict[str, Any] = {}
    n = max(int(len(df)), 1)
    for col in numeric_candidates:
        s = pd.to_numeric(df[col], errors="coerce")
        finite = np.isfinite(s)
        nan_fraction = float(s.isna().sum() / n)
        inf_fraction = float((~finite & ~s.isna()).sum() / n)
        metrics[col] = robust_quantiles(s)
        metrics[col]["nan_fraction"] = nan_fraction
        metrics[col]["inf_fraction"] = inf_fraction

        if col in ["frame", "x", "y"]:
            max_nan = float(thresholds.get("max_nan_fraction_key_columns", 0.001))
            max_inf = float(thresholds.get("max_inf_fraction_key_columns", 0.0))
            if nan_fraction > max_nan:
                add_flag(
                    flags,
                    "fail",
                    f"{col}_nan_fraction_high",
                    f"Key column '{col}' has too many NaN values.",
                    metric=f"numeric.{col}.nan_fraction",
                    value=nan_fraction,
                    threshold=max_nan,
                )
            if inf_fraction > max_inf:
                add_flag(
                    flags,
                    "fail",
                    f"{col}_inf_fraction_high",
                    f"Key column '{col}' has infinite values.",
                    metric=f"numeric.{col}.inf_fraction",
                    value=inf_fraction,
                    threshold=max_inf,
                )

    return metrics


def physical_sanity_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    n = max(int(len(df)), 1)

    if "photons" in df.columns:
        photons = pd.to_numeric(df["photons"], errors="coerce")
        frac = float((photons < 0).sum() / n)
        metrics["negative_photon_fraction"] = frac
        limit = float(thresholds.get("max_negative_photon_fraction", 0.0))
        if frac > limit:
            add_flag(
                flags,
                "warning",
                "negative_photons_detected",
                "Some localizations have negative photon counts.",
                metric="sanity.negative_photon_fraction",
                value=frac,
                threshold=limit,
                recommendation="Check canonical column mapping; photon/intensity column may be incorrectly mapped.",
            )

    if "background" in df.columns:
        bg = pd.to_numeric(df["background"], errors="coerce")
        frac = float((bg < 0).sum() / n)
        metrics["negative_background_fraction"] = frac
        limit = float(thresholds.get("max_negative_background_fraction", 0.05))
        if frac > limit:
            add_flag(
                flags,
                "warning",
                "negative_background_detected",
                "A notable fraction of localizations have negative background values.",
                metric="sanity.negative_background_fraction",
                value=frac,
                threshold=limit,
            )

    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        lo = float(thresholds.get("confidence_min", 0.0))
        hi = float(thresholds.get("confidence_max", 1.0))
        out_frac = float(((conf < lo) | (conf > hi)).sum() / n)
        metrics["confidence_outside_expected_range_fraction"] = out_frac
        if out_frac > 0:
            add_flag(
                flags,
                "warning",
                "confidence_outside_expected_range",
                "Confidence/probability values fall outside the configured expected range.",
                metric="sanity.confidence_outside_expected_range_fraction",
                value=out_frac,
                threshold=[lo, hi],
                recommendation="If this column is not a probability, rename it or adjust profile.quality.thresholds.confidence_min/max.",
            )

    for axis in ["x", "y", "z"]:
        if axis in df.columns:
            values = pd.to_numeric(df[axis], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.shape[0] > 2 and float(finite.max() - finite.min()) == 0.0:
                # x/y constant is usually a broken conversion. z can legitimately be constant
                # in 2D SMLM or astigmatic runs where z was not estimated for this export.
                severity = "warning" if axis == "z" else "fail"
                add_flag(
                    flags,
                    severity,
                    f"{axis}_constant",
                    f"Column '{axis}' is constant across all finite localizations.",
                    metric=f"sanity.{axis}_range",
                    value=0.0,
                    threshold="> 0",
                    recommendation="For x/y, check backend output mapping. For z, this may be normal for 2D exports.",
                )

    return metrics


def frame_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
    input_qc: Optional[Mapping[str, Any]] = None,
    assets_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if "frame" not in df.columns:
        return {"available": False}

    frame = pd.to_numeric(df["frame"], errors="coerce")
    valid = df.loc[np.isfinite(frame)].copy()
    valid["frame"] = pd.to_numeric(valid["frame"], errors="coerce").astype(int)
    if valid.empty:
        add_flag(flags, "fail", "no_valid_frames", "No valid frame values were found.")
        return {"available": True, "valid_frame_count": 0}

    counts = valid.groupby("frame").size().sort_index()
    mean_count = float(counts.mean())
    std_count = float(counts.std(ddof=1)) if counts.shape[0] > 1 else 0.0
    cv = float(std_count / mean_count) if mean_count > 0 else None

    metrics: Dict[str, Any] = {
        "available": True,
        "min_frame": int(counts.index.min()),
        "max_frame": int(counts.index.max()),
        "n_frames_with_localizations": int(counts.shape[0]),
        "localizations_per_frame_mean": mean_count,
        "localizations_per_frame_median": float(counts.median()),
        "localizations_per_frame_std": std_count,
        "localizations_per_frame_cv": cv,
        "localizations_per_frame_min": int(counts.min()),
        "localizations_per_frame_max": int(counts.max()),
    }

    if counts.index.min() < 0:
        add_flag(
            flags,
            "warning",
            "negative_frame_index",
            "Negative frame indices were detected.",
            metric="frame.min_frame",
            value=int(counts.index.min()),
            threshold=">= 0",
        )

    cv_limit = float(thresholds.get("frame_count_cv_warn", 2.5))
    if cv is not None and cv > cv_limit:
        add_flag(
            flags,
            "warning",
            "high_frame_count_variability",
            "Localization counts vary strongly across frames.",
            metric="frame.localizations_per_frame_cv",
            value=cv,
            threshold=cv_limit,
            recommendation="Inspect photobleaching, drift, thresholding, and possible movie segmentation issues.",
        )

    # Optional estimate of expected number of frames from input_qc.json.
    estimated_frames = estimate_frame_count_from_input_qc(input_qc or {})
    if estimated_frames is not None:
        zero_frames = max(0, int(estimated_frames) - int(counts.shape[0]))
        metrics["estimated_movie_frames_from_input_qc"] = int(estimated_frames)
        metrics["estimated_zero_localization_frames"] = int(zero_frames)
        metrics["estimated_zero_localization_frame_fraction"] = float(zero_frames / max(estimated_frames, 1))

    if assets_dir is not None:
        p = save_simple_line_plot(
            counts,
            assets_dir / "quality_localizations_per_frame.png",
            "Localizations per frame",
            "localizations",
        )
        if p is not None:
            metrics["plot_localizations_per_frame"] = str(p)

    return metrics


def estimate_frame_count_from_input_qc(input_qc: Mapping[str, Any]) -> Optional[int]:
    if not input_qc:
        return None
    # Accept several likely formats from qc_input.py.
    for key in ["n_frames", "frames", "frame_count"]:
        val = safe_int(input_qc.get(key))
        if val is not None and val > 0:
            return val
    shape = input_qc.get("shape") or input_qc.get("image_shape") or input_qc.get("array_shape")
    axes = input_qc.get("axes") or input_qc.get("axes_guess")
    if isinstance(shape, str):
        nums = re.findall(r"\d+", shape)
        shape = [int(x) for x in nums]
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        shape_list = [safe_int(x) for x in shape]
        shape_list = [x for x in shape_list if x is not None]
        if not shape_list:
            return None
        if isinstance(axes, str) and "T" in axes.upper() and len(axes) == len(shape_list):
            idx = axes.upper().index("T")
            return int(shape_list[idx])
        # SMLM movies are usually TYX or CYX/TYX. If 3D, first axis is often frames.
        if len(shape_list) >= 3:
            return int(shape_list[0])
    return None


def duplicate_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    n = max(int(len(df)), 1)
    exact_cols = [c for c in ["frame", "x", "y", "z"] if c in df.columns]
    if len(exact_cols) >= 3:
        exact_dups = int(df.duplicated(subset=exact_cols, keep=False).sum())
        frac = float(exact_dups / n)
        metrics["exact_duplicate_rows_subset"] = exact_cols
        metrics["exact_duplicate_localizations"] = exact_dups
        metrics["exact_duplicate_fraction"] = frac
        limit = float(thresholds.get("duplicate_exact_warn_fraction", 0.001))
        if frac > limit:
            add_flag(
                flags,
                "warning",
                "exact_duplicate_localizations",
                "Exact duplicate localizations were detected.",
                metric="duplicates.exact_duplicate_fraction",
                value=frac,
                threshold=limit,
                recommendation="Check whether post-inference concatenation duplicated batch outputs.",
            )

    if cKDTree is None:
        metrics["close_pair_qc_available"] = False
        metrics["close_pair_qc_reason"] = "scipy.spatial.cKDTree unavailable"
        return metrics

    if not all(c in df.columns for c in ["frame", "x", "y"]):
        metrics["close_pair_qc_available"] = False
        metrics["close_pair_qc_reason"] = "requires frame, x, y"
        return metrics

    radius = float(thresholds.get("close_pair_radius_xy", 30.0))
    max_total = int(thresholds.get("max_points_for_close_pair_qc", 250000))
    max_per_frame = int(thresholds.get("max_points_per_frame_for_close_pair_qc", 5000))

    work = df[["frame", "x", "y"]].copy()
    for c in ["frame", "x", "y"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        metrics["close_pair_qc_available"] = False
        metrics["close_pair_qc_reason"] = "no finite frame/x/y rows"
        return metrics

    if work.shape[0] > max_total:
        work = work.sample(n=max_total, random_state=42)
        metrics["close_pair_qc_sampled"] = True
        metrics["close_pair_qc_sample_size"] = int(max_total)
    else:
        metrics["close_pair_qc_sampled"] = False
        metrics["close_pair_qc_sample_size"] = int(work.shape[0])

    close_pairs = 0
    frames_checked = 0
    points_checked = 0
    for _, g in work.groupby("frame"):
        if g.shape[0] < 2:
            continue
        if g.shape[0] > max_per_frame:
            g = g.sample(n=max_per_frame, random_state=42)
        pts = g[["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=radius, output_type="set")
        close_pairs += int(len(pairs))
        frames_checked += 1
        points_checked += int(pts.shape[0])

    pair_fraction_per_point = float(close_pairs / max(points_checked, 1))
    metrics.update(
        {
            "close_pair_qc_available": True,
            "close_pair_radius_xy": radius,
            "close_pairs_same_frame": int(close_pairs),
            "close_pair_fraction_per_checked_point": pair_fraction_per_point,
            "close_pair_frames_checked": int(frames_checked),
            "close_pair_points_checked": int(points_checked),
        }
    )

    limit = float(thresholds.get("close_pair_warn_fraction", 0.05))
    if pair_fraction_per_point > limit:
        add_flag(
            flags,
            "warning",
            "high_same_frame_close_pair_fraction",
            "Many same-frame localizations are closer than the configured radius.",
            metric="duplicates.close_pair_fraction_per_checked_point",
            value=pair_fraction_per_point,
            threshold=limit,
            recommendation="Check duplicate detections, thresholding, and the coordinate unit used by the canonical schema.",
        )

    return metrics


def density_and_fft_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
    assets_dir: Path,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"available": False}
    if not all(c in df.columns for c in ["x", "y"]):
        return metrics

    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)
    if x.size < 2:
        return metrics

    bins = int(thresholds.get("density_bins", 256))
    bins = max(32, min(bins, 512))

    try:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        metrics["available"] = True
        metrics["density_bins"] = bins
        metrics["fov_x_min"] = float(np.min(x))
        metrics["fov_x_max"] = float(np.max(x))
        metrics["fov_y_min"] = float(np.min(y))
        metrics["fov_y_max"] = float(np.max(y))
        metrics["fov_x_span"] = float(np.max(x) - np.min(x))
        metrics["fov_y_span"] = float(np.max(y) - np.min(y))
        metrics["density_nonzero_bins_fraction"] = float(np.count_nonzero(H) / H.size)
        metrics["density_max_bin_count"] = float(np.max(H))
        metrics["density_mean_bin_count"] = float(np.mean(H))

        assets_dir.mkdir(parents=True, exist_ok=True)
        density_png = assets_dir / "quality_localization_density_log.png"
        plt.figure(figsize=(6, 5.5))
        plt.imshow(np.log1p(H.T), origin="lower", aspect="auto")
        plt.title("Localization density, log1p(count)")
        plt.xlabel("x bin")
        plt.ylabel("y bin")
        plt.colorbar(label="log1p(count)")
        plt.tight_layout()
        plt.savefig(density_png, dpi=180)
        plt.close()
        metrics["plot_density_log"] = str(density_png)

        H0 = H - np.mean(H)
        F = np.abs(np.fft.fftshift(np.fft.fft2(H0)))
        center_x = F.shape[0] // 2
        center_y = F.shape[1] // 2
        F[max(0, center_x - 2) : center_x + 3, max(0, center_y - 2) : center_y + 3] = 0
        positive = F[F > 0]
        if positive.size:
            median_power = float(np.median(positive))
            q99_power = float(np.quantile(positive, 0.99))
            max_power = float(np.max(positive))
            grid_score = float(max_power / median_power) if median_power > 0 else None
        else:
            median_power = 0.0
            q99_power = 0.0
            max_power = 0.0
            grid_score = None

        fft_png = assets_dir / "quality_density_fft_log.png"
        plt.figure(figsize=(6, 5.5))
        plt.imshow(np.log1p(F.T), origin="lower", aspect="auto")
        plt.title("FFT of localization density, log1p(power)")
        plt.xlabel("frequency x")
        plt.ylabel("frequency y")
        plt.colorbar(label="log1p(power)")
        plt.tight_layout()
        plt.savefig(fft_png, dpi=180)
        plt.close()

        metrics.update(
            {
                "plot_density_fft_log": str(fft_png),
                "fft_median_power": median_power,
                "fft_q99_power": q99_power,
                "fft_max_power": max_power,
                "fft_grid_artifact_score": grid_score,
            }
        )
        limit = float(thresholds.get("grid_fft_warn_score", 20.0))
        if grid_score is not None and grid_score > limit:
            add_flag(
                flags,
                "warning",
                "possible_grid_artifact",
                "FFT of the localization density has a strong non-central peak.",
                metric="density_fft.fft_grid_artifact_score",
                value=grid_score,
                threshold=limit,
                recommendation="Inspect density/FFT plots for camera/grid/tiling artifacts or batch stitching artifacts.",
            )
    except Exception as exc:
        add_flag(
            flags,
            "warning",
            "density_fft_qc_failed",
            f"Density/FFT QC failed: {exc}",
            recommendation="This does not necessarily invalidate the run; inspect the canonical CSV and plotting environment.",
        )
    finally:
        plt.close("all")

    return metrics


def drift_proxy_quality(
    df: pd.DataFrame,
    flags: List[Dict[str, Any]],
    thresholds: Mapping[str, Any],
    assets_dir: Path,
) -> Dict[str, Any]:
    if not all(c in df.columns for c in ["frame", "x", "y"]):
        return {"available": False}

    work = df[["frame", "x", "y"]].copy()
    for c in ["frame", "x", "y"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty or work["frame"].nunique() < 3:
        return {"available": False, "reason": "not enough valid frames"}

    grouped = work.groupby(work["frame"].astype(int))[['x', 'y']].median().sort_index()
    dx = grouped["x"] - grouped["x"].iloc[0]
    dy = grouped["y"] - grouped["y"].iloc[0]
    displacement = np.sqrt(dx**2 + dy**2)

    fov_x = float(work["x"].max() - work["x"].min())
    fov_y = float(work["y"].max() - work["y"].min())
    fov_diag = math.sqrt(fov_x * fov_x + fov_y * fov_y) if fov_x > 0 and fov_y > 0 else None
    max_disp = float(displacement.max())
    frac_fov = float(max_disp / fov_diag) if fov_diag and fov_diag > 0 else None

    metrics: Dict[str, Any] = {
        "available": True,
        "method": "per-frame median center displacement proxy; not a replacement for fiducial drift correction",
        "max_median_center_displacement": max_disp,
        "final_median_center_displacement": float(displacement.iloc[-1]),
        "max_median_center_displacement_fraction_of_fov_diagonal": frac_fov,
    }

    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        p = assets_dir / "quality_drift_proxy_median_center.png"
        plt.figure(figsize=(8, 4.5))
        plt.plot(grouped.index, displacement.values, linewidth=1)
        plt.title("Drift proxy: median center displacement")
        plt.xlabel("frame")
        plt.ylabel("displacement in canonical x/y units")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        metrics["plot_drift_proxy"] = str(p)
    except Exception:
        plt.close("all")

    limit = float(thresholds.get("center_drift_warn_fraction_of_fov", 0.15))
    if frac_fov is not None and frac_fov > limit:
        add_flag(
            flags,
            "warning",
            "large_median_center_shift",
            "The per-frame median localization center shifts strongly relative to the field of view.",
            metric="drift_proxy.max_median_center_displacement_fraction_of_fov_diagonal",
            value=frac_fov,
            threshold=limit,
            recommendation="Inspect drift correction, stage stability, photobleaching pattern, and sample motion.",
        )

    return metrics


def plot_numeric_distributions(df: pd.DataFrame, assets_dir: Path) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    for col in ["x", "y", "z", "photons", "background", "confidence", "crlb_x", "crlb_y", "crlb_z"]:
        if col in df.columns:
            p = save_histogram(df[col], assets_dir / f"quality_hist_{col}.png", f"Distribution of {col}", col)
            if p is not None:
                plots[f"hist_{col}"] = str(p)
    return plots


def analyze_localizations(
    canonical_csv: Path,
    out_dir: Path,
    profile: Optional[Mapping[str, Any]] = None,
    input_qc: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flags: List[Dict[str, Any]] = []
    thresholds = merge_thresholds(profile)
    assets_dir = ensure_dir(out_dir / "quality_assets")

    metrics: Dict[str, Any] = {
        "canonical_csv": str(canonical_csv),
        "thresholds_used": thresholds,
        "assets_dir": str(assets_dir),
    }

    if not canonical_csv.exists():
        add_flag(
            flags,
            "fail",
            "canonical_csv_missing",
            "Canonical localization CSV does not exist.",
            metric="files.canonical_csv_exists",
            value=False,
            threshold=True,
        )
        return metrics, flags

    if canonical_csv.stat().st_size == 0:
        add_flag(flags, "fail", "canonical_csv_empty_file", "Canonical localization CSV exists but is empty.")
        return metrics, flags

    try:
        raw_df = load_localization_csv(canonical_csv)
    except Exception as exc:
        add_flag(flags, "fail", "canonical_csv_read_failed", f"Could not read canonical CSV: {exc}")
        metrics["read_error_traceback"] = traceback.format_exc(limit=5)
        return metrics, flags

    df, alias_mapping = canonicalize_localization_columns(raw_df)
    numeric_cols = list(set(REQUIRED_INFER_COLUMNS + OPTIONAL_LOCALIZATION_COLUMNS) & set(df.columns))
    df = coerce_numeric(df, [c for c in numeric_cols if c != "backend"])

    n_locs = int(len(df))
    metrics["n_localizations"] = n_locs
    metrics["n_rows"] = n_locs
    metrics["input_columns"] = list(raw_df.columns)
    metrics["canonicalized_column_mapping"] = alias_mapping

    min_locs = int(thresholds.get("min_localizations", 1))
    if n_locs < min_locs:
        add_flag(
            flags,
            "fail",
            "too_few_localizations",
            "The canonical CSV contains too few localizations.",
            metric="localizations.n_localizations",
            value=n_locs,
            threshold=min_locs,
            recommendation="Check LiteLoc inference output path, threshold, model checkpoint, and post_inference conversion.",
        )
        return metrics, flags

    metrics["schema"] = schema_quality(df, flags)
    metrics["numeric"] = numeric_column_quality(df, flags, thresholds)
    metrics["sanity"] = physical_sanity_quality(df, flags, thresholds)
    metrics["frame"] = frame_quality(df, flags, thresholds, input_qc=input_qc, assets_dir=assets_dir)
    metrics["duplicates"] = duplicate_quality(df, flags, thresholds)
    metrics["density_fft"] = density_and_fft_quality(df, flags, thresholds, assets_dir)
    metrics["drift_proxy"] = drift_proxy_quality(df, flags, thresholds, assets_dir)
    metrics["plots"] = plot_numeric_distributions(df, assets_dir)

    # CRLB/RMSE summary if present.
    precision_cols = [c for c in ["crlb_x", "crlb_y", "crlb_z", "rmse_x", "rmse_y", "rmse_z"] if c in df.columns]
    metrics["precision_columns_present"] = precision_cols
    if precision_cols:
        metrics["precision_summary"] = {c: robust_quantiles(df[c]) for c in precision_cols}
    else:
        add_flag(
            flags,
            "info",
            "no_crlb_rmse_columns",
            "No CRLB/RMSE columns were found in the canonical CSV; precision metrics are skipped.",
            recommendation="Add these columns during conversion if LiteLoc exports them or if simulation ground truth is available.",
        )

    return metrics, flags


# -----------------------------------------------------------------------------
# Training QC
# -----------------------------------------------------------------------------

def checkpoint_quality(
    checkpoint: Optional[Path],
    run_dir: Optional[Path],
    flags: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"checkpoint_provided": checkpoint is not None}
    candidates: List[Path] = []
    if checkpoint is not None:
        candidates.append(checkpoint)
    candidates.extend(
        find_files_by_patterns(
            run_dir,
            patterns=[r"checkpoint", r"\.pkl$", r"\.pt$", r"\.pth$", r"model", r"weights"],
            max_files=20,
        )
    )
    # Deduplicate while preserving order.
    seen = set()
    unique_candidates = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    metrics["checkpoint_candidates"] = [str(p) for p in unique_candidates]
    existing = [p for p in unique_candidates if p.exists()]
    metrics["n_existing_checkpoint_candidates"] = int(len(existing))

    if not existing:
        add_flag(
            flags,
            "warning",
            "no_checkpoint_found",
            "No model checkpoint/weights file was found.",
            recommendation="Confirm that liteloc_adapter.py records the produced checkpoint path in the run registry.",
        )
        return metrics

    main = existing[0]
    stat = main.stat()
    metrics.update(
        {
            "selected_checkpoint": str(main),
            "selected_checkpoint_suffix": main.suffix.lower(),
            "selected_checkpoint_size_bytes": int(stat.st_size),
            "selected_checkpoint_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    if stat.st_size < 1024:
        add_flag(
            flags,
            "warning",
            "checkpoint_suspiciously_small",
            "Selected checkpoint file is suspiciously small.",
            metric="checkpoint.size_bytes",
            value=int(stat.st_size),
            threshold=">= 1024",
        )
    else:
        add_flag(flags, "info", "checkpoint_found", "A checkpoint/weights candidate was found.")
    return metrics


def training_log_quality(run_dir: Optional[Path], flags: List[Dict[str, Any]], assets_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"available": False}
    candidates = find_files_by_patterns(
        run_dir,
        patterns=[r"loss.*\.csv$", r"train.*\.csv$", r"metrics.*\.csv$", r"history.*\.csv$", r"log.*\.csv$"],
        max_files=20,
    )
    metrics["candidate_training_csvs"] = [str(p) for p in candidates]
    if not candidates:
        add_flag(
            flags,
            "info",
            "no_training_metric_csv_found",
            "No training loss/metric CSV was found; training-curve QC is skipped.",
        )
        return metrics

    selected = candidates[0]
    metrics["available"] = True
    metrics["selected_training_csv"] = str(selected)
    try:
        df = pd.read_csv(selected)
    except Exception as exc:
        add_flag(flags, "warning", "training_metric_csv_read_failed", f"Could not read training metric CSV: {exc}")
        return metrics

    metrics["n_rows"] = int(len(df))
    metrics["columns"] = list(df.columns)
    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(2, int(0.5 * len(df))):
            numeric_cols.append(c)
    metrics["numeric_metric_columns"] = numeric_cols
    metrics["numeric_metric_summary"] = {c: robust_quantiles(df[c]) for c in numeric_cols}

    for col in numeric_cols[:8]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.shape[0] < 2:
            continue
        p = save_simple_line_plot(s.reset_index(drop=True), assets_dir / f"quality_training_{col}.png", f"Training metric: {col}", col)
        if p is not None:
            metrics.setdefault("plots", {})[col] = str(p)

        # Very conservative warning: final value much worse than initial for loss-like columns.
        if "loss" in col.lower() and s.iloc[-1] > s.iloc[0] * 1.25:
            add_flag(
                flags,
                "warning",
                "loss_increased",
                f"Loss-like metric '{col}' ended substantially higher than it started.",
                metric=f"training.{col}.final_vs_initial",
                value=float(s.iloc[-1] / s.iloc[0]) if s.iloc[0] != 0 else None,
                threshold="<= 1.25",
                recommendation="Inspect training logs, learning rate, model/data pairing, and PSF configuration.",
            )
    return metrics


def analyze_training(
    run_dir: Optional[Path],
    checkpoint: Optional[Path],
    out_dir: Path,
    profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flags: List[Dict[str, Any]] = []
    assets_dir = ensure_dir(out_dir / "quality_assets")
    metrics: Dict[str, Any] = {
        "run_dir": str(run_dir) if run_dir else None,
        "assets_dir": str(assets_dir),
        "profile_backend": get_nested(profile, ["backend", "name"], None) if profile else None,
    }
    metrics["checkpoint"] = checkpoint_quality(checkpoint, run_dir, flags)
    metrics["training_logs"] = training_log_quality(run_dir, flags, assets_dir)
    metrics["file_inventory"] = file_inventory(run_dir, max_files=200)
    return metrics, flags


# -----------------------------------------------------------------------------
# Calibration/PSF QC
# -----------------------------------------------------------------------------

def load_array_for_diagnostics(path: Path) -> Optional[np.ndarray]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            return np.asarray(np.load(path, allow_pickle=False))
        if suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            # Pick the first array-like object.
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.size > 0:
                    return arr
        if suffix in [".tif", ".tiff", ".ome.tif", ".ome.tiff"] and tifffile is not None:
            return np.asarray(tifffile.imread(path))
        # Do not parse .mat here to avoid adding scipy.io hard dependency and version quirks.
        return None
    except Exception:
        return None


def array_diagnostics(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr)
    finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.number) else np.array([])
    metrics: Dict[str, Any] = {
        "shape": list(a.shape),
        "ndim": int(a.ndim),
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if finite.size:
        metrics.update(
            {
                "finite_fraction": float(finite.size / max(a.size, 1)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "median": float(np.median(finite)),
                "dynamic_range": float(np.max(finite) - np.min(finite)),
            }
        )
    return metrics


def save_array_preview(arr: np.ndarray, path: Path, title: str) -> Optional[Path]:
    try:
        a = np.asarray(arr)
        if a.size == 0 or not np.issubdtype(a.dtype, np.number):
            return None
        # Reduce to 2D central slice/projection.
        while a.ndim > 2:
            idx = a.shape[0] // 2
            a = a[idx]
        if a.ndim != 2:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5.5, 5))
        plt.imshow(a, origin="lower", aspect="auto")
        plt.title(title)
        plt.colorbar(label="intensity")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path
    except Exception:
        plt.close("all")
        return None


def calibration_quality(
    calibration_file: Optional[Path],
    run_dir: Optional[Path],
    out_dir: Path,
    flags: List[Dict[str, Any]],
) -> Dict[str, Any]:
    assets_dir = ensure_dir(out_dir / "quality_assets")
    metrics: Dict[str, Any] = {"calibration_file_provided": calibration_file is not None}
    candidates: List[Path] = []
    if calibration_file is not None:
        candidates.append(calibration_file)
    candidates.extend(
        find_files_by_patterns(
            run_dir,
            patterns=[r"psf", r"calib", r"spline", r"bead", r"\.mat$", r"\.npy$", r"\.npz$", r"\.tif$", r"\.tiff$"],
            max_files=40,
        )
    )
    seen = set()
    unique_candidates = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    metrics["calibration_candidates"] = [str(p) for p in unique_candidates]
    existing = [p for p in unique_candidates if p.exists()]
    metrics["n_existing_calibration_candidates"] = int(len(existing))

    if not existing:
        add_flag(
            flags,
            "warning",
            "no_calibration_artifact_found",
            "No calibration/PSF artifact was found.",
            recommendation="Confirm that the calibration stage writes its output path into the registry.",
        )
        return metrics

    selected = existing[0]
    stat = selected.stat()
    metrics.update(
        {
            "selected_calibration_file": str(selected),
            "selected_calibration_suffix": selected.suffix.lower(),
            "selected_calibration_size_bytes": int(stat.st_size),
            "selected_calibration_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    add_flag(flags, "info", "calibration_artifact_found", "A calibration/PSF artifact candidate was found.")

    arr = load_array_for_diagnostics(selected)
    if arr is None:
        metrics["array_diagnostics_available"] = False
        metrics["array_diagnostics_reason"] = "Unsupported format or optional reader unavailable. File existence/size was still checked."
        return metrics

    diag = array_diagnostics(arr)
    metrics["array_diagnostics_available"] = True
    metrics["array"] = diag
    p = save_array_preview(arr, assets_dir / "quality_calibration_array_preview.png", "Calibration/PSF array preview")
    if p is not None:
        metrics["plot_calibration_array_preview"] = str(p)

    if diag.get("finite_fraction", 1.0) < 1.0:
        add_flag(
            flags,
            "warning",
            "calibration_array_nonfinite_values",
            "Calibration/PSF array contains non-finite values.",
            metric="calibration.array.finite_fraction",
            value=diag.get("finite_fraction"),
            threshold=1.0,
        )
    if diag.get("dynamic_range") == 0.0:
        add_flag(
            flags,
            "fail",
            "calibration_array_zero_dynamic_range",
            "Calibration/PSF array has zero dynamic range.",
            metric="calibration.array.dynamic_range",
            value=0.0,
            threshold="> 0",
        )

    return metrics


def analyze_calibration(
    run_dir: Optional[Path],
    calibration_file: Optional[Path],
    out_dir: Path,
    profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flags: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "run_dir": str(run_dir) if run_dir else None,
        "profile_psf": get_nested(profile, ["psf"], {}) if profile else {},
    }
    metrics["calibration"] = calibration_quality(calibration_file, run_dir, out_dir, flags)
    metrics["file_inventory"] = file_inventory(run_dir, max_files=200)
    return metrics, flags


# -----------------------------------------------------------------------------
# Report writing
# -----------------------------------------------------------------------------

def markdown_escape(value: Any) -> str:
    text = str(to_builtin(value))
    return text.replace("|", "\\|").replace("\n", " ")


def compact_value(value: Any, max_len: int = 140) -> str:
    value = to_builtin(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def flags_to_markdown(flags: Sequence[Mapping[str, Any]]) -> str:
    if not flags:
        return "No flags.\n"
    lines = ["| Severity | Code | Metric | Value | Threshold | Message |", "|---|---|---|---:|---:|---|"]
    for flag in flags:
        lines.append(
            "| {severity} | {code} | {metric} | {value} | {threshold} | {message} |".format(
                severity=markdown_escape(flag.get("severity", "")),
                code=markdown_escape(flag.get("code", "")),
                metric=markdown_escape(flag.get("metric", "")),
                value=markdown_escape(compact_value(flag.get("value", ""))),
                threshold=markdown_escape(compact_value(flag.get("threshold", ""))),
                message=markdown_escape(flag.get("message", "")),
            )
        )
    return "\n".join(lines) + "\n"


def selected_summary_rows(payload: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    rows: List[Tuple[str, Any]] = []
    for key in [
        "step",
        "status",
        "generated_at_utc",
        "script_version",
    ]:
        rows.append((key, payload.get(key)))
    for path in [
        ("n_localizations", ["localizations", "n_localizations"]),
        ("schema_missing_required", ["localizations", "schema", "missing_required_columns"]),
        ("fft_grid_artifact_score", ["localizations", "density_fft", "fft_grid_artifact_score"]),
        ("close_pair_fraction", ["localizations", "duplicates", "close_pair_fraction_per_checked_point"]),
        ("frame_count_cv", ["localizations", "frame", "localizations_per_frame_cv"]),
        ("checkpoint_size_bytes", ["training", "checkpoint", "selected_checkpoint_size_bytes"]),
        ("calibration_file", ["calibration", "calibration", "selected_calibration_file"]),
        ("calibration_dynamic_range", ["calibration", "calibration", "array", "dynamic_range"]),
    ]:
        rows.append((path[0], get_nested(metrics, path[1], None)))
    return [(k, v) for k, v in rows if not is_missing_value(v)]


def make_quality_markdown(*args: Any) -> str:
    """Render Markdown. Accepts make_quality_markdown(payload) and legacy make_quality_markdown(step, payload)."""
    if len(args) == 1:
        payload = args[0]
    elif len(args) == 2:
        _step, payload = args
        if isinstance(payload, Mapping) and "step" not in payload:
            payload = {**payload, "step": _step}
    else:
        raise TypeError("make_quality_markdown expects payload or (step, payload)")
    if not isinstance(payload, Mapping):
        payload = {
            "step": "unknown",
            "status": "error",
            "generated_at_utc": utc_now_iso(),
            "script_version": SCRIPT_VERSION,
            "metrics": {},
            "flags": [{"severity": "error", "code": "bad_payload", "message": "Markdown payload was not a mapping."}],
            "output_paths": {},
        }
    lines: List[str] = []
    lines.append("# Quality metrics report")
    lines.append("")
    lines.append(f"- **Step:** `{payload.get('step', 'unknown')}`")
    lines.append(f"- **Status:** `{payload.get('status', 'unknown')}`")
    lines.append(f"- **Generated UTC:** `{payload.get('generated_at_utc', '')}`")
    lines.append(f"- **Script version:** `{payload.get('script_version', SCRIPT_VERSION)}`")
    lines.append("")

    lines.append("## Quick summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in selected_summary_rows(payload):
        lines.append(f"| {markdown_escape(key)} | {markdown_escape(compact_value(value))} |")
    lines.append("")

    lines.append("## Flags")
    lines.append("")
    lines.append(flags_to_markdown(payload.get("flags", [])))
    lines.append("")

    output_paths = payload.get("output_paths", {})
    if isinstance(output_paths, Mapping) and output_paths:
        lines.append("## Output files")
        lines.append("")
        for key, value in output_paths.items():
            if not is_missing_value(value):
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    assets = []
    flat = flatten_dict(payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {})
    for key, value in flat.items():
        if isinstance(value, str) and value.lower().endswith(".png"):
            assets.append((key, value))
    if assets:
        lines.append("## Figure assets")
        lines.append("")
        for key, value in assets:
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    lines.append("## Full metric index")
    lines.append("")
    lines.append("Full nested metrics are stored in `quality_metrics.json`. A flattened table is stored in `quality_summary.csv`.")
    lines.append("")
    return "\n".join(lines)


def finalize_quality_payload(step: str, out_dir: Path, metrics: Mapping[str, Any], flags: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    status = overall_status(flags)
    payload: Dict[str, Any] = {
        "step": step,
        "status": status,
        "generated_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "metrics": to_builtin(metrics),
        "flags": to_builtin(list(flags)),
        "output_paths": {},
    }

    out_dir = ensure_dir(out_dir)
    json_path = out_dir / "quality_metrics.json"
    md_path = out_dir / "quality_metrics.md"
    summary_csv_path = out_dir / "quality_summary.csv"
    flags_csv_path = out_dir / "quality_flags.csv"

    # Write JSON first without output_paths, then update once all outputs are known.
    write_json(payload, json_path)
    md_path.write_text(make_quality_markdown(payload), encoding="utf-8")

    flat = flatten_dict(payload["metrics"] if isinstance(payload.get("metrics"), Mapping) else {})
    summary_rows = [{"metric": k, "value": compact_value(v, max_len=500)} for k, v in sorted(flat.items())]
    write_csv_rows(summary_rows, summary_csv_path)
    write_csv_rows(list(flags), flags_csv_path)

    payload["output_paths"] = {
        "quality_metrics_json": str(json_path),
        "quality_metrics_md": str(md_path),
        "quality_summary_csv": str(summary_csv_path),
        "quality_flags_csv": str(flags_csv_path),
    }

    # Rewrite final JSON/MD with output paths included.
    write_json(payload, json_path)
    md_path.write_text(make_quality_markdown(payload), encoding="utf-8")
    return payload


# -----------------------------------------------------------------------------
# Public pipeline API
# -----------------------------------------------------------------------------

def _coerce_paths_mapping(
    paths: Optional[Mapping[str, Any]] = None,
    *,
    run_parent: Any = None,
    run_dir: Any = None,
    results_dir: Any = None,
    benchmarks_dir: Any = None,
    reports_dir: Any = None,
    registry_dir: Any = None,
    step_result: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    folders: Any = None,
    out_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Accept both the newer paths={...} API and the older run_parent/results_dir API."""
    out: Dict[str, Any] = {}
    if isinstance(paths, Mapping):
        out.update(dict(paths))

    if folders is not None:
        for attr, key in [
            ("parent", "run_dir"),
            ("results", "results_dir"),
            ("benchmarks", "benchmarks_dir"),
            ("reports", "reports_dir"),
            ("registry", "registry_dir"),
        ]:
            value = getattr(folders, attr, None)
            if value is not None:
                out.setdefault(key, value)

    if run_parent is not None:
        parent = Path(run_parent)
        out.setdefault("run_dir", parent)
        out.setdefault("results_dir", parent / "results")
        out.setdefault("benchmarks_dir", parent / "benchmarks")
        out.setdefault("reports_dir", parent / "reports")
        out.setdefault("registry_dir", parent / "registry")
    if run_dir is not None:
        out.setdefault("run_dir", run_dir)
    if results_dir is not None:
        out.setdefault("results_dir", results_dir)
    if benchmarks_dir is not None:
        out.setdefault("benchmarks_dir", benchmarks_dir)
    if reports_dir is not None:
        out.setdefault("reports_dir", reports_dir)
    if registry_dir is not None:
        out.setdefault("registry_dir", registry_dir)
    if out_dir is not None:
        out.setdefault("out_dir", out_dir)
        out.setdefault("reports_dir", out_dir)

    merged_step: Dict[str, Any] = {}
    if isinstance(summary, Mapping):
        merged_step.update(dict(summary))
    if isinstance(step_result, Mapping):
        merged_step.update(dict(step_result))
    for key, value in merged_step.items():
        if value is not None:
            out.setdefault(str(key), value)

    for key in [
        "canonical_csv", "canonical_output", "canonical_localizations",
        "canonical_localizations_csv", "post_inference_csv", "input_qc_json",
        "qc_json", "checkpoint", "checkpoint_path", "model_path",
        "model_checkpoint", "weights", "calibration_file", "psf_file",
        "psf_path", "spline_file", "calibration_output", "bead_calibration",
    ]:
        value = kwargs.get(key)
        if value is not None:
            out.setdefault(key, value)

    results = as_path(out.get("results_dir"))
    if results is None:
        rd = as_path(out.get("run_dir"))
        if rd is not None:
            candidate = rd / "results"
            if candidate.exists():
                results = candidate
                out.setdefault("results_dir", candidate)

    if results is not None and results.exists():
        if not any(k in out for k in ["canonical_csv", "canonical_output", "canonical_localizations", "canonical_localizations_csv", "post_inference_csv"]):
            candidates = list(results.rglob("canonical_localizations.csv"))
            if candidates:
                out["canonical_csv"] = candidates[0]
        if not any(k in out for k in ["checkpoint", "checkpoint_path", "model_path", "model_checkpoint", "weights"]):
            candidates: List[Path] = []
            for pattern in ["checkpoint.pkl", "*.pkl", "*.pt", "*.pth"]:
                candidates.extend(results.rglob(pattern))
            if candidates:
                out["checkpoint"] = candidates[0]
        if not any(k in out for k in ["calibration_file", "psf_file", "psf_path", "spline_file", "calibration_output", "bead_calibration"]):
            candidates: List[Path] = []
            for pattern in ["*calib*", "*psf*", "*.mat", "*.npy", "*.npz", "*.yaml", "*.yml"]:
                candidates.extend(results.rglob(pattern))
            if candidates:
                out["calibration_file"] = candidates[0]

    return out


def _legacy_output_dirs(paths: Mapping[str, Any], reports_dir: Path) -> Tuple[Optional[Path], Path]:
    benchmarks_dir = as_path(paths.get("benchmarks_dir"))
    if benchmarks_dir is None:
        run_dir = as_path(paths.get("run_dir"))
        if run_dir is not None:
            benchmarks_dir = run_dir / "benchmarks"
    if benchmarks_dir is not None:
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return benchmarks_dir, reports_dir


def _write_legacy_quality_outputs(step: str, payload: Mapping[str, Any], paths: Mapping[str, Any], reports_dir: Path) -> None:
    """Also write the older project filenames expected by earlier run_pipeline/check scripts."""
    benchmarks_dir, legacy_reports_dir = _legacy_output_dirs(paths, reports_dir)

    if benchmarks_dir is not None:
        write_json(payload, benchmarks_dir / f"quality_metrics_after_{step}.json")
        metrics = payload.get("metrics", {})
        if isinstance(metrics, Mapping):
            flat = flatten_dict(metrics)
            rows: List[Mapping[str, Any]] = [{"metric": k, "value": compact_value(v, max_len=500)} for k, v in sorted(flat.items())]
        else:
            rows = [{"metric": "metrics", "value": compact_value(metrics, max_len=500)}]
        write_csv_rows(rows, benchmarks_dir / f"quality_metrics_after_{step}.csv")

    legacy_reports_dir.mkdir(parents=True, exist_ok=True)
    (legacy_reports_dir / f"quality_metrics_after_{step}.md").write_text(
        make_quality_markdown(payload),
        encoding="utf-8",
    )


def run_quality_after_infer(
    paths: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run automatic QC after inference/post-inference.

    Expected paths keys, any subset accepted:
    - run_dir
    - canonical_csv / canonical_output / canonical_localizations
    - input_qc_json / qc_json
    - out_dir / reports_dir
    """
    paths = _coerce_paths_mapping(paths, out_dir=out_dir, **kwargs)
    reports_dir = infer_out_dir(paths, out_dir)
    canonical_csv = best_existing_path(
        paths,
        ["canonical_csv", "canonical_output", "canonical_localizations", "canonical_localizations_csv", "post_inference_csv"],
    )
    input_qc_path = best_existing_path(paths, ["input_qc_json", "qc_json", "input_qc"])

    input_qc: Optional[Dict[str, Any]] = None
    if input_qc_path is not None and input_qc_path.exists():
        try:
            input_qc = read_json(input_qc_path)
        except Exception:
            input_qc = {"read_error": f"Could not read {input_qc_path}"}

    flags: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "paths_received": {k: str(v) for k, v in paths.items()},
        "input_qc_json": str(input_qc_path) if input_qc_path else None,
        "input_qc": input_qc or {},
    }

    if canonical_csv is None:
        add_flag(
            flags,
            "fail",
            "canonical_csv_path_not_provided",
            "No canonical CSV path was provided to run_quality_after_infer.",
            recommendation="Pass paths['canonical_csv'] from post_inference.py to quality_metrics.py.",
        )
    else:
        loc_metrics, loc_flags = analyze_localizations(canonical_csv, reports_dir, profile=profile, input_qc=input_qc)
        metrics["localizations"] = loc_metrics
        flags.extend(loc_flags)

    payload = finalize_quality_payload("infer", reports_dir, metrics, flags)
    _write_legacy_quality_outputs("infer", payload, paths, reports_dir)
    return payload


def run_quality_after_train(
    paths: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run automatic QC after training."""
    paths = _coerce_paths_mapping(paths, out_dir=out_dir, **kwargs)
    reports_dir = infer_out_dir(paths, out_dir)
    run_dir = as_path(paths.get("run_dir"))
    checkpoint = best_existing_path(paths, ["checkpoint", "checkpoint_path", "model_path", "model_checkpoint", "weights"])
    metrics, flags = analyze_training(run_dir=run_dir, checkpoint=checkpoint, out_dir=reports_dir, profile=profile)
    metrics["paths_received"] = {k: str(v) for k, v in paths.items()}
    payload = finalize_quality_payload("train", reports_dir, metrics, flags)
    _write_legacy_quality_outputs("train", payload, paths, reports_dir)
    return payload


def run_quality_after_calibrate(
    paths: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run automatic QC after PSF/calibration."""
    paths = _coerce_paths_mapping(paths, out_dir=out_dir, **kwargs)
    reports_dir = infer_out_dir(paths, out_dir)
    run_dir = as_path(paths.get("run_dir"))
    calibration_file = best_existing_path(
        paths,
        ["calibration_file", "psf_file", "psf_path", "spline_file", "calibration_output", "bead_calibration"],
    )
    metrics, flags = analyze_calibration(run_dir=run_dir, calibration_file=calibration_file, out_dir=reports_dir, profile=profile)
    metrics["paths_received"] = {k: str(v) for k, v in paths.items()}
    payload = finalize_quality_payload("calibrate", reports_dir, metrics, flags)
    _write_legacy_quality_outputs("calibrate", payload, paths, reports_dir)
    return payload


def run_quality(
    step: str,
    paths: Optional[Mapping[str, Any]] = None,
    profile: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    step = step.lower().strip()
    if step in ["infer", "inference", "post_inference"]:
        return run_quality_after_infer(paths, profile=profile, out_dir=out_dir, **kwargs)
    if step in ["train", "training"]:
        return run_quality_after_train(paths, profile=profile, out_dir=out_dir, **kwargs)
    if step in ["calibrate", "calibration", "psf"]:
        return run_quality_after_calibrate(paths, profile=profile, out_dir=out_dir, **kwargs)
    raise ValueError(f"Unknown quality step: {step!r}. Expected infer, train, or calibrate.")


# Backward-compatible aliases in case your orchestrator imports these names.
quality_after_infer = run_quality_after_infer
quality_after_train = run_quality_after_train
quality_after_calibrate = run_quality_after_calibrate

run_quality_metrics = run_quality
run_after_infer = run_quality_after_infer
run_after_train = run_quality_after_train
run_after_calibrate = run_quality_after_calibrate
after_infer = run_quality_after_infer
after_train = run_quality_after_train
after_calibrate = run_quality_after_calibrate
run_infer_quality_metrics = run_quality_after_infer
run_train_quality_metrics = run_quality_after_train
run_calibrate_quality_metrics = run_quality_after_calibrate


# -----------------------------------------------------------------------------
# Optional YAML profile loading
# -----------------------------------------------------------------------------

def load_profile(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        return read_json(p)
    if suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load YAML profiles. Install pyyaml or pass JSON.") from exc
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    raise ValueError(f"Unsupported profile format: {p.suffix}. Use .yaml, .yml, or .json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic scientific QC for SMLM/LiteLoc pipeline outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("step", choices=["infer", "train", "calibrate"], help="Pipeline stage to QC.")
    parser.add_argument("--run-dir", default=None, help="Run directory to scan for artifacts.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", default=None, help="Reports output directory.")
    parser.add_argument("--profile", default=None, help="Optional YAML/JSON profile with quality thresholds.")

    # Infer inputs
    parser.add_argument("--canonical", "--canonical-csv", dest="canonical_csv", default=None, help="Canonical localization CSV.")
    parser.add_argument("--input-qc", "--input-qc-json", dest="input_qc_json", default=None, help="input_qc.json produced by qc_input.py.")

    # Train inputs
    parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint", default=None, help="Model checkpoint/weights path.")

    # Calibration inputs
    parser.add_argument("--calibration-file", "--psf-file", dest="calibration_file", default=None, help="Calibration/PSF artifact path.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)
    profile = load_profile(args.profile) if args.profile else None

    paths: Dict[str, Any] = {
        "run_dir": args.run_dir,
        "out_dir": args.out_dir,
        "canonical_csv": args.canonical_csv,
        "input_qc_json": args.input_qc_json,
        "checkpoint": args.checkpoint,
        "calibration_file": args.calibration_file,
    }
    # Remove None values for cleaner reports.
    paths = {k: v for k, v in paths.items() if v is not None}

    payload = run_quality(args.step, paths=paths, profile=profile, out_dir=args.out_dir)
    print(json.dumps(to_builtin({"status": payload.get("status"), "output_paths": payload.get("output_paths")}), indent=2))
    return 0 if payload.get("status") not in ["fail", "error"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
