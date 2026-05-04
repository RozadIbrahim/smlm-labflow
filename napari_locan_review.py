#!/usr/bin/env python3
"""
napari_locan_review.py

Optional downstream scientific review stage for the SMLM wrapper pipeline.

Run this AFTER run_pipeline.py / post_inference.py.

Recommended environment:
    conda activate napari_locan_env

Clean CLI:
    --input   batch directory, run directory, or direct localization CSV
    --out     review output directory

Purpose:
    1. Read post_inference outputs:
        - exports/locan/locan_localizations.csv
        - exports/napari/napari_points.csv
        - canonical_localizations.csv
        - exports/generic/smlm_generic_localizations.csv
        - combined run-level exports when applicable

    2. Standardize localization tables into:
        - position_x
        - position_y
        - position_z, optional
        - frame, optional
        - intensity, optional
        - background, optional
        - confidence, optional
        - channel, optional
        - file, optional
        - backend, optional

    3. Generate scientist-facing review outputs:
        - standardized filtered review table
        - scientific QC JSON summaries
        - XY render / density map
        - frame-count plot
        - intensity/background/confidence/Z histograms
        - nearest-neighbor distribution
        - simple drift proxy
        - optional DBSCAN cluster analysis
        - optional Ripley K / L spatial-statistics proxy
        - napari helper script
        - Locan helper script

Typical non-GUI usage:

    python napari_locan_review.py \
        --input results/run_001/batches/0001_movie \
        --out results/run_001/batches/0001_movie/review/napari_locan \
        --coord-units nm \
        --analysis-units nm \
        --pixel-size-nm 65 \
        --locan-review

With DBSCAN and Ripley proxy:

    python napari_locan_review.py \
        --input results/run_001/batches/0001_movie \
        --out results/run_001/batches/0001_movie/review/napari_locan \
        --coord-units nm \
        --analysis-units nm \
        --pixel-size-nm 65 \
        --locan-review \
        --dbscan \
        --dbscan-eps 50 \
        --dbscan-min-samples 10 \
        --ripley

Open napari too:

    python napari_locan_review.py \
        --input results/run_001/batches/0001_movie \
        --out results/run_001/batches/0001_movie/review/napari_locan \
        --coord-units nm \
        --analysis-units nm \
        --pixel-size-nm 65 \
        --locan-review \
        --open-napari \
        --napari-color-by confidence

Important:
    napari is GUI-based and can block execution.
    Use --open-napari only in a graphical session.

Notes:
    - The raw movie overlay is inferred from batch_manifest.json/csv when possible.
    - If no movie is found, napari opens localizations only.
    - DBSCAN and Ripley outputs are exploratory diagnostics, not final biological proof.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# JSON-safe utilities
# =============================================================================

def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert NumPy/pandas/path objects into JSON-safe Python types.

    Prevents errors such as:
        TypeError: Object of type bool_ is not JSON serializable
    """
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    return str(obj)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def display_path(path: Path | str | None) -> str:
    if path is None:
        return ""

    path = Path(path)

    if str(path).strip() == "":
        return ""

    try:
        resolved = path.expanduser().resolve()
        cwd = Path.cwd().resolve()

        try:
            return str(resolved.relative_to(cwd))
        except ValueError:
            return str(resolved)

    except Exception:
        return str(path)


# =============================================================================
# File discovery and table loading
# =============================================================================

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        print(f"[warning] CSV is empty: {path}")

    return df


def find_default_localization_file(input_dir: Path) -> Path:
    """
    Find the best available localization file in a post_inference output folder.

    --input may be:
        - one batch directory:
              results/run_001/batches/0001_movie

        - one run directory:
              results/run_001

        - a directory containing combined exports

    Priority:
        1. Locan adapted export
        2. napari points export
        3. canonical localizations
        4. generic SMLM export
        5. possible combined run-level exports
    """
    input_dir = Path(input_dir).expanduser().resolve()

    candidates = [
        input_dir / "exports" / "locan" / "locan_localizations.csv",
        input_dir / "exports" / "napari" / "napari_points.csv",
        input_dir / "canonical_localizations.csv",
        input_dir / "exports" / "generic" / "smlm_generic_localizations.csv",

        input_dir / "combined" / "locan_all_localizations.csv",
        input_dir / "combined" / "napari_all_points.csv",
        input_dir / "combined" / "canonical_all_localizations.csv",

        input_dir / "combined_exports" / "locan_all_localizations.csv",
        input_dir / "combined_exports" / "napari_all_points.csv",
        input_dir / "combined_exports" / "canonical_all_localizations.csv",

        input_dir / "combined_outputs" / "locan_all_localizations.csv",
        input_dir / "combined_outputs" / "napari_all_points.csv",
        input_dir / "combined_outputs" / "canonical_all_localizations.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No localization file found. Expected one of:\n"
        + "\n".join(str(c) for c in candidates)
    )


def find_batch_dir_from_path(path: Path) -> Optional[Path]:
    """
    Find a batch directory like:
        results/run_001/batches/0001_movie

    Works when --input is:
        - the batch directory itself
        - a CSV somewhere inside the batch directory
    """
    path = Path(path).expanduser().resolve()
    start = path if path.is_dir() else path.parent

    for candidate in [start, *start.parents]:
        if candidate.parent.name == "batches":
            return candidate

    return None


def find_run_dir_from_path(path: Path) -> Optional[Path]:
    """
    Find the top-level run directory containing batch_manifest.json/csv.

    Works for:
        - batch directory
        - direct CSV inside a batch directory
        - run directory
    """
    path = Path(path).expanduser().resolve()

    batch_dir = find_batch_dir_from_path(path)
    if batch_dir is not None:
        return batch_dir.parent.parent

    start = path if path.is_dir() else path.parent

    for candidate in [start, *start.parents]:
        if (candidate / "batch_manifest.json").exists():
            return candidate

        if (candidate / "batch_manifest.csv").exists():
            return candidate

        if (candidate / "run_summary.json").exists() and (candidate / "batches").exists():
            return candidate

    return None


def infer_movie_path_from_manifest(input_path: Path) -> Optional[Path]:
    """
    Infer the original raw TIFF/OME-TIFF movie from batch_manifest.json/csv.

    This avoids a mandatory --movie argument.

    Works best when --input is a batch directory created by run_pipeline.py:
        results/run_001/batches/0001_movie
    """
    input_path = Path(input_path).expanduser().resolve()

    batch_dir = find_batch_dir_from_path(input_path)
    run_dir = find_run_dir_from_path(input_path)

    if run_dir is None:
        return None

    manifest_json = run_dir / "batch_manifest.json"
    manifest_csv = run_dir / "batch_manifest.csv"

    def maybe_valid_movie(value: Any) -> Optional[Path]:
        if value is None:
            return None

        value = str(value).strip()

        if not value or value.lower() == "nan":
            return None

        movie = Path(value).expanduser().resolve()

        if movie.exists():
            return movie

        return None

    if manifest_json.exists():
        try:
            rows = json.loads(manifest_json.read_text(encoding="utf-8"))

            if isinstance(rows, dict):
                rows = [rows]

            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    row_run_dir = row.get("run_dir", "")

                    if batch_dir is not None and row_run_dir:
                        try:
                            if Path(row_run_dir).expanduser().resolve() == batch_dir:
                                movie = maybe_valid_movie(row.get("input_path", ""))
                                if movie is not None:
                                    return movie
                        except Exception:
                            pass

                if len(rows) == 1 and isinstance(rows[0], dict):
                    movie = maybe_valid_movie(rows[0].get("input_path", ""))
                    if movie is not None:
                        return movie

        except Exception:
            pass

    if manifest_csv.exists():
        try:
            rows = pd.read_csv(manifest_csv)

            if "run_dir" in rows.columns and "input_path" in rows.columns:
                for _, row in rows.iterrows():
                    row_run_dir = row.get("run_dir", "")

                    if batch_dir is not None and pd.notna(row_run_dir):
                        try:
                            if Path(str(row_run_dir)).expanduser().resolve() == batch_dir:
                                movie = maybe_valid_movie(row.get("input_path", ""))
                                if movie is not None:
                                    return movie
                        except Exception:
                            pass

            if len(rows) == 1 and "input_path" in rows.columns:
                movie = maybe_valid_movie(rows.iloc[0].get("input_path", ""))
                if movie is not None:
                    return movie

        except Exception:
            pass

    return None


def resolve_input_localization(input_path: Path) -> Tuple[Path, Path]:
    """
    Resolve --input into:
        localization_csv, input_root

    --input may be:
        1. a batch directory from run_pipeline.py
        2. a run directory containing combined exports
        3. a direct localization CSV
    """
    input_path = Path(input_path).expanduser().resolve()

    if input_path.is_file():
        input_root = find_batch_dir_from_path(input_path)

        if input_root is None:
            input_root = input_path.parent

        return input_path, input_root

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input must be a directory or CSV file: {input_path}")

    localization_csv = find_default_localization_file(input_path)

    return localization_csv, input_path


def detect_input_format(df: pd.DataFrame) -> str:
    """
    Detect supported localization table formats.

    Supported:
        locan:
            position_x, position_y

        canonical:
            x, y

        napari_points:
            axis_0, axis_1
    """
    cols = set(df.columns)

    if {"position_x", "position_y"}.issubset(cols):
        return "locan"

    if {"x", "y"}.issubset(cols):
        return "canonical"

    if {"axis_0", "axis_1"}.issubset(cols):
        return "napari_points"

    raise ValueError(
        "Could not detect localization table format. "
        "Expected either position_x/position_y, x/y, or axis_0/axis_1 columns. "
        f"Observed columns: {list(df.columns)}"
    )


# =============================================================================
# Numeric and unit helpers
# =============================================================================

def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_units(
    values: pd.Series,
    from_units: str,
    to_units: str,
    pixel_size_nm: Optional[float],
) -> pd.Series:
    values = numeric(values)

    if from_units == to_units:
        return values

    if pixel_size_nm is None or pixel_size_nm <= 0:
        return values

    if from_units == "nm" and to_units == "pixel":
        return values / float(pixel_size_nm)

    if from_units == "pixel" and to_units == "nm":
        return values * float(pixel_size_nm)

    return values


def robust_numeric_summary(series: pd.Series) -> Dict[str, Any]:
    values = numeric(series)
    valid = values.dropna()

    if len(valid) == 0:
        return {
            "available": False,
            "n_valid": 0,
            "n_missing": int(values.isna().sum()),
        }

    return {
        "available": True,
        "n_valid": int(len(valid)),
        "n_missing": int(values.isna().sum()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "std": float(valid.std()) if len(valid) > 1 else 0.0,
        "p01": float(valid.quantile(0.01)),
        "p05": float(valid.quantile(0.05)),
        "p95": float(valid.quantile(0.95)),
        "p99": float(valid.quantile(0.99)),
    }


# =============================================================================
# Standardization and filtering
# =============================================================================

def standardize_localizations(
    df: pd.DataFrame,
    coord_units: str,
    target_units: str,
    pixel_size_nm: Optional[float],
) -> Tuple[pd.DataFrame, str]:
    """
    Convert supported localization tables to a common scientific schema:

        position_x
        position_y
        position_z, optional
        frame, optional
        intensity, optional
        background, optional
        confidence, optional
        channel, optional
        file, optional
        backend, optional
    """
    detected_format = detect_input_format(df)
    out = pd.DataFrame(index=df.index)

    if detected_format == "locan":
        out["position_x"] = convert_units(
            df["position_x"],
            coord_units,
            target_units,
            pixel_size_nm,
        )
        out["position_y"] = convert_units(
            df["position_y"],
            coord_units,
            target_units,
            pixel_size_nm,
        )

        if "position_z" in df.columns:
            z = numeric(df["position_z"])
            if z.notna().any():
                out["position_z"] = convert_units(
                    df["position_z"],
                    coord_units,
                    target_units,
                    pixel_size_nm,
                )

    elif detected_format == "canonical":
        out["position_x"] = convert_units(
            df["x"],
            coord_units,
            target_units,
            pixel_size_nm,
        )
        out["position_y"] = convert_units(
            df["y"],
            coord_units,
            target_units,
            pixel_size_nm,
        )

        if "z" in df.columns:
            z = numeric(df["z"])
            if z.notna().any():
                out["position_z"] = convert_units(
                    df["z"],
                    coord_units,
                    target_units,
                    pixel_size_nm,
                )

    elif detected_format == "napari_points":
        # post_inference convention:
        #   2D: axis_0 = y, axis_1 = x
        #   3D: axis_0 = z, axis_1 = y, axis_2 = x
        if "axis_2" in df.columns:
            out["position_z"] = convert_units(
                df["axis_0"],
                coord_units,
                target_units,
                pixel_size_nm,
            )
            out["position_y"] = convert_units(
                df["axis_1"],
                coord_units,
                target_units,
                pixel_size_nm,
            )
            out["position_x"] = convert_units(
                df["axis_2"],
                coord_units,
                target_units,
                pixel_size_nm,
            )
        else:
            out["position_y"] = convert_units(
                df["axis_0"],
                coord_units,
                target_units,
                pixel_size_nm,
            )
            out["position_x"] = convert_units(
                df["axis_1"],
                coord_units,
                target_units,
                pixel_size_nm,
            )

    optional_map = {
        "frame": ["frame"],
        "intensity": ["intensity", "photons", "amplitude", "signal"],
        "background": ["background", "bg", "bkg"],
        "confidence": ["confidence", "score", "probability", "prob"],
        "channel": ["channel", "batch_index", "group"],
        "file": ["file", "input_name", "source_file"],
        "backend": ["backend"],
    }

    for output_col, candidates in optional_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                out[output_col] = df[candidate]
                break

    numeric_cols = [
        "position_x",
        "position_y",
        "position_z",
        "frame",
        "intensity",
        "background",
        "confidence",
        "channel",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = numeric(out[col])

    out = out.dropna(subset=["position_x", "position_y"]).reset_index(drop=True)

    return out, detected_format


def apply_filters(
    df: pd.DataFrame,
    min_confidence: Optional[float],
    max_confidence: Optional[float],
    min_intensity: Optional[float],
    max_intensity: Optional[float],
    min_frame: Optional[int],
    max_frame: Optional[int],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    z_min: Optional[float],
    z_max: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filtered = df.copy()
    n_before = int(len(filtered))

    filter_report: Dict[str, Any] = {
        "n_before": n_before,
        "filters": {},
    }

    def apply_range(column: str, low: Any, high: Any) -> None:
        nonlocal filtered

        if column not in filtered.columns:
            return

        values = numeric(filtered[column])

        if low is not None:
            filtered = filtered[values >= low]
            filter_report["filters"][f"{column}_min"] = low

        if high is not None:
            values = numeric(filtered[column])
            filtered = filtered[values <= high]
            filter_report["filters"][f"{column}_max"] = high

    apply_range("confidence", min_confidence, max_confidence)
    apply_range("intensity", min_intensity, max_intensity)
    apply_range("frame", min_frame, max_frame)
    apply_range("position_x", x_min, x_max)
    apply_range("position_y", y_min, y_max)
    apply_range("position_z", z_min, z_max)

    filtered = filtered.reset_index(drop=True)

    filter_report["n_after"] = int(len(filtered))
    filter_report["n_removed"] = int(n_before - len(filtered))
    filter_report["fraction_kept"] = (
        float(len(filtered) / n_before)
        if n_before > 0
        else 0.0
    )

    return filtered, filter_report


# =============================================================================
# Scientific summaries
# =============================================================================

def summarize_positions(df: pd.DataFrame, units: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "units": units,
        "n_localizations": int(len(df)),
        "has_z": bool("position_z" in df.columns and df["position_z"].notna().any()),
    }

    if len(df) == 0:
        return summary

    summary["numeric_summary"] = {}

    for col in [
        "position_x",
        "position_y",
        "position_z",
        "frame",
        "intensity",
        "background",
        "confidence",
        "channel",
    ]:
        if col in df.columns:
            summary["numeric_summary"][col] = robust_numeric_summary(df[col])

    x = numeric(df["position_x"]).dropna()
    y = numeric(df["position_y"]).dropna()

    if len(x) > 0 and len(y) > 0:
        width = float(x.max() - x.min())
        height = float(y.max() - y.min())
        area = width * height

        summary["bounding_box"] = {
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "width": width,
            "height": height,
            "area": area,
        }

        if area > 0:
            summary["density_per_unit2"] = float(len(df) / area)

    if "position_z" in df.columns:
        z = numeric(df["position_z"]).dropna()

        if len(z) > 0:
            summary["z_range"] = {
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "z_span": float(z.max() - z.min()),
            }

    if "frame" in df.columns:
        frames = numeric(df["frame"]).dropna()

        if len(frames) > 0:
            summary["frames"] = {
                "first": int(frames.min()),
                "last": int(frames.max()),
                "n_unique": int(frames.nunique()),
            }

            counts = frames.astype(int).value_counts().sort_index()

            summary["localizations_per_frame"] = {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": float(counts.mean()),
                "median": float(counts.median()),
                "std": float(counts.std()) if len(counts) > 1 else 0.0,
            }

    return summary


def build_quality_flags(
    df: pd.DataFrame,
    units: str,
    pixel_size_nm: Optional[float],
    filter_report: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    flags: list[Dict[str, Any]] = []

    if len(df) == 0:
        flags.append({
            "level": "error",
            "code": "EMPTY_STANDARDIZED_TABLE",
            "message": "No valid x/y localizations after standardization/filtering.",
        })
        return flags

    for col in ["position_x", "position_y"]:
        if col not in df.columns or df[col].dropna().empty:
            flags.append({
                "level": "error",
                "code": f"MISSING_{col.upper()}",
                "message": f"{col} is missing or empty.",
            })

    x = numeric(df["position_x"]).dropna()
    y = numeric(df["position_y"]).dropna()

    if len(x) > 0 and len(y) > 0:
        if float(x.max() - x.min()) == 0 or float(y.max() - y.min()) == 0:
            flags.append({
                "level": "warning",
                "code": "DEGENERATE_XY_RANGE",
                "message": "X or Y range is degenerate. Check coordinate mapping.",
            })

    if "frame" not in df.columns or df["frame"].dropna().empty:
        flags.append({
            "level": "info",
            "code": "NO_FRAME_INFORMATION",
            "message": "Frame information is missing. Temporal QC and drift proxy are limited.",
        })

    if units == "pixel" and pixel_size_nm is None:
        flags.append({
            "level": "info",
            "code": "PIXEL_SIZE_NOT_PROVIDED",
            "message": "Pixel size was not provided. Physical-unit conversion is unavailable.",
        })

    if "confidence" in df.columns:
        confidence = numeric(df["confidence"]).dropna()

        if len(confidence) > 0:
            if confidence.min() < 0:
                flags.append({
                    "level": "warning",
                    "code": "NEGATIVE_CONFIDENCE",
                    "message": "Some confidence values are negative.",
                })

            if confidence.max() > 1.5:
                flags.append({
                    "level": "info",
                    "code": "CONFIDENCE_MAY_BE_SCORE",
                    "message": "Confidence values exceed 1. They may be scores rather than probabilities.",
                })

    if filter_report is not None and filter_report.get("fraction_kept", 1.0) < 0.1:
        flags.append({
            "level": "warning",
            "code": "FILTER_REMOVED_MOST_LOCALIZATIONS",
            "message": "Filters kept less than 10% of localizations. Check thresholds.",
        })

    return flags


# =============================================================================
# Plot helpers
# =============================================================================

def save_empty_plot(out_path: Path, title: str, message: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_xy_render(
    df: pd.DataFrame,
    out_path: Path,
    bin_size: float,
    units: str,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "path": str(out_path),
        "enabled": False,
    }

    if len(df) == 0:
        save_empty_plot(out_path, "XY render", "No localizations")
        report["reason"] = "empty_table"
        return report

    valid = df[["position_x", "position_y"]].copy()
    valid["position_x"] = numeric(valid["position_x"])
    valid["position_y"] = numeric(valid["position_y"])
    valid = valid.dropna()

    if len(valid) == 0:
        save_empty_plot(out_path, "XY render", "No valid x/y coordinates")
        report["reason"] = "no_valid_xy"
        return report

    x_min = float(valid["position_x"].min())
    x_max = float(valid["position_x"].max())
    y_min = float(valid["position_y"].min())
    y_max = float(valid["position_y"].max())

    if x_max <= x_min or y_max <= y_min:
        save_empty_plot(out_path, "XY render", "Degenerate x/y range")
        report["reason"] = "degenerate_range"
        return report

    if bin_size <= 0:
        bin_size = max((x_max - x_min), (y_max - y_min)) / 512

    x_bins = max(10, int(np.ceil((x_max - x_min) / bin_size)))
    y_bins = max(10, int(np.ceil((y_max - y_min) / bin_size)))

    max_bins = 4000

    if x_bins > max_bins or y_bins > max_bins:
        scale = max(x_bins / max_bins, y_bins / max_bins)
        x_bins = max(10, int(x_bins / scale))
        y_bins = max(10, int(y_bins / scale))

    hist, x_edges, y_edges = np.histogram2d(
        valid["position_x"],
        valid["position_y"],
        bins=[x_bins, y_bins],
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(
        hist.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
    )
    plt.xlabel(f"x ({units})")
    plt.ylabel(f"y ({units})")
    plt.title(f"SMLM XY render, bin≈{bin_size:g} {units}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    report.update({
        "enabled": True,
        "n_points": int(len(valid)),
        "x_bins": int(x_bins),
        "y_bins": int(y_bins),
        "bin_size_requested": float(bin_size),
        "units": units,
    })

    return report


def plot_frame_counts(df: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "path": str(out_path),
        "enabled": False,
    }

    if "frame" not in df.columns:
        save_empty_plot(out_path, "Localizations per frame", "No frame column")
        report["reason"] = "missing_frame"
        return report

    frames = numeric(df["frame"]).dropna()

    if len(frames) == 0:
        save_empty_plot(out_path, "Localizations per frame", "No valid frame values")
        report["reason"] = "no_valid_frame"
        return report

    counts = frames.astype(int).value_counts().sort_index()

    plt.figure(figsize=(9, 4))
    plt.plot(counts.index, counts.values, linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Localizations")
    plt.title("Localizations per frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    report.update({
        "enabled": True,
        "n_frames": int(len(counts)),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "mean_count": float(counts.mean()),
    })

    return report


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "path": str(out_path),
        "enabled": False,
        "column": column,
    }

    if column not in df.columns:
        save_empty_plot(out_path, title, f"No {column} column")
        report["reason"] = "missing_column"
        return report

    values = numeric(df[column]).dropna()

    if len(values) == 0:
        save_empty_plot(out_path, title, f"No valid {column} values")
        report["reason"] = "no_valid_values"
        return report

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    report.update({
        "enabled": True,
        "n_values": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
    })

    return report


# =============================================================================
# Spatial analyses
# =============================================================================

def plot_nearest_neighbor_histogram(
    df: pd.DataFrame,
    out_path: Path,
    units: str,
    max_points: int,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "path": str(out_path),
    }

    try:
        from scipy.spatial import cKDTree
    except Exception:
        save_empty_plot(
            out_path,
            "Nearest-neighbor distance",
            "scipy not installed; nearest-neighbor QC skipped",
        )
        report["reason"] = "scipy_not_installed"
        return report

    cols = ["position_x", "position_y"]

    if "position_z" in df.columns and bool(df["position_z"].notna().any()):
        cols = ["position_x", "position_y", "position_z"]

    points_df = df[cols].copy()

    for col in cols:
        points_df[col] = numeric(points_df[col])

    points_df = points_df.dropna()

    if len(points_df) < 3:
        save_empty_plot(out_path, "Nearest-neighbor distance", "Too few points")
        report["reason"] = "too_few_points"
        return report

    if len(points_df) > max_points:
        points_df = points_df.sample(max_points, random_state=42)

    points = points_df.to_numpy(dtype=float)

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)

    nn = distances[:, 1]

    plt.figure(figsize=(7, 4))
    plt.hist(nn, bins=80)
    plt.xlabel(f"Nearest-neighbor distance ({units})")
    plt.ylabel("Count")
    plt.title("Nearest-neighbor distance distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    report.update({
        "enabled": True,
        "n_points_used": int(len(points)),
        "dimension": int(points.shape[1]),
        "units": units,
        "mean": float(np.mean(nn)),
        "median": float(np.median(nn)),
        "p05": float(np.quantile(nn, 0.05)),
        "p95": float(np.quantile(nn, 0.95)),
    })

    return report


def plot_drift_proxy(
    df: pd.DataFrame,
    out_path: Path,
    units: str,
    frame_bin: int,
) -> Dict[str, Any]:
    """
    Simple drift proxy:
        compute median x/y per frame-bin.

    This is not drift correction.
    It is a diagnostic trend plot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "path": str(out_path),
    }

    required = {"frame", "position_x", "position_y"}

    if not required.issubset(df.columns):
        save_empty_plot(out_path, "Drift proxy", "Missing frame/x/y information")
        report["reason"] = "missing_columns"
        return report

    tmp = df[["frame", "position_x", "position_y"]].copy()
    tmp["frame"] = numeric(tmp["frame"])
    tmp["position_x"] = numeric(tmp["position_x"])
    tmp["position_y"] = numeric(tmp["position_y"])
    tmp = tmp.dropna()

    if len(tmp) < 10:
        save_empty_plot(out_path, "Drift proxy", "Too few localizations")
        report["reason"] = "too_few_points"
        return report

    if frame_bin <= 0:
        frame_bin = 100

    tmp["frame_bin"] = (tmp["frame"] // frame_bin).astype(int) * frame_bin

    grouped = tmp.groupby("frame_bin", as_index=False).agg(
        median_x=("position_x", "median"),
        median_y=("position_y", "median"),
        n=("position_x", "size"),
    )

    if len(grouped) < 2:
        save_empty_plot(out_path, "Drift proxy", "Too few frame bins")
        report["reason"] = "too_few_frame_bins"
        return report

    plt.figure(figsize=(9, 4))
    plt.plot(grouped["frame_bin"], grouped["median_x"], linewidth=1, label="median x")
    plt.plot(grouped["frame_bin"], grouped["median_y"], linewidth=1, label="median y")
    plt.xlabel("Frame bin")
    plt.ylabel(f"Median position ({units})")
    plt.title(f"Drift proxy, frame bin={frame_bin}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    grouped_path = out_path.with_suffix(".csv")
    grouped.to_csv(grouped_path, index=False)

    report.update({
        "enabled": True,
        "frame_bin": int(frame_bin),
        "n_bins": int(len(grouped)),
        "x_span_median": float(grouped["median_x"].max() - grouped["median_x"].min()),
        "y_span_median": float(grouped["median_y"].max() - grouped["median_y"].min()),
        "table": str(grouped_path),
    })

    return report


def run_dbscan_clustering(
    df: pd.DataFrame,
    out_dir: Path,
    units: str,
    eps: float,
    min_samples: int,
    max_points: int,
    use_3d: bool,
) -> Dict[str, Any]:
    """
    Optional DBSCAN clustering.

    Notes:
        - DBSCAN parameters are dataset-dependent.
        - eps is in analysis_units.
        - For huge datasets, this samples points unless max_points is raised.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_plot_path = out_dir / "locan_dbscan_clusters.png"
    cluster_table_path = out_dir / "locan_dbscan_clustered_points.csv"
    cluster_summary_path = out_dir / "locan_dbscan_cluster_summary.csv"

    report: Dict[str, Any] = {
        "enabled": False,
        "path_clustered_points": str(cluster_table_path),
        "path_cluster_summary": str(cluster_summary_path),
        "plot": str(cluster_plot_path),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "units": units,
    }

    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        save_empty_plot(cluster_plot_path, "DBSCAN clusters", "scikit-learn not installed")
        report["reason"] = "sklearn_not_installed"
        return report

    if eps <= 0 or min_samples <= 0:
        save_empty_plot(cluster_plot_path, "DBSCAN clusters", "Invalid DBSCAN parameters")
        report["reason"] = "invalid_parameters"
        return report

    cols = ["position_x", "position_y"]

    if use_3d and "position_z" in df.columns and bool(df["position_z"].notna().any()):
        cols = ["position_x", "position_y", "position_z"]

    points_df = df[cols].copy()

    for col in cols:
        points_df[col] = numeric(points_df[col])

    points_df = points_df.dropna()

    if len(points_df) < min_samples:
        save_empty_plot(cluster_plot_path, "DBSCAN clusters", "Too few points")
        report["reason"] = "too_few_points"
        return report

    sampled = False

    if len(points_df) > max_points:
        points_df = points_df.sample(max_points, random_state=42)
        sampled = True

    points = points_df.to_numpy(dtype=float)

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points)

    clustered = points_df.copy()
    clustered["cluster_id"] = labels
    clustered.to_csv(cluster_table_path, index=False)

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))

    cluster_rows = []

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue

        mask = labels == cluster_id
        cluster_points = points[mask]

        row = {
            "cluster_id": int(cluster_id),
            "n_points": int(mask.sum()),
            "centroid_x": float(cluster_points[:, 0].mean()),
            "centroid_y": float(cluster_points[:, 1].mean()),
            "span_x": float(cluster_points[:, 0].max() - cluster_points[:, 0].min()),
            "span_y": float(cluster_points[:, 1].max() - cluster_points[:, 1].min()),
        }

        if cluster_points.shape[1] == 3:
            row["centroid_z"] = float(cluster_points[:, 2].mean())
            row["span_z"] = float(cluster_points[:, 2].max() - cluster_points[:, 2].min())

        cluster_rows.append(row)

    cluster_summary = pd.DataFrame(cluster_rows)
    cluster_summary.to_csv(cluster_summary_path, index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=0.5)
    plt.xlabel(f"x ({units})")
    plt.ylabel(f"y ({units})")
    plt.title(f"DBSCAN clusters: eps={eps:g} {units}, min_samples={min_samples}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(cluster_plot_path, dpi=300)
    plt.close()

    report.update({
        "enabled": True,
        "sampled": bool(sampled),
        "n_points_used": int(len(points)),
        "dimension": int(points.shape[1]),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / len(points)) if len(points) > 0 else None,
    })

    return report


def run_ripley_l_proxy(
    df: pd.DataFrame,
    out_dir: Path,
    units: str,
    max_radius: float,
    n_radii: int,
    max_points: int,
) -> Dict[str, Any]:
    """
    Approximate 2D Ripley K/L proxy.

    This is a practical diagnostic, not a fully edge-corrected spatial-statistics
    implementation. It is useful for comparing runs/settings, but final
    biological conclusions should use validated statistical workflows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = out_dir / "locan_ripley_l_proxy.png"
    table_path = out_dir / "locan_ripley_l_proxy.csv"

    report: Dict[str, Any] = {
        "enabled": False,
        "plot": str(plot_path),
        "table": str(table_path),
        "units": units,
    }

    try:
        from scipy.spatial import cKDTree
    except Exception:
        save_empty_plot(plot_path, "Ripley L proxy", "scipy not installed")
        report["reason"] = "scipy_not_installed"
        return report

    points_df = df[["position_x", "position_y"]].copy()
    points_df["position_x"] = numeric(points_df["position_x"])
    points_df["position_y"] = numeric(points_df["position_y"])
    points_df = points_df.dropna()

    if len(points_df) < 10:
        save_empty_plot(plot_path, "Ripley L proxy", "Too few points")
        report["reason"] = "too_few_points"
        return report

    if len(points_df) > max_points:
        points_df = points_df.sample(max_points, random_state=42)

    points = points_df.to_numpy(dtype=float)

    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    area = (x_max - x_min) * (y_max - y_min)

    if area <= 0:
        save_empty_plot(plot_path, "Ripley L proxy", "Degenerate area")
        report["reason"] = "degenerate_area"
        return report

    if max_radius <= 0:
        max_radius = min(x_max - x_min, y_max - y_min) / 10

    if n_radii <= 1:
        n_radii = 50

    radii = np.linspace(max_radius / n_radii, max_radius, n_radii)

    tree = cKDTree(points)
    n = len(points)

    rows = []

    for r in radii:
        counts = tree.query_ball_point(points, r, return_length=True)

        # subtract self-count
        total_neighbors = float(np.sum(counts - 1))

        # K estimate without edge correction
        k_value = area * total_neighbors / (n * (n - 1))
        l_value = math.sqrt(k_value / math.pi)
        l_minus_r = l_value - r

        rows.append({
            "radius": float(r),
            "K": float(k_value),
            "L": float(l_value),
            "L_minus_r": float(l_minus_r),
        })

    result = pd.DataFrame(rows)
    result.to_csv(table_path, index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(result["radius"], result["L_minus_r"], linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.xlabel(f"Radius ({units})")
    plt.ylabel(f"L(r) - r ({units})")
    plt.title("Ripley L proxy, no edge correction")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    idx_max = result["L_minus_r"].idxmax()

    report.update({
        "enabled": True,
        "n_points_used": int(n),
        "area": float(area),
        "max_radius": float(max_radius),
        "n_radii": int(n_radii),
        "max_L_minus_r": float(result["L_minus_r"].max()),
        "radius_at_max_L_minus_r": float(result.loc[idx_max, "radius"]),
    })

    return report


# =============================================================================
# Locan-style review
# =============================================================================

def run_locan_review(
    standardized: pd.DataFrame,
    review_dir: Path,
    units: str,
    render_bin_size: float,
    max_nn_points: int,
    drift_frame_bin: int,
    run_dbscan: bool,
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_max_points: int,
    dbscan_use_3d: bool,
    run_ripley: bool,
    ripley_max_radius: float,
    ripley_n_radii: int,
    ripley_max_points: int,
    pixel_size_nm: Optional[float],
    filter_report: Dict[str, Any],
) -> Dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_positions(standardized, units=units)

    summary["filter_report"] = filter_report
    summary["quality_flags"] = build_quality_flags(
        standardized,
        units=units,
        pixel_size_nm=pixel_size_nm,
        filter_report=filter_report,
    )

    review_table_path = review_dir / "locan_review_table.csv"
    standardized.to_csv(review_table_path, index=False)

    render_path = review_dir / "locan_xy_render.png"
    frame_path = review_dir / "locan_frame_counts.png"
    confidence_path = review_dir / "locan_confidence_histogram.png"
    intensity_path = review_dir / "locan_intensity_histogram.png"
    background_path = review_dir / "locan_background_histogram.png"
    z_path = review_dir / "locan_z_histogram.png"
    nn_path = review_dir / "locan_nearest_neighbor_histogram.png"
    drift_path = review_dir / "locan_drift_proxy.png"

    plot_reports = {
        "xy_render": plot_xy_render(
            df=standardized,
            out_path=render_path,
            bin_size=render_bin_size,
            units=units,
        ),
        "frame_counts": plot_frame_counts(
            df=standardized,
            out_path=frame_path,
        ),
        "confidence_histogram": plot_histogram(
            df=standardized,
            column="confidence",
            out_path=confidence_path,
            title="Confidence / score distribution",
            xlabel="Confidence / score",
        ),
        "intensity_histogram": plot_histogram(
            df=standardized,
            column="intensity",
            out_path=intensity_path,
            title="Intensity / photon distribution",
            xlabel="Intensity / photons",
        ),
        "background_histogram": plot_histogram(
            df=standardized,
            column="background",
            out_path=background_path,
            title="Background distribution",
            xlabel="Background",
        ),
        "z_histogram": plot_histogram(
            df=standardized,
            column="position_z",
            out_path=z_path,
            title="Z distribution",
            xlabel=f"z ({units})",
        ),
    }

    nn_report = plot_nearest_neighbor_histogram(
        df=standardized,
        out_path=nn_path,
        units=units,
        max_points=max_nn_points,
    )

    drift_report = plot_drift_proxy(
        df=standardized,
        out_path=drift_path,
        units=units,
        frame_bin=drift_frame_bin,
    )

    dbscan_report = {
        "enabled": False,
        "reason": "not_requested",
    }

    if run_dbscan:
        dbscan_report = run_dbscan_clustering(
            df=standardized,
            out_dir=review_dir,
            units=units,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            max_points=dbscan_max_points,
            use_3d=dbscan_use_3d,
        )

    ripley_report = {
        "enabled": False,
        "reason": "not_requested",
    }

    if run_ripley:
        ripley_report = run_ripley_l_proxy(
            df=standardized,
            out_dir=review_dir,
            units=units,
            max_radius=ripley_max_radius,
            n_radii=ripley_n_radii,
            max_points=ripley_max_points,
        )

    summary["plots"] = plot_reports
    summary["nearest_neighbor"] = nn_report
    summary["drift_proxy"] = drift_report
    summary["dbscan"] = dbscan_report
    summary["ripley_l_proxy"] = ripley_report

    summary["artifacts"] = {
        "review_table": str(review_table_path),
        "xy_render": str(render_path),
        "frame_counts": str(frame_path),
        "confidence_histogram": str(confidence_path),
        "intensity_histogram": str(intensity_path),
        "background_histogram": str(background_path),
        "z_histogram": str(z_path),
        "nearest_neighbor_histogram": str(nn_path),
        "drift_proxy": str(drift_path),
    }

    try:
        import locan as lc

        summary["locan_package"] = {
            "available": True,
            "version": getattr(lc, "__version__", "unknown"),
        }

    except Exception:
        summary["locan_package"] = {
            "available": False,
            "message": "Python package 'locan' is not installed or could not be imported.",
        }

    summary_path = review_dir / "locan_review_summary.json"
    write_json(summary, summary_path)

    return summary


# =============================================================================
# napari integration
# =============================================================================

def load_movie_projection(movie_path: Path) -> Optional[np.ndarray]:
    if movie_path is None:
        return None

    if not movie_path.exists():
        raise FileNotFoundError(f"Movie not found: {movie_path}")

    try:
        import tifffile
    except Exception as exc:
        raise ImportError(
            "tifffile is required to load TIFF movies for napari preview. "
            "Install with: pip install tifffile"
        ) from exc

    arr = tifffile.imread(movie_path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        return np.max(arr, axis=0)

    if arr.ndim >= 4:
        leading_axes = tuple(range(arr.ndim - 2))
        return np.max(arr, axis=leading_axes)

    return arr


def create_napari_points(
    standardized: pd.DataFrame,
    units: str,
    use_3d: bool,
    max_points: int,
) -> Tuple[np.ndarray, pd.DataFrame, list[str]]:
    df = standardized.copy()

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).reset_index(drop=True)

    has_z = bool("position_z" in df.columns and df["position_z"].notna().any())

    if use_3d and has_z:
        coord_cols = ["position_z", "position_y", "position_x"]
        axis_labels = ["z", "y", "x"]
    else:
        coord_cols = ["position_y", "position_x"]
        axis_labels = ["y", "x"]

    points = df[coord_cols].to_numpy(dtype=float)

    feature_cols = [
        c for c in df.columns
        if c not in ["position_x", "position_y", "position_z"]
    ]

    features = df[feature_cols].copy()
    features["viewer_units"] = units

    return points, features, axis_labels


def open_in_napari(
    standardized: pd.DataFrame,
    movie_path: Optional[Path],
    units: str,
    pixel_size_nm: Optional[float],
    use_3d: bool,
    point_size: float,
    max_points: int,
    color_by: Optional[str],
) -> None:
    try:
        import napari
    except Exception as exc:
        raise ImportError(
            "napari is not installed in this environment. "
            "Run this script inside napari_locan_env."
        ) from exc

    viewer = napari.Viewer()

    if movie_path is not None:
        projection = load_movie_projection(movie_path)

        if projection is not None:
            if units == "nm" and pixel_size_nm is not None and pixel_size_nm > 0:
                image_scale = (pixel_size_nm, pixel_size_nm)
            else:
                image_scale = (1, 1)

            viewer.add_image(
                projection,
                name="input max projection",
                scale=image_scale,
            )

    points, features, axis_labels = create_napari_points(
        standardized=standardized,
        units=units,
        use_3d=use_3d,
        max_points=max_points,
    )

    add_kwargs = {
        "features": features,
        "size": point_size,
        "name": "SMLM localizations",
    }

    if color_by and color_by in features.columns:
        add_kwargs["face_color"] = color_by
        add_kwargs["face_colormap"] = "viridis"

    viewer.add_points(points, **add_kwargs)

    try:
        viewer.layers["SMLM localizations"].axis_labels = tuple(axis_labels)
    except Exception:
        pass

    print("[napari] Viewer opened.")
    print("[napari] Close the napari window to return to the terminal.")

    napari.run()


def write_napari_helper(
    review_dir: Path,
    review_table_path: Path,
    movie_path: Optional[Path],
    units: str,
    pixel_size_nm: Optional[float],
    color_by: Optional[str],
) -> Path:
    helper_path = review_dir / "open_review_in_napari.py"

    movie_line = (
        f"movie_path = Path(r'{movie_path}')"
        if movie_path
        else "movie_path = None"
    )

    color_line = (
        f'color_by = "{color_by}"'
        if color_by
        else "color_by = None"
    )

    helper_code = f'''\
from pathlib import Path
import pandas as pd
import numpy as np
import napari

try:
    import tifffile
except Exception:
    tifffile = None

table_path = Path(r"{review_table_path}")
{movie_line}
units = "{units}"
pixel_size_nm = {pixel_size_nm!r}
{color_line}

df = pd.read_csv(table_path)

viewer = napari.Viewer()

if movie_path is not None and tifffile is not None:
    arr = tifffile.imread(movie_path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        projection = arr
    elif arr.ndim == 3:
        projection = np.max(arr, axis=0)
    else:
        projection = np.max(arr, axis=tuple(range(arr.ndim - 2)))

    scale = (pixel_size_nm, pixel_size_nm) if units == "nm" and pixel_size_nm else (1, 1)

    viewer.add_image(
        projection,
        name="input max projection",
        scale=scale,
    )

has_z = bool("position_z" in df.columns and df["position_z"].notna().any())

if has_z:
    points = df[["position_z", "position_y", "position_x"]].to_numpy()
else:
    points = df[["position_y", "position_x"]].to_numpy()

features = df.drop(
    columns=["position_x", "position_y", "position_z"],
    errors="ignore",
)

kwargs = {{
    "features": features,
    "size": 2,
    "name": "SMLM localizations",
}}

if color_by is not None and color_by in features.columns:
    kwargs["face_color"] = color_by
    kwargs["face_colormap"] = "viridis"

viewer.add_points(points, **kwargs)

napari.run()
'''

    helper_path.write_text(helper_code, encoding="utf-8")

    return helper_path


def write_locan_helper(
    review_dir: Path,
    review_table_path: Path,
) -> Path:
    helper_path = review_dir / "load_review_table_with_locan_example.py"

    helper_code = f'''\
"""
Example helper for loading the standardized review localization table.

Run inside napari_locan_env or another environment with locan installed.
"""

from pathlib import Path
import pandas as pd

table_path = Path(r"{review_table_path}")
df = pd.read_csv(table_path)

print(df.head())
print(df.describe(include="all"))

try:
    import locan as lc
    print("Locan version:", getattr(lc, "__version__", "unknown"))

    # The DataFrame is already standardized with:
    # position_x, position_y, optional position_z.
    #
    # Use this table as the clean input for Locan notebooks/scripts.
    #
    # Locan's exact constructors/API can vary by version, so this helper
    # intentionally keeps the data-loading step conservative.
except Exception as exc:
    print("Could not import locan:", repr(exc))
'''

    helper_path.write_text(helper_code, encoding="utf-8")

    return helper_path


# =============================================================================
# Main review runner
# =============================================================================

def run_review(
    input_path: str | Path,
    out_dir: str | Path,
    coord_units: str = "nm",
    analysis_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
    locan_review: bool = True,
    open_napari_viewer: bool = False,
    napari_3d: bool = False,
    point_size: float = 2.0,
    max_napari_points: int = 300_000,
    napari_color_by: Optional[str] = None,
    render_bin_size: float = 20.0,
    max_nn_points: int = 100_000,
    drift_frame_bin: int = 100,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    min_intensity: Optional[float] = None,
    max_intensity: Optional[float] = None,
    min_frame: Optional[int] = None,
    max_frame: Optional[int] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    dbscan: bool = False,
    dbscan_eps: float = 50.0,
    dbscan_min_samples: int = 10,
    dbscan_max_points: int = 200_000,
    dbscan_3d: bool = False,
    ripley: bool = False,
    ripley_max_radius: float = 500.0,
    ripley_n_radii: int = 50,
    ripley_max_points: int = 20_000,
) -> Dict[str, Any]:
    """
    Main scientific review runner.

    Clean interface:
        --input = batch directory, run directory, or localization CSV
        --out   = review output directory

    No --run.
    No mandatory --movie.
    Movie overlay is inferred from batch_manifest.json/csv when possible.
    """
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    localization_csv, input_root = resolve_input_localization(input_path)
    movie_path_obj = infer_movie_path_from_manifest(input_path)

    raw_df = read_csv_checked(localization_csv)

    standardized_raw, detected_format = standardize_localizations(
        df=raw_df,
        coord_units=coord_units,
        target_units=analysis_units,
        pixel_size_nm=pixel_size_nm,
    )

    standardized, filter_report = apply_filters(
        df=standardized_raw,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        min_frame=min_frame,
        max_frame=max_frame,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    review_table_path = out_dir / "locan_review_table.csv"
    standardized.to_csv(review_table_path, index=False)

    locan_summary: Dict[str, Any] = {}

    if locan_review:
        locan_summary = run_locan_review(
            standardized=standardized,
            review_dir=out_dir,
            units=analysis_units,
            render_bin_size=render_bin_size,
            max_nn_points=max_nn_points,
            drift_frame_bin=drift_frame_bin,
            run_dbscan=dbscan,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            dbscan_max_points=dbscan_max_points,
            dbscan_use_3d=dbscan_3d,
            run_ripley=ripley,
            ripley_max_radius=ripley_max_radius,
            ripley_n_radii=ripley_n_radii,
            ripley_max_points=ripley_max_points,
            pixel_size_nm=pixel_size_nm,
            filter_report=filter_report,
        )

    quality_flags = build_quality_flags(
        standardized,
        units=analysis_units,
        pixel_size_nm=pixel_size_nm,
        filter_report=filter_report,
    )

    napari_helper_path = write_napari_helper(
        review_dir=out_dir,
        review_table_path=review_table_path,
        movie_path=movie_path_obj,
        units=analysis_units,
        pixel_size_nm=pixel_size_nm,
        color_by=napari_color_by,
    )

    locan_helper_path = write_locan_helper(
        review_dir=out_dir,
        review_table_path=review_table_path,
    )

    summary = {
        "created_at": now_iso(),
        "stage": "napari_locan_review",
        "input": str(input_path),
        "input_root": str(input_root),
        "localization_csv": str(localization_csv),
        "detected_input_format": detected_format,
        "out_dir": str(out_dir),
        "review_dir": str(out_dir),
        "movie_path_inferred": str(movie_path_obj) if movie_path_obj else None,
        "coord_units_input": coord_units,
        "analysis_units": analysis_units,
        "pixel_size_nm": pixel_size_nm,
        "n_input_rows": int(len(raw_df)),
        "n_standardized_before_filter": int(len(standardized_raw)),
        "n_localizations": int(len(standardized)),
        "filter_report": filter_report,
        "review_table": str(review_table_path),
        "napari_helper_script": str(napari_helper_path),
        "locan_helper_script": str(locan_helper_path),
        "locan_review_enabled": bool(locan_review),
        "napari_opened": bool(open_napari_viewer),
        "napari_color_by": napari_color_by,
        "quality_flags": quality_flags,
        "locan_summary": locan_summary,
    }

    summary_path = out_dir / "napari_locan_review_summary.json"
    write_json(summary, summary_path)

    if open_napari_viewer:
        open_in_napari(
            standardized=standardized,
            movie_path=movie_path_obj,
            units=analysis_units,
            pixel_size_nm=pixel_size_nm,
            use_3d=napari_3d,
            point_size=point_size,
            max_points=max_napari_points,
            color_by=napari_color_by,
        )

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional napari + Locan-style scientific review for SMLM pipeline outputs."
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input batch directory, run directory, or localization CSV. "
            "Example: results/run_001/batches/0001_movie"
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output review directory.",
    )

    parser.add_argument(
        "--coord-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Units of the input localization coordinates.",
    )

    parser.add_argument(
        "--analysis-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Units used for analysis and napari points.",
    )

    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        default=None,
        help="Pixel size in nm, needed for nm <-> pixel conversion.",
    )

    parser.add_argument(
        "--locan-review",
        action="store_true",
        help="Generate Locan-style review summary and plots.",
    )

    parser.add_argument(
        "--open-napari",
        action="store_true",
        help="Open napari GUI viewer. Raw movie overlay is inferred from manifest if possible.",
    )

    parser.add_argument(
        "--napari-3d",
        action="store_true",
        help="Use z/y/x points in napari when z is available.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="napari point size.",
    )

    parser.add_argument(
        "--max-napari-points",
        type=int,
        default=300_000,
        help="Maximum number of points sent to napari.",
    )

    parser.add_argument(
        "--napari-color-by",
        default=None,
        help="Optional feature column for napari point coloring, e.g. confidence, frame, intensity.",
    )

    parser.add_argument(
        "--render-bin-size",
        type=float,
        default=20.0,
        help="XY render bin size in analysis units.",
    )

    parser.add_argument(
        "--max-nn-points",
        type=int,
        default=100_000,
        help="Maximum number of points used for nearest-neighbor QC.",
    )

    parser.add_argument(
        "--drift-frame-bin",
        type=int,
        default=100,
        help="Frame bin size for simple drift-proxy plot.",
    )

    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-confidence", type=float, default=None)
    parser.add_argument("--min-intensity", type=float, default=None)
    parser.add_argument("--max-intensity", type=float, default=None)
    parser.add_argument("--min-frame", type=int, default=None)
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)

    parser.add_argument(
        "--dbscan",
        action="store_true",
        help="Run optional DBSCAN cluster analysis if scikit-learn is installed.",
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=50.0,
        help="DBSCAN eps radius in analysis units.",
    )

    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=10,
        help="DBSCAN minimum samples.",
    )

    parser.add_argument(
        "--dbscan-max-points",
        type=int,
        default=200_000,
        help="Maximum points used for DBSCAN.",
    )

    parser.add_argument(
        "--dbscan-3d",
        action="store_true",
        help="Use x/y/z for DBSCAN if z is available.",
    )

    parser.add_argument(
        "--ripley",
        action="store_true",
        help="Run approximate 2D Ripley L proxy if scipy is installed.",
    )

    parser.add_argument(
        "--ripley-max-radius",
        type=float,
        default=500.0,
        help="Maximum radius for Ripley L proxy.",
    )

    parser.add_argument(
        "--ripley-n-radii",
        type=int,
        default=50,
        help="Number of radii for Ripley L proxy.",
    )

    parser.add_argument(
        "--ripley-max-points",
        type=int,
        default=20_000,
        help="Maximum points used for Ripley L proxy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("napari + Locan-style scientific review")
    print("=" * 70)
    print(f"Input:          {args.input}")
    print(f"Output:         {args.out}")
    print(f"Coord units:    {args.coord_units}")
    print(f"Analysis units: {args.analysis_units}")
    print(f"Pixel size nm:  {args.pixel_size_nm}")
    print("=" * 70)

    summary = run_review(
        input_path=args.input,
        out_dir=args.out,
        coord_units=args.coord_units,
        analysis_units=args.analysis_units,
        pixel_size_nm=args.pixel_size_nm,
        locan_review=args.locan_review,
        open_napari_viewer=args.open_napari,
        napari_3d=args.napari_3d,
        point_size=args.point_size,
        max_napari_points=args.max_napari_points,
        napari_color_by=args.napari_color_by,
        render_bin_size=args.render_bin_size,
        max_nn_points=args.max_nn_points,
        drift_frame_bin=args.drift_frame_bin,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        dbscan=args.dbscan,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_max_points=args.dbscan_max_points,
        dbscan_3d=args.dbscan_3d,
        ripley=args.ripley,
        ripley_max_radius=args.ripley_max_radius,
        ripley_n_radii=args.ripley_n_radii,
        ripley_max_points=args.ripley_max_points,
    )

    print("[review] Saved:")
    print(f"  - review table: {display_path(summary['review_table'])}")
    print(f"  - napari helper: {display_path(summary['napari_helper_script'])}")
    print(f"  - locan helper: {display_path(summary['locan_helper_script'])}")
    print(f"  - summary: {display_path(Path(summary['review_dir']) / 'napari_locan_review_summary.json')}")

    inferred_movie = summary.get("movie_path_inferred")

    if inferred_movie:
        print(f"  - inferred movie overlay: {display_path(inferred_movie)}")
    else:
        print("  - inferred movie overlay: none")

    if summary["locan_review_enabled"]:
        artifacts = summary.get("locan_summary", {}).get("artifacts", {})

        for name, path in artifacts.items():
            print(f"  - {name}: {display_path(path)}")

        dbscan_report = summary.get("locan_summary", {}).get("dbscan", {})

        if dbscan_report.get("enabled"):
            print(f"  - dbscan clustered points: {display_path(dbscan_report.get('path_clustered_points'))}")
            print(f"  - dbscan cluster summary: {display_path(dbscan_report.get('path_cluster_summary'))}")

        ripley_report = summary.get("locan_summary", {}).get("ripley_l_proxy", {})

        if ripley_report.get("enabled"):
            print(f"  - ripley table: {display_path(ripley_report.get('table'))}")
            print(f"  - ripley plot: {display_path(ripley_report.get('plot'))}")

    flags = summary.get("quality_flags", [])

    if flags:
        print("[review] Quality flags:")

        for flag in flags:
            print(
                f"  - {flag.get('level', '').upper()} | "
                f"{flag.get('code', '')}: {flag.get('message', '')}"
            )

    print(f"[review] Localizations: {summary['n_localizations']}")
    print("[review] Done.")


if __name__ == "__main__":
    main()