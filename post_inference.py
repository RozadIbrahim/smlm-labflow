#!/usr/bin/env python3
"""
post_inference.py

Complete post-inference stage for the SMLM wrapper pipeline.

This script replaces convert_to_canonical.py and any older post_inference.py.

It performs:

1. Backend/localization CSV -> canonical_localizations.csv
2. Canonical conversion report
3. Localization QC JSON
4. QC plots
5. Tool-specific downstream exports:
   - SMAP-adapted CSV      -> exports/smap/smap_localizations.csv
   - Picasso-adapted CSV   -> exports/picasso/picasso_localizations.csv
   - napari points CSV     -> exports/napari/napari_points.csv
   - Locan-style CSV       -> exports/locan/locan_localizations.csv
   - generic SMLM CSV      -> exports/generic/smlm_generic_localizations.csv
   - untouched backend CSV -> exports/vanilla/<raw_backend_output_name>.csv
6. Post-inference summary JSON

Standard usage:

    python post_inference.py \
        --input results/run_001/batches/0001_movie/liteloc_raw_output.csv \
        --out results/run_001/batches/0001_movie \
        --profile profiles/dna_paint_standard.yaml \
        --backend liteloc \
        --pixel-size-nm 65 \
        --coord-units nm

Important idea:

    canonical CSV is the internal truth table.
    SMAP/Picasso/napari/Locan exports are separate adapted formats.

Coordinate convention:

    --coord-units nm:
        input/canonical x, y, z are treated as nanometers.

    --coord-units pixel:
        input/canonical x, y, z are treated as camera pixels.

    --coord-units auto:
        infer from coordinate range.
        Large x/y ranges usually mean nm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONCRETE_EXPORT_CHOICES = {"raw", "generic", "smap", "picasso", "napari", "locan"}
EXPORT_CHOICES = CONCRETE_EXPORT_CHOICES | {"all", "none"}
EXPORT_ALIASES = {
    "backend": "raw",
    "backend_raw": "raw",
    "vanilla": "raw",
    "vanilla_backend": "raw",
}


# =============================================================================
# Canonical schema
# =============================================================================

DEFAULT_CANONICAL_COLUMNS = [
    "batch_index",
    "run_id",
    "input_name",
    "input_path",
    "frame",
    "x",
    "y",
    "z",
    "photons",
    "background",
    "confidence",
    "backend",
    "source_file",
]

try:
    from schema import CANONICAL_COLUMNS as SCHEMA_CANONICAL_COLUMNS
except Exception:
    SCHEMA_CANONICAL_COLUMNS = DEFAULT_CANONICAL_COLUMNS


def ensure_columns(base_columns: list[str], extra_columns: list[str]) -> list[str]:
    cols = list(base_columns)

    for col in extra_columns:
        if col not in cols:
            if "backend" in cols:
                cols.insert(cols.index("backend"), col)
            else:
                cols.append(col)

    return cols


POST_INFERENCE_COLUMNS = ensure_columns(
    list(SCHEMA_CANONICAL_COLUMNS),
    ["lpx", "lpy", "lpz"],
)


COLUMN_ALIASES = {
    "batch_index": ["batch_index", "batch", "batch_id", "batch_idx"],
    "run_id": ["run_id", "run", "run_name"],
    "input_name": ["input_name", "movie_name", "file_name", "filename", "name"],
    "input_path": ["input_path", "movie_path", "file_path", "filepath", "path"],
    "frame": [
        "frame",
        "frame_idx",
        "frame_index",
        "frame_id",
        "nframe",
        "t",
        "time",
        "image",
        "image_id",
        "image_index",
        "img",
        "img_index",
        "slice",
        "slice_idx",
    ],
    "x": [
        "x",
        "x_nm",
        "x[nm]",
        "xnm",
        "x_nm_",
        "x_position",
        "x_pos",
        "xpos",
        "x_est",
        "xrec",
        "xpix",
        "x_pix",
        "x_pixel",
        "x_px",
        "xcoord",
        "x_coordinate",
        "xlocal",
        "x_local",
    ],
    "y": [
        "y",
        "y_nm",
        "y[nm]",
        "ynm",
        "y_nm_",
        "y_position",
        "y_pos",
        "ypos",
        "y_est",
        "yrec",
        "ypix",
        "y_pix",
        "y_pixel",
        "y_px",
        "ycoord",
        "y_coordinate",
        "ylocal",
        "y_local",
    ],
    "z": [
        "z",
        "z_nm",
        "z[nm]",
        "znm",
        "z_nm_",
        "z_position",
        "z_pos",
        "zpos",
        "z_est",
        "zrec",
        "zcoord",
        "z_coordinate",
        "zlocal",
        "z_local",
    ],
    "photons": [
        "photons",
        "photon",
        "n_photons",
        "nphotons",
        "intensity",
        "amp",
        "amplitude",
        "signal",
        "brightness",
        "height",
        "i",
        "phot",
    ],
    "background": [
        "background",
        "bg",
        "bkg",
        "bkgd",
        "backg",
        "offset",
        "noise",
        "baseline",
    ],
    "confidence": [
        "confidence",
        "prob",
        "probability",
        "score",
        "detection_score",
        "pred_score",
        "p",
        "likelihood",
        "conf",
    ],
    "lpx": [
        "lpx",
        "sigma_x",
        "sx",
        "x_sigma",
        "x_precision",
        "x_precision_nm",
        "uncertainty_x",
        "loc_precision_x",
        "locprecnm",
        "crlb_x",
    ],
    "lpy": [
        "lpy",
        "sigma_y",
        "sy",
        "y_sigma",
        "y_precision",
        "y_precision_nm",
        "uncertainty_y",
        "loc_precision_y",
        "locprecnm",
        "crlb_y",
    ],
    "lpz": [
        "lpz",
        "sigma_z",
        "sz",
        "z_sigma",
        "z_precision",
        "z_precision_nm",
        "uncertainty_z",
        "loc_precision_z",
        "crlb_z",
    ],
}


# =============================================================================
# Utilities
# =============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def elapsed_minutes_between(start_time: Any, end_time: Any) -> Optional[float]:
    try:
        start_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return None


def timing_record(start_time: str, end_time: str) -> Dict[str, Any]:
    elapsed_min = elapsed_minutes_between(start_time, end_time)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_min": round(elapsed_min, 6) if elapsed_min is not None else None,
    }


def run_timed(
    timing: Dict[str, Any],
    name: str,
    func,
    *args,
    **kwargs,
):
    start_time = now_iso()
    try:
        return func(*args, **kwargs)
    finally:
        end_time = now_iso()
        timing[name] = timing_record(start_time, end_time)


def normalize_export_choices(values: Optional[list[str]]) -> set[str]:
    if not values:
        return {"raw"}

    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip().lower() for part in str(value).split(","))

    normalized = {
        EXPORT_ALIASES.get(token, token)
        for token in tokens
        if token
    }

    invalid = normalized - EXPORT_CHOICES
    if invalid:
        raise ValueError(
            "Invalid --export value(s): "
            f"{', '.join(sorted(invalid))}. "
            f"Expected one of: {', '.join(sorted(EXPORT_CHOICES))}."
        )

    if "none" in normalized and len(normalized) > 1:
        raise ValueError("--export none cannot be combined with other exports.")

    if "none" in normalized:
        return set()

    if "all" in normalized:
        return set(CONCRETE_EXPORT_CHOICES)

    return {value for value in normalized if value in CONCRETE_EXPORT_CHOICES}


def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("[", "_").replace("]", "_")
    name = name.replace("(", "_").replace(")", "_")
    name = name.replace("/", "_")
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profile(profile_path: Optional[str | Path]) -> Dict[str, Any]:
    if profile_path is None:
        return {}

    path = Path(profile_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to read --profile. Install with: pip install pyyaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)

    return profile or {}


def get_nested(data: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur = data

    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)

    return default if cur is None else cur


def infer_pixel_size_nm(
    profile: Dict[str, Any], cli_pixel_size_nm: Optional[float]
) -> Optional[float]:
    if cli_pixel_size_nm is not None:
        return float(cli_pixel_size_nm)

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


def read_input_table(input_path: Path) -> pd.DataFrame:
    errors = []

    readers = [
        lambda p: pd.read_csv(p),
        lambda p: pd.read_csv(p, sep=None, engine="python"),
        lambda p: pd.read_csv(p, sep=r"\s+", engine="python"),
    ]

    for reader in readers:
        try:
            df = reader(input_path)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Could not read localization table: {input_path}\n"
        f"Reader errors:\n" + "\n".join(errors)
    )


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_name(col): col for col in df.columns}


def find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    lookup = build_column_lookup(df)

    aliases = COLUMN_ALIASES.get(canonical_name, [])
    normalized_aliases = [normalize_name(alias) for alias in aliases]

    for alias in normalized_aliases:
        if alias in lookup:
            return lookup[alias]

    for raw_norm, raw_original in lookup.items():
        if raw_norm == normalize_name(canonical_name):
            return raw_original

    return None


def clean_numeric_frame(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.round().astype("Int64")


def infer_coord_units(canonical: pd.DataFrame, requested: str) -> str:
    if requested in {"nm", "pixel"}:
        return requested

    x = pd.to_numeric(canonical.get("x"), errors="coerce").dropna()
    y = pd.to_numeric(canonical.get("y"), errors="coerce").dropna()

    if len(x) == 0 or len(y) == 0:
        return "nm"

    max_xy = max(float(x.max()), float(y.max()))

    # SMLM images in pixels are usually hundreds to a few thousands.
    # Values like 28,000 are much more likely to be nm.
    if max_xy > 4096:
        return "nm"

    return "pixel"


def nm_to_pixel(values: pd.Series, pixel_size_nm: Optional[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    if pixel_size_nm is None or pixel_size_nm <= 0:
        return values

    return values / float(pixel_size_nm)


def pixel_to_nm(values: pd.Series, pixel_size_nm: Optional[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    if pixel_size_nm is None or pixel_size_nm <= 0:
        return values

    return values * float(pixel_size_nm)


def get_axis_series(
    canonical: pd.DataFrame,
    axis: str,
    target_units: str,
    coord_units: str,
    pixel_size_nm: Optional[float],
) -> pd.Series:
    values = pd.to_numeric(canonical[axis], errors="coerce")

    if target_units == coord_units:
        return values

    if target_units == "pixel" and coord_units == "nm":
        return nm_to_pixel(values, pixel_size_nm)

    if target_units == "nm" and coord_units == "pixel":
        return pixel_to_nm(values, pixel_size_nm)

    return values


def get_precision_series(
    canonical: pd.DataFrame,
    precision_col: str,
    target_units: str,
    coord_units: str,
    pixel_size_nm: Optional[float],
    fallback_nm: float,
    fallback_pixel: float,
) -> tuple[pd.Series, bool]:
    if precision_col in canonical.columns and canonical[precision_col].notna().any():
        values = pd.to_numeric(canonical[precision_col], errors="coerce")

        if target_units == coord_units:
            return values, False

        if target_units == "pixel" and coord_units == "nm":
            return nm_to_pixel(values, pixel_size_nm), False

        if target_units == "nm" and coord_units == "pixel":
            return pixel_to_nm(values, pixel_size_nm), False

        return values, False

    if target_units == "nm":
        return pd.Series(fallback_nm, index=canonical.index), True

    return pd.Series(fallback_pixel, index=canonical.index), True


# =============================================================================
# Canonical conversion
# =============================================================================


def convert_inference_to_canonical(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Optional[Dict[str, Any]] = None,
    backend_name: str = "liteloc",
    source_file: Optional[str] = None,
    canonical_name: str = "canonical_localizations.csv",
    drop_missing_xy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Path]:
    profile = profile or {}

    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = out_dir / canonical_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input localization CSV not found: {input_path}")

    input_df = read_input_table(input_path)
    canonical = pd.DataFrame(index=input_df.index)

    conversion_report: Dict[str, Any] = {
        "created_at": now_iso(),
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "canonical_path": str(canonical_path),
        "backend": backend_name,
        "n_input_rows": int(len(input_df)),
        "input_columns": list(input_df.columns),
        "canonical_columns_requested": POST_INFERENCE_COLUMNS,
        "column_mapping": {},
        "missing_columns": [],
        "drop_missing_xy": bool(drop_missing_xy),
    }

    for canonical_col in POST_INFERENCE_COLUMNS:
        if canonical_col == "backend":
            canonical[canonical_col] = backend_name
            conversion_report["column_mapping"][canonical_col] = "__constant_backend__"
            continue

        if canonical_col == "source_file":
            canonical[canonical_col] = source_file or str(input_path)
            conversion_report["column_mapping"][canonical_col] = (
                "__constant_source_file__"
            )
            continue

        input_col = find_column(input_df, canonical_col)

        if input_col is None:
            canonical[canonical_col] = pd.NA
            conversion_report["missing_columns"].append(canonical_col)
            conversion_report["column_mapping"][canonical_col] = None
        else:
            canonical[canonical_col] = input_df[input_col]
            conversion_report["column_mapping"][canonical_col] = input_col

    numeric_cols = [
        "batch_index",
        "frame",
        "x",
        "y",
        "z",
        "photons",
        "background",
        "confidence",
        "lpx",
        "lpy",
        "lpz",
    ]

    for col in numeric_cols:
        if col in canonical.columns:
            canonical[col] = coerce_numeric(canonical[col])

    if "frame" in canonical.columns:
        try:
            canonical["frame"] = clean_numeric_frame(canonical["frame"])
        except Exception:
            pass

    if "batch_index" in canonical.columns:
        try:
            canonical["batch_index"] = clean_numeric_frame(canonical["batch_index"])
        except Exception:
            pass

    n_before_drop = len(canonical)

    if drop_missing_xy and {"x", "y"}.issubset(canonical.columns):
        canonical = canonical.dropna(subset=["x", "y"]).reset_index(drop=True)

    n_after_drop = len(canonical)

    conversion_report["n_output_rows"] = int(len(canonical))
    conversion_report["n_dropped_missing_xy"] = int(n_before_drop - n_after_drop)
    conversion_report["canonical_columns_written"] = list(canonical.columns)

    required_missing = [
        col
        for col in ["frame", "x", "y"]
        if col in conversion_report["missing_columns"]
    ]

    if len(canonical) == 0:
        conversion_report["status"] = "warning_empty_output"
    elif required_missing:
        conversion_report["status"] = "warning_required_columns_missing"
    else:
        conversion_report["status"] = "passed"

    canonical.to_csv(canonical_path, index=False)

    report_path = out_dir / "canonical_conversion_report.json"
    write_json(conversion_report, report_path)

    return canonical, conversion_report, canonical_path


# =============================================================================
# QC
# =============================================================================


def numeric_summary(series: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
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


def build_quality_flags(
    canonical: pd.DataFrame,
    coord_units: str,
    pixel_size_nm: Optional[float],
) -> list[Dict[str, Any]]:
    flags: list[Dict[str, Any]] = []

    if len(canonical) == 0:
        flags.append(
            {
                "level": "error",
                "code": "EMPTY_LOCALIZATION_TABLE",
                "message": "Canonical localization table contains zero rows.",
            }
        )
        return flags

    for col in ["x", "y"]:
        if col not in canonical.columns or canonical[col].dropna().empty:
            flags.append(
                {
                    "level": "error",
                    "code": f"MISSING_{col.upper()}",
                    "message": f"Column {col} is missing or fully empty.",
                }
            )

    if "frame" in canonical.columns and canonical["frame"].dropna().empty:
        flags.append(
            {
                "level": "warning",
                "code": "MISSING_FRAME",
                "message": "Frame column is missing or empty. Temporal QC will be limited.",
            }
        )

    if coord_units == "nm" and pixel_size_nm is None:
        flags.append(
            {
                "level": "warning",
                "code": "MISSING_PIXEL_SIZE_FOR_PIXEL_EXPORTS",
                "message": (
                    "Coordinates look like nm but pixel_size_nm is missing. "
                    "Picasso pixel export cannot be physically correct without it."
                ),
            }
        )

    if "confidence" in canonical.columns:
        conf = pd.to_numeric(canonical["confidence"], errors="coerce").dropna()

        if len(conf) > 0 and conf.max() > 1.5:
            flags.append(
                {
                    "level": "info",
                    "code": "CONFIDENCE_NOT_PROBABILITY_SCALE",
                    "message": "Confidence values exceed 1. This may be a score, not a probability.",
                }
            )

    if "background" in canonical.columns:
        bg = pd.to_numeric(canonical["background"], errors="coerce").dropna()

        if len(bg) == 0:
            flags.append(
                {
                    "level": "info",
                    "code": "BACKGROUND_MISSING",
                    "message": "Background column is empty. Downstream exports will use 0 where needed.",
                }
            )

    return flags


def compute_localization_qc(
    canonical: pd.DataFrame,
    conversion_report: Dict[str, Any],
    coord_units: str,
    pixel_size_nm: Optional[float],
) -> Dict[str, Any]:
    qc: Dict[str, Any] = {
        "created_at": now_iso(),
        "n_localizations": int(len(canonical)),
        "columns": list(canonical.columns),
        "coord_units": coord_units,
        "pixel_size_nm": pixel_size_nm,
        "conversion": conversion_report,
    }

    numeric_cols = [
        "batch_index",
        "frame",
        "x",
        "y",
        "z",
        "photons",
        "background",
        "confidence",
        "lpx",
        "lpy",
        "lpz",
    ]

    qc["numeric_summary"] = {}

    for col in numeric_cols:
        if col in canonical.columns:
            qc["numeric_summary"][col] = numeric_summary(canonical[col])

    if "frame" in canonical.columns and canonical["frame"].notna().any():
        counts = canonical["frame"].value_counts().sort_index()

        qc["localizations_per_frame"] = {
            "n_frames_detected": int(len(counts)),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "std": float(counts.std()) if len(counts) > 1 else 0.0,
            "first_frame": int(counts.index.min()),
            "last_frame": int(counts.index.max()),
        }

    qc["quality_flags"] = build_quality_flags(
        canonical=canonical,
        coord_units=coord_units,
        pixel_size_nm=pixel_size_nm,
    )

    qc["status"] = "passed"

    if any(flag["level"] == "error" for flag in qc["quality_flags"]):
        qc["status"] = "failed"
    elif any(flag["level"] == "warning" for flag in qc["quality_flags"]):
        qc["status"] = "warning"

    return qc


# =============================================================================
# Plots
# =============================================================================


def save_empty_plot(out_path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_xy_preview(
    canonical: pd.DataFrame, out_path: Path, max_points: int = 200_000
) -> None:
    if not {"x", "y"}.issubset(canonical.columns):
        save_empty_plot(out_path, "Localization XY preview", "Missing x/y columns")
        return

    df = canonical[["x", "y"]].dropna()

    if len(df) == 0:
        save_empty_plot(
            out_path, "Localization XY preview", "No valid x/y localizations"
        )
        return

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    plt.figure(figsize=(7, 7))
    plt.scatter(df["x"], df["y"], s=0.2, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Canonical XY preview")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_frame_counts(canonical: pd.DataFrame, out_path: Path) -> None:
    if "frame" not in canonical.columns:
        save_empty_plot(out_path, "Localizations per frame", "Missing frame column")
        return

    frames = canonical["frame"].dropna()

    if len(frames) == 0:
        save_empty_plot(out_path, "Localizations per frame", "No valid frame values")
        return

    counts = frames.value_counts().sort_index()

    plt.figure(figsize=(9, 4))
    plt.plot(counts.index, counts.values, linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Number of localizations")
    plt.title("Localizations per frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_histogram(
    canonical: pd.DataFrame,
    column: str,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
) -> None:
    if column not in canonical.columns:
        save_empty_plot(out_path, title, f"Missing {column} column")
        return

    values = pd.to_numeric(canonical[column], errors="coerce").dropna()

    if len(values) == 0:
        save_empty_plot(out_path, title, f"No valid {column} values")
        return

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_qc_plots(canonical: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    paths = {
        "xy_preview": out_dir / "localization_xy_preview.png",
        "frame_counts": out_dir / "localization_frame_counts.png",
        "confidence_histogram": out_dir / "localization_confidence_histogram.png",
        "photons_histogram": out_dir / "localization_photons_histogram.png",
        "background_histogram": out_dir / "localization_background_histogram.png",
        "z_histogram": out_dir / "localization_z_histogram.png",
    }

    plot_xy_preview(canonical, paths["xy_preview"])
    plot_frame_counts(canonical, paths["frame_counts"])

    plot_histogram(
        canonical,
        "confidence",
        paths["confidence_histogram"],
        "Confidence / probability / score distribution",
        "Confidence / probability / score",
    )

    plot_histogram(
        canonical,
        "photons",
        paths["photons_histogram"],
        "Photon / intensity distribution",
        "Photons / intensity",
    )

    plot_histogram(
        canonical,
        "background",
        paths["background_histogram"],
        "Background distribution",
        "Background",
    )

    plot_histogram(
        canonical,
        "z",
        paths["z_histogram"],
        "Z distribution",
        "z",
    )

    return {key: str(path) for key, path in paths.items()}


# =============================================================================
# Downstream export adapters
# =============================================================================


def export_smap(
    canonical: pd.DataFrame,
    out_path: Path,
    coord_units: str,
    pixel_size_nm: Optional[float],
    default_locprec_nm: float,
) -> Dict[str, Any]:
    """
    SMAP-adapted CSV.

    SMAP-friendly principle:
        use nanometer coordinate names:
            xnm, ynm, znm

    Also provide:
        frame
        photons
        bg
        locprecnm
        channel
        file
        xpix, ypix when pixel size is known

    This is safer than generic x/y/z for SMAP rendering.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()

    df["frame"] = (
        canonical["frame"].fillna(0).astype(int) if "frame" in canonical.columns else 0
    )

    df["xnm"] = get_axis_series(canonical, "x", "nm", coord_units, pixel_size_nm)
    df["ynm"] = get_axis_series(canonical, "y", "nm", coord_units, pixel_size_nm)

    if "z" in canonical.columns and canonical["z"].notna().any():
        df["znm"] = get_axis_series(canonical, "z", "nm", coord_units, pixel_size_nm)
    else:
        df["znm"] = 0.0

    if "photons" in canonical.columns and canonical["photons"].notna().any():
        df["photons"] = canonical["photons"]
    else:
        df["photons"] = 0.0

    if "background" in canonical.columns and canonical["background"].notna().any():
        df["bg"] = canonical["background"]
    else:
        df["bg"] = 0.0

    lpx_nm, used_placeholder_x = get_precision_series(
        canonical,
        "lpx",
        "nm",
        coord_units,
        pixel_size_nm,
        fallback_nm=default_locprec_nm,
        fallback_pixel=1.0,
    )

    lpy_nm, used_placeholder_y = get_precision_series(
        canonical,
        "lpy",
        "nm",
        coord_units,
        pixel_size_nm,
        fallback_nm=default_locprec_nm,
        fallback_pixel=1.0,
    )

    df["locprecnm"] = pd.concat([lpx_nm, lpy_nm], axis=1).mean(axis=1)

    if "confidence" in canonical.columns and canonical["confidence"].notna().any():
        df["score"] = canonical["confidence"]

    if "batch_index" in canonical.columns and canonical["batch_index"].notna().any():
        df["channel"] = canonical["batch_index"].fillna(1).astype(int)
    else:
        df["channel"] = 1

    if "input_name" in canonical.columns:
        df["file"] = canonical["input_name"].astype(str)
    elif "source_file" in canonical.columns:
        df["file"] = canonical["source_file"].astype(str)
    else:
        df["file"] = "localizations"

    if pixel_size_nm is not None and pixel_size_nm > 0:
        df["xpix"] = df["xnm"] / pixel_size_nm
        df["ypix"] = df["ynm"] / pixel_size_nm

    df = df.dropna(subset=["xnm", "ynm"])
    df.to_csv(out_path, index=False)

    return {
        "enabled": True,
        "tool": "smap",
        "path": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coordinate_units_written": "nm",
        "placeholder_locprec_used": bool(used_placeholder_x or used_placeholder_y),
        "default_locprec_nm": float(default_locprec_nm),
    }


def export_picasso(
    canonical: pd.DataFrame,
    out_path: Path,
    coord_units: str,
    pixel_size_nm: Optional[float],
    default_lpx_px: float,
) -> Dict[str, Any]:
    """
    Picasso-adapted CSV.

    Picasso expects x/y/frame/lpx/lpy.
    x/y/z should be camera pixels, not nm.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()

    df["frame"] = (
        canonical["frame"].fillna(0).astype(int) if "frame" in canonical.columns else 0
    )

    df["x"] = get_axis_series(canonical, "x", "pixel", coord_units, pixel_size_nm)
    df["y"] = get_axis_series(canonical, "y", "pixel", coord_units, pixel_size_nm)

    lpx, used_placeholder_x = get_precision_series(
        canonical,
        "lpx",
        "pixel",
        coord_units,
        pixel_size_nm,
        fallback_nm=20.0,
        fallback_pixel=default_lpx_px,
    )

    lpy, used_placeholder_y = get_precision_series(
        canonical,
        "lpy",
        "pixel",
        coord_units,
        pixel_size_nm,
        fallback_nm=20.0,
        fallback_pixel=default_lpx_px,
    )

    df["lpx"] = lpx
    df["lpy"] = lpy

    if "z" in canonical.columns and canonical["z"].notna().any():
        df["z"] = get_axis_series(canonical, "z", "pixel", coord_units, pixel_size_nm)

    if "lpz" in canonical.columns and canonical["lpz"].notna().any():
        lpz, _ = get_precision_series(
            canonical,
            "lpz",
            "pixel",
            coord_units,
            pixel_size_nm,
            fallback_nm=50.0,
            fallback_pixel=default_lpx_px,
        )
        df["lpz"] = lpz

    if "photons" in canonical.columns and canonical["photons"].notna().any():
        df["photons"] = canonical["photons"]

    if "background" in canonical.columns and canonical["background"].notna().any():
        df["bg"] = canonical["background"]
    else:
        df["bg"] = 0.0

    if "confidence" in canonical.columns and canonical["confidence"].notna().any():
        df["score"] = canonical["confidence"]

    if "batch_index" in canonical.columns and canonical["batch_index"].notna().any():
        df["group"] = canonical["batch_index"].fillna(1).astype(int)

    df = df.dropna(subset=["x", "y"])
    df.to_csv(out_path, index=False)

    return {
        "enabled": True,
        "tool": "picasso",
        "path": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coordinate_units_written": "pixel",
        "placeholder_precision_used": bool(used_placeholder_x or used_placeholder_y),
        "default_lpx_px": float(default_lpx_px),
        "pixel_size_nm": pixel_size_nm,
    }


def export_napari_points(
    canonical: pd.DataFrame,
    out_path: Path,
    coord_units: str,
    pixel_size_nm: Optional[float],
    napari_units: str,
) -> Dict[str, Any]:
    """
    napari-adapted points table.

    napari point coordinates should be a clean coordinate table.
    For 3D SMLM, use:
        axis_0 = z
        axis_1 = y
        axis_2 = x

    For 2D:
        axis_0 = y
        axis_1 = x

    This CSV is easy to load manually or with a tiny napari helper.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_z = "z" in canonical.columns and canonical["z"].notna().any()

    target_units = "nm" if napari_units == "nm" else "pixel"

    df = pd.DataFrame()

    if has_z:
        df["axis_0"] = get_axis_series(
            canonical, "z", target_units, coord_units, pixel_size_nm
        )
        df["axis_1"] = get_axis_series(
            canonical, "y", target_units, coord_units, pixel_size_nm
        )
        df["axis_2"] = get_axis_series(
            canonical, "x", target_units, coord_units, pixel_size_nm
        )
        df["axis_0_name"] = "z"
        df["axis_1_name"] = "y"
        df["axis_2_name"] = "x"
    else:
        df["axis_0"] = get_axis_series(
            canonical, "y", target_units, coord_units, pixel_size_nm
        )
        df["axis_1"] = get_axis_series(
            canonical, "x", target_units, coord_units, pixel_size_nm
        )
        df["axis_0_name"] = "y"
        df["axis_1_name"] = "x"

    feature_cols = [
        "frame",
        "photons",
        "background",
        "confidence",
        "batch_index",
        "run_id",
        "input_name",
        "backend",
    ]

    for col in feature_cols:
        if col in canonical.columns:
            df[col] = canonical[col]

    coordinate_cols = ["axis_0", "axis_1", "axis_2"] if has_z else ["axis_0", "axis_1"]
    df = df.dropna(subset=coordinate_cols)

    df.to_csv(out_path, index=False)

    helper_path = out_path.parent / "load_napari_points_example.py"
    helper_code = f"""\
import pandas as pd
import napari

df = pd.read_csv("{out_path.name}")

coord_cols = {[c for c in coordinate_cols]!r}
points = df[coord_cols].to_numpy()
features = df.drop(columns=coord_cols, errors="ignore")

viewer = napari.Viewer()
viewer.add_points(
    points,
    features=features,
    size=2,
    name="SMLM localizations"
)
napari.run()
"""
    helper_path.write_text(helper_code, encoding="utf-8")

    return {
        "enabled": True,
        "tool": "napari",
        "path": str(out_path),
        "helper_script": str(helper_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coordinate_units_written": target_units,
        "dimension": 3 if has_z else 2,
    }


def export_locan(
    canonical: pd.DataFrame,
    out_path: Path,
    coord_units: str,
    pixel_size_nm: Optional[float],
    locan_units: str,
) -> Dict[str, Any]:
    """
    Locan-style CSV.

    Uses explicit semantic coordinate names:
        position_x
        position_y
        position_z

    By default this writes nm, which is normally the most interpretable
    SMLM coordinate space.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_units = "nm" if locan_units == "nm" else "pixel"

    df = pd.DataFrame()

    df["position_x"] = get_axis_series(
        canonical, "x", target_units, coord_units, pixel_size_nm
    )
    df["position_y"] = get_axis_series(
        canonical, "y", target_units, coord_units, pixel_size_nm
    )

    if "z" in canonical.columns and canonical["z"].notna().any():
        df["position_z"] = get_axis_series(
            canonical, "z", target_units, coord_units, pixel_size_nm
        )

    if "frame" in canonical.columns:
        df["frame"] = canonical["frame"]

    if "photons" in canonical.columns and canonical["photons"].notna().any():
        df["intensity"] = canonical["photons"]

    if "background" in canonical.columns and canonical["background"].notna().any():
        df["background"] = canonical["background"]

    if "confidence" in canonical.columns and canonical["confidence"].notna().any():
        df["confidence"] = canonical["confidence"]

    if "batch_index" in canonical.columns:
        df["channel"] = canonical["batch_index"]

    if "input_name" in canonical.columns:
        df["file"] = canonical["input_name"]

    df = df.dropna(subset=["position_x", "position_y"])
    df.to_csv(out_path, index=False)

    return {
        "enabled": True,
        "tool": "locan",
        "path": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coordinate_units_written": target_units,
    }


def export_generic_smlm(canonical: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_cols = [
        "batch_index",
        "run_id",
        "input_name",
        "input_path",
        "frame",
        "x",
        "y",
        "z",
        "photons",
        "background",
        "confidence",
        "lpx",
        "lpy",
        "lpz",
        "backend",
        "source_file",
    ]

    cols = [c for c in preferred_cols if c in canonical.columns]

    df = canonical[cols].copy()
    df.to_csv(out_path, index=False)

    return {
        "enabled": True,
        "tool": "generic",
        "path": str(out_path),
        "rows": int(len(df)),
        "columns": cols,
    }


def export_vanilla_backend_raw(raw_input_path: str | Path, out_dir: Path) -> Dict[str, Any]:
    """
    Preserve the backend output exactly as it came from inference.

    This is intentionally not parsed, canonicalized, filtered, or column-mapped.
    It gives users a plain "vanilla" artifact for auditing and backend-native
    comparisons.
    """
    raw_path = Path(raw_input_path).expanduser().resolve()
    if not raw_path.exists():
        return {
            "enabled": False,
            "tool": "vanilla_backend",
            "status": "missing_raw_input",
            "path": "",
            "source_path": str(raw_path),
        }

    vanilla_dir = out_dir / "exports" / "vanilla"
    vanilla_dir.mkdir(parents=True, exist_ok=True)
    out_path = vanilla_dir / raw_path.name

    shutil.copy2(raw_path, out_path)

    source_sha = file_sha256(raw_path)
    export_sha = file_sha256(out_path)

    return {
        "enabled": True,
        "tool": "vanilla_backend",
        "status": "passed",
        "path": str(out_path),
        "source_path": str(raw_path),
        "bytes": int(out_path.stat().st_size),
        "source_sha256": source_sha,
        "export_sha256": export_sha,
        "copied_without_modification": source_sha == export_sha,
    }


def run_exports(
    canonical: pd.DataFrame,
    out_dir: Path,
    profile: Dict[str, Any],
    coord_units: str,
    pixel_size_nm: Optional[float],
    default_locprec_nm: float,
    default_lpx_px: float,
    napari_units: str,
    locan_units: str,
    export_smap_enabled: Optional[bool] = None,
    export_picasso_enabled: Optional[bool] = None,
    export_napari_enabled: Optional[bool] = None,
    export_locan_enabled: Optional[bool] = None,
    export_generic_enabled: Optional[bool] = None,
    export_raw_enabled: Optional[bool] = None,
    raw_input_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    downstream = profile.get("downstream", {}) if isinstance(profile, dict) else {}

    if export_smap_enabled is None:
        export_smap_enabled = bool(downstream.get("export_smap", False))

    if export_picasso_enabled is None:
        export_picasso_enabled = bool(downstream.get("export_picasso", False))

    if export_napari_enabled is None:
        export_napari_enabled = bool(downstream.get("export_napari", False))

    if export_locan_enabled is None:
        export_locan_enabled = bool(downstream.get("export_locan", False))

    if export_generic_enabled is None:
        export_generic_enabled = bool(downstream.get("export_generic", False))

    if export_raw_enabled is None:
        export_raw_enabled = bool(downstream.get("export_raw", True))

    report: Dict[str, Any] = {}
    export_timing: Dict[str, Any] = {}

    if export_smap_enabled:
        report["smap"] = run_timed(
            export_timing,
            "smap",
            export_smap,
            canonical=canonical,
            out_path=out_dir / "exports" / "smap" / "smap_localizations.csv",
            coord_units=coord_units,
            pixel_size_nm=pixel_size_nm,
            default_locprec_nm=default_locprec_nm,
        )
        report["smap"]["timing"] = export_timing["smap"]
    else:
        report["smap"] = {"enabled": False}

    if export_picasso_enabled:
        report["picasso"] = run_timed(
            export_timing,
            "picasso",
            export_picasso,
            canonical=canonical,
            out_path=out_dir / "exports" / "picasso" / "picasso_localizations.csv",
            coord_units=coord_units,
            pixel_size_nm=pixel_size_nm,
            default_lpx_px=default_lpx_px,
        )
        report["picasso"]["timing"] = export_timing["picasso"]
    else:
        report["picasso"] = {"enabled": False}

    if export_napari_enabled:
        report["napari"] = run_timed(
            export_timing,
            "napari",
            export_napari_points,
            canonical=canonical,
            out_path=out_dir / "exports" / "napari" / "napari_points.csv",
            coord_units=coord_units,
            pixel_size_nm=pixel_size_nm,
            napari_units=napari_units,
        )
        report["napari"]["timing"] = export_timing["napari"]
    else:
        report["napari"] = {"enabled": False}

    if export_locan_enabled:
        report["locan"] = run_timed(
            export_timing,
            "locan",
            export_locan,
            canonical=canonical,
            out_path=out_dir / "exports" / "locan" / "locan_localizations.csv",
            coord_units=coord_units,
            pixel_size_nm=pixel_size_nm,
            locan_units=locan_units,
        )
        report["locan"]["timing"] = export_timing["locan"]
    else:
        report["locan"] = {"enabled": False}

    if export_generic_enabled:
        report["generic_smlm"] = run_timed(
            export_timing,
            "generic_smlm",
            export_generic_smlm,
            canonical=canonical,
            out_path=out_dir / "exports" / "generic" / "smlm_generic_localizations.csv",
        )
        report["generic_smlm"]["timing"] = export_timing["generic_smlm"]
    else:
        report["generic_smlm"] = {"enabled": False}

    if export_raw_enabled and raw_input_path is not None:
        report["vanilla_backend"] = run_timed(
            export_timing,
            "vanilla_backend",
            export_vanilla_backend_raw,
            raw_input_path=raw_input_path,
            out_dir=out_dir,
        )
        report["vanilla_backend"]["timing"] = export_timing["vanilla_backend"]
    elif export_raw_enabled:
        report["vanilla_backend"] = {
            "enabled": False,
            "tool": "vanilla_backend",
            "status": "missing_raw_input",
        }
    else:
        report["vanilla_backend"] = {
            "enabled": False,
            "tool": "vanilla_backend",
        }

    report["timing"] = export_timing
    return report


# =============================================================================
# Main runner
# =============================================================================


def run_post_inference(
    input_path: str | Path,
    out_dir: str | Path,
    profile: Optional[Dict[str, Any]] = None,
    backend_name: str = "liteloc",
    source_file: Optional[str] = None,
    canonical_name: str = "canonical_localizations.csv",
    coord_units: str = "auto",
    pixel_size_nm: Optional[float] = None,
    default_locprec_nm: float = 20.0,
    default_lpx_px: float = 1.0,
    napari_units: str = "nm",
    locan_units: str = "nm",
    drop_missing_xy: bool = True,
    export_smap_enabled: Optional[bool] = None,
    export_picasso_enabled: Optional[bool] = None,
    export_napari_enabled: Optional[bool] = None,
    export_locan_enabled: Optional[bool] = None,
    export_generic_enabled: Optional[bool] = None,
    export_raw_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    profile = profile or {}
    post_start_time = now_iso()
    timing: Dict[str, Any] = {}

    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical, conversion_report, canonical_path = run_timed(
        timing,
        "canonical_conversion",
        convert_inference_to_canonical,
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        backend_name=backend_name,
        source_file=source_file,
        canonical_name=canonical_name,
        drop_missing_xy=drop_missing_xy,
    )

    resolved_coord_units = infer_coord_units(canonical, coord_units)

    qc = run_timed(
        timing,
        "localization_qc",
        compute_localization_qc,
        canonical=canonical,
        conversion_report=conversion_report,
        coord_units=resolved_coord_units,
        pixel_size_nm=pixel_size_nm,
    )

    plot_report = run_timed(
        timing,
        "qc_plots",
        generate_qc_plots,
        canonical=canonical,
        out_dir=out_dir,
    )

    export_report = run_timed(
        timing,
        "exports_total",
        run_exports,
        canonical=canonical,
        out_dir=out_dir,
        profile=profile,
        coord_units=resolved_coord_units,
        pixel_size_nm=pixel_size_nm,
        default_locprec_nm=default_locprec_nm,
        default_lpx_px=default_lpx_px,
        napari_units=napari_units,
        locan_units=locan_units,
        export_smap_enabled=export_smap_enabled,
        export_picasso_enabled=export_picasso_enabled,
        export_napari_enabled=export_napari_enabled,
        export_locan_enabled=export_locan_enabled,
        export_generic_enabled=export_generic_enabled,
        export_raw_enabled=export_raw_enabled,
        raw_input_path=input_path,
    )
    export_timing = export_report.pop("timing", {})
    timing["exports"] = export_timing

    qc["plots"] = plot_report
    qc["exports"] = export_report

    qc_path = out_dir / "localization_qc.json"
    run_timed(timing, "write_localization_qc_json", write_json, qc, qc_path)

    timing["total_before_summary_write"] = timing_record(post_start_time, now_iso())

    summary = {
        "created_at": now_iso(),
        "stage": "post_inference",
        "status": qc["status"],
        "input": str(input_path),
        "out_dir": str(out_dir),
        "backend": backend_name,
        "canonical_csv": str(canonical_path),
        "canonical_conversion_report": str(
            out_dir / "canonical_conversion_report.json"
        ),
        "localization_qc": str(qc_path),
        "post_inference_summary": str(out_dir / "post_inference_summary.json"),
        "coord_units_detected": resolved_coord_units,
        "pixel_size_nm": pixel_size_nm,
        "plots": plot_report,
        "exports": export_report,
        "timing": timing,
        "export_choices": [
            name
            for name, report in export_report.items()
            if isinstance(report, dict) and report.get("enabled")
        ],
        "n_localizations": int(len(canonical)),
        "quality_flags": qc["quality_flags"],
    }

    summary_path = out_dir / "post_inference_summary.json"
    write_json(summary, summary_path)

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process inference localization output into canonical CSV, "
            "QC reports, plots, and adapted downstream exports."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input localization CSV produced by inference.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output run/batch directory for post-inference artifacts.",
    )

    parser.add_argument(
        "--profile",
        default=None,
        help="Pipeline YAML profile.",
    )

    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name. If omitted, uses backend.name from profile, else 'liteloc'.",
    )

    parser.add_argument(
        "--source-file",
        default=None,
        help="Original movie path/name. Optional.",
    )

    parser.add_argument(
        "--canonical-name",
        default="canonical_localizations.csv",
        help="Name of canonical CSV written inside --out.",
    )

    parser.add_argument(
        "--coord-units",
        choices=["auto", "nm", "pixel"],
        default="auto",
        help="Units of input/canonical x/y/z coordinates.",
    )

    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        default=None,
        help=(
            "Camera pixel size in nm. Required for correct nm<->pixel conversion, "
            "especially Picasso export."
        ),
    )

    parser.add_argument(
        "--default-locprec-nm",
        type=float,
        default=20.0,
        help="Default localization precision in nm for SMAP when lpx/lpy are missing.",
    )

    parser.add_argument(
        "--default-lpx-px",
        type=float,
        default=1.0,
        help="Default localization precision in pixels for Picasso when lpx/lpy are missing.",
    )

    parser.add_argument(
        "--napari-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Coordinate units written to napari_points.csv.",
    )

    parser.add_argument(
        "--locan-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Coordinate units written to locan_localizations.csv.",
    )

    parser.add_argument(
        "--keep-missing-xy",
        action="store_true",
        help="Do not drop rows with missing x/y from canonical output.",
    )

    parser.add_argument(
        "--export",
        action="append",
        default=None,
        metavar="{raw,generic,smap,picasso,napari,locan,all,none}",
        help=(
            "Downstream export to write. Repeat this option or use comma-separated "
            "values. Defaults to raw backend output only."
        ),
    )
    parser.add_argument("--no-smap", action="store_true", help="Disable SMAP export.")
    parser.add_argument(
        "--no-picasso", action="store_true", help="Disable Picasso export."
    )
    parser.add_argument(
        "--no-napari", action="store_true", help="Disable napari export."
    )
    parser.add_argument("--no-locan", action="store_true", help="Disable Locan export.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    profile = load_profile(args.profile)

    backend = args.backend
    if backend is None:
        backend = get_nested(profile, ["backend", "name"], default="liteloc")

    pixel_size_nm = infer_pixel_size_nm(
        profile=profile,
        cli_pixel_size_nm=args.pixel_size_nm,
    )
    export_choices = normalize_export_choices(args.export)
    if args.no_smap:
        export_choices.discard("smap")
    if args.no_picasso:
        export_choices.discard("picasso")
    if args.no_napari:
        export_choices.discard("napari")
    if args.no_locan:
        export_choices.discard("locan")

    print("=" * 70)
    print("Post-inference processing")
    print("=" * 70)
    print(f"Input:         {input_path}")
    print(f"Output:        {out_dir}")
    print(f"Profile:       {args.profile if args.profile else 'None'}")
    print(f"Backend:       {backend}")
    print(f"Coord units:   {args.coord_units}")
    print(f"Pixel size nm: {pixel_size_nm}")
    print(f"Exports:       {', '.join(sorted(export_choices)) if export_choices else 'none'}")
    print("=" * 70)

    summary = run_post_inference(
        input_path=input_path,
        out_dir=out_dir,
        profile=profile,
        backend_name=backend,
        source_file=args.source_file,
        canonical_name=args.canonical_name,
        coord_units=args.coord_units,
        pixel_size_nm=pixel_size_nm,
        default_locprec_nm=args.default_locprec_nm,
        default_lpx_px=args.default_lpx_px,
        napari_units=args.napari_units,
        locan_units=args.locan_units,
        drop_missing_xy=not args.keep_missing_xy,
        export_smap_enabled="smap" in export_choices,
        export_picasso_enabled="picasso" in export_choices,
        export_napari_enabled="napari" in export_choices,
        export_locan_enabled="locan" in export_choices,
        export_generic_enabled="generic" in export_choices,
        export_raw_enabled="raw" in export_choices,
    )

    print("[post] Saved:")
    print(f"  - {summary['canonical_csv']}")
    print(f"  - {summary['canonical_conversion_report']}")
    print(f"  - {summary['localization_qc']}")
    print(f"  - {summary['post_inference_summary']}")

    print("[post] Exports:")
    for name, report in summary["exports"].items():
        if report.get("enabled"):
            print(f"  - {name}: {report.get('path')}")

    if summary["quality_flags"]:
        print("[post] Quality flags:")
        for flag in summary["quality_flags"]:
            print(f"  - {flag['level'].upper()} | {flag['code']}: {flag['message']}")

    print(f"[post] Status: {summary['status']}")
    print(f"[post] Localizations: {summary['n_localizations']}")
    print(f"[post] Coordinate units detected: {summary['coord_units_detected']}")
    print("[post] Done.")


if __name__ == "__main__":
    main()
