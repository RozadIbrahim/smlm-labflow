#!/usr/bin/env python3
"""
napari_locan_review.py

Optional downstream review stage for the SMLM wrapper pipeline.

Purpose:
    Use the post_inference outputs for:
        1. napari visualization
        2. Locan-style localization analysis
        3. review plots and summary JSON

This script is designed to run after post_inference.py.

Typical inputs:
    - canonical_localizations.csv
    - exports/locan/locan_localizations.csv
    - exports/napari/napari_points.csv

Typical usage:

    python napari_locan_review.py \
        --run results/run_001/batches/0001_movie \
        --coord-units nm \
        --pixel-size-nm 65 \
        --locan-review

Open napari too:

    python napari_locan_review.py \
        --run results/run_001/batches/0001_movie \
        --movie /path/to/movie.tif \
        --coord-units nm \
        --pixel-size-nm 65 \
        --locan-review \
        --open-napari

Important:
    This script should NOT be forced in the main automated pipeline by default.
    napari is GUI-based and can block execution. Use it as an optional review step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        print(f"[warning] CSV is empty: {path}")

    return df


def find_default_localization_file(run_dir: Path) -> Path:
    """
    Find the best available localization file in a post_inference output folder.

    Priority:
        1. Locan adapted export
        2. canonical_localizations.csv
        3. generic SMLM export
        4. napari points export
    """
    candidates = [
        run_dir / "exports" / "locan" / "locan_localizations.csv",
        run_dir / "canonical_localizations.csv",
        run_dir / "exports" / "generic" / "smlm_generic_localizations.csv",
        run_dir / "exports" / "napari" / "napari_points.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No localization file found. Expected one of:\n"
        + "\n".join(str(c) for c in candidates)
    )


def detect_input_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)

    if {"position_x", "position_y"}.issubset(cols):
        return "locan"

    if {"x", "y"}.issubset(cols):
        return "canonical"

    if {"axis_0", "axis_1"}.issubset(cols):
        return "napari_points"

    raise ValueError(
        "Could not detect localization table format. "
        "Expected either position_x/position_y, x/y, or axis_0/axis_1 columns."
    )


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
        return values / pixel_size_nm

    if from_units == "pixel" and to_units == "nm":
        return values * pixel_size_nm

    return values


def standardize_localizations(
    df: pd.DataFrame,
    coord_units: str,
    target_units: str,
    pixel_size_nm: Optional[float],
) -> Tuple[pd.DataFrame, str]:
    """
    Convert supported localization table formats into a common table:

        position_x
        position_y
        position_z, optional
        frame, optional
        intensity, optional
        background, optional
        confidence, optional

    Returns:
        standardized_df, detected_format
    """
    detected_format = detect_input_format(df)

    out = pd.DataFrame(index=df.index)

    if detected_format == "locan":
        out["position_x"] = convert_units(
            df["position_x"], coord_units, target_units, pixel_size_nm
        )
        out["position_y"] = convert_units(
            df["position_y"], coord_units, target_units, pixel_size_nm
        )

        if "position_z" in df.columns:
            out["position_z"] = convert_units(
                df["position_z"], coord_units, target_units, pixel_size_nm
            )

    elif detected_format == "canonical":
        out["position_x"] = convert_units(
            df["x"], coord_units, target_units, pixel_size_nm
        )
        out["position_y"] = convert_units(
            df["y"], coord_units, target_units, pixel_size_nm
        )

        if "z" in df.columns:
            z = numeric(df["z"])
            if z.notna().any():
                out["position_z"] = convert_units(
                    df["z"], coord_units, target_units, pixel_size_nm
                )

    elif detected_format == "napari_points":
        # napari export convention from post_inference.py:
        #   2D: axis_0 = y, axis_1 = x
        #   3D: axis_0 = z, axis_1 = y, axis_2 = x
        if "axis_2" in df.columns:
            out["position_z"] = numeric(df["axis_0"])
            out["position_y"] = numeric(df["axis_1"])
            out["position_x"] = numeric(df["axis_2"])
        else:
            out["position_y"] = numeric(df["axis_0"])
            out["position_x"] = numeric(df["axis_1"])

    optional_map = {
        "frame": ["frame"],
        "intensity": ["intensity", "photons"],
        "background": ["background", "bg"],
        "confidence": ["confidence", "score"],
        "channel": ["channel", "batch_index", "group"],
        "file": ["file", "input_name", "source_file"],
        "backend": ["backend"],
    }

    for output_col, candidates in optional_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                out[output_col] = df[candidate]
                break

    out = out.dropna(subset=["position_x", "position_y"]).reset_index(drop=True)

    return out, detected_format


# =============================================================================
# Locan-style review analysis
# =============================================================================

def summarize_positions(df: pd.DataFrame, units: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "units": units,
        "n_localizations": int(len(df)),
        "has_z": "position_z" in df.columns and df["position_z"].notna().any(),
    }

    if len(df) == 0:
        return summary

    for col in ["position_x", "position_y", "position_z"]:
        if col in df.columns:
            values = numeric(df[col]).dropna()

            if len(values) > 0:
                summary[col] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                }

    x = numeric(df["position_x"]).dropna()
    y = numeric(df["position_y"]).dropna()

    if len(x) > 0 and len(y) > 0:
        width = float(x.max() - x.min())
        height = float(y.max() - y.min())
        area = width * height

        summary["bounding_box"] = {
            "width": width,
            "height": height,
            "area": area,
        }

        if area > 0:
            summary["density_per_unit2"] = float(len(df) / area)

    if "frame" in df.columns:
        frames = numeric(df["frame"]).dropna()

        if len(frames) > 0:
            summary["frames"] = {
                "first": int(frames.min()),
                "last": int(frames.max()),
                "n_unique": int(frames.nunique()),
            }

    for col in ["intensity", "background", "confidence"]:
        if col in df.columns:
            values = numeric(df[col]).dropna()

            if len(values) > 0:
                summary[col] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                }

    return summary


def plot_xy_render(
    df: pd.DataFrame,
    out_path: Path,
    bin_size: float,
    units: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        save_empty_plot(out_path, "XY render", "No localizations")
        return

    x = numeric(df["position_x"]).dropna()
    y = numeric(df["position_y"]).dropna()

    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(valid) == 0:
        save_empty_plot(out_path, "XY render", "No valid x/y coordinates")
        return

    x_min, x_max = float(valid["x"].min()), float(valid["x"].max())
    y_min, y_max = float(valid["y"].min()), float(valid["y"].max())

    if x_max <= x_min or y_max <= y_min:
        save_empty_plot(out_path, "XY render", "Degenerate x/y range")
        return

    x_bins = max(10, int(np.ceil((x_max - x_min) / bin_size)))
    y_bins = max(10, int(np.ceil((y_max - y_min) / bin_size)))

    hist, x_edges, y_edges = np.histogram2d(
        valid["x"],
        valid["y"],
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
    plt.title(f"SMLM XY render, bin={bin_size:g} {units}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_frame_counts(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "frame" not in df.columns:
        save_empty_plot(out_path, "Localizations per frame", "No frame column")
        return

    frames = numeric(df["frame"]).dropna()

    if len(frames) == 0:
        save_empty_plot(out_path, "Localizations per frame", "No valid frame values")
        return

    counts = frames.astype(int).value_counts().sort_index()

    plt.figure(figsize=(9, 4))
    plt.plot(counts.index, counts.values, linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Localizations")
    plt.title("Localizations per frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confidence_histogram(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "confidence" not in df.columns:
        save_empty_plot(out_path, "Confidence histogram", "No confidence column")
        return

    values = numeric(df["confidence"]).dropna()

    if len(values) == 0:
        save_empty_plot(out_path, "Confidence histogram", "No valid confidence values")
        return

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=60)
    plt.xlabel("Confidence / score")
    plt.ylabel("Count")
    plt.title("Confidence distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_nearest_neighbor_histogram(
    df: pd.DataFrame,
    out_path: Path,
    units: str,
    max_points: int,
) -> Dict[str, Any]:
    """
    Approximate nearest-neighbor QC.

    Uses scipy if available. If scipy is missing, skips gracefully.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
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
    if "position_z" in df.columns and df["position_z"].notna().any():
        cols = ["position_x", "position_y", "position_z"]

    points_df = df[cols].dropna()

    if len(points_df) < 3:
        save_empty_plot(
            out_path,
            "Nearest-neighbor distance",
            "Too few points",
        )
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
        "units": units,
        "mean": float(np.mean(nn)),
        "median": float(np.median(nn)),
        "p05": float(np.quantile(nn, 0.05)),
        "p95": float(np.quantile(nn, 0.95)),
    })

    return report


def save_empty_plot(out_path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_locan_review(
    standardized: pd.DataFrame,
    review_dir: Path,
    units: str,
    render_bin_size: float,
    max_nn_points: int,
) -> Dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_positions(standardized, units=units)

    render_path = review_dir / "locan_xy_render.png"
    frame_path = review_dir / "locan_frame_counts.png"
    confidence_path = review_dir / "locan_confidence_histogram.png"
    nn_path = review_dir / "locan_nearest_neighbor_histogram.png"

    plot_xy_render(
        df=standardized,
        out_path=render_path,
        bin_size=render_bin_size,
        units=units,
    )

    plot_frame_counts(
        df=standardized,
        out_path=frame_path,
    )

    plot_confidence_histogram(
        df=standardized,
        out_path=confidence_path,
    )

    nn_report = plot_nearest_neighbor_histogram(
        df=standardized,
        out_path=nn_path,
        units=units,
        max_points=max_nn_points,
    )

    standardized_path = review_dir / "locan_review_table.csv"
    standardized.to_csv(standardized_path, index=False)

    summary["nearest_neighbor"] = nn_report
    summary["artifacts"] = {
        "review_table": str(standardized_path),
        "xy_render": str(render_path),
        "frame_counts": str(frame_path),
        "confidence_histogram": str(confidence_path),
        "nearest_neighbor_histogram": str(nn_path),
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
    """
    Load a TIFF/OME-TIFF movie and return a 2D max projection.

    Keeps napari overlay simple:
        image layer = 2D projection
        points layer = 2D y/x or 3D z/y/x if requested without image overlay
    """
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

    # Common cases:
    #   YX
    #   TYX
    #   CYX
    #   TCYX or CZYX-like shapes
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Treat first dimension as frame/channel and project.
        return np.max(arr, axis=0)

    if arr.ndim >= 4:
        # Very conservative: project all leading dimensions, keep last two as YX.
        leading_axes = tuple(range(arr.ndim - 2))
        return np.max(arr, axis=leading_axes)

    return arr


def create_napari_points(
    standardized: pd.DataFrame,
    units: str,
    use_3d: bool,
    max_points: int,
) -> Tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Build napari points and features table.

    2D convention:
        points[:, 0] = y
        points[:, 1] = x

    3D convention:
        points[:, 0] = z
        points[:, 1] = y
        points[:, 2] = x
    """
    df = standardized.copy()

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).reset_index(drop=True)

    has_z = "position_z" in df.columns and df["position_z"].notna().any()

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
) -> None:
    """
    Open localization points in napari.

    If movie_path is provided, displays a 2D max projection underneath.
    If units are nm and pixel_size_nm is provided, image scale is set so
    the nm-coordinate points overlay correctly on the projection.
    """
    try:
        import napari
    except Exception as exc:
        raise ImportError(
            "napari is not installed in this environment. "
            "Use your napari_locan.yml environment or install napari first."
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

    viewer.add_points(
        points,
        features=features,
        size=point_size,
        name="SMLM localizations",
    )

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
) -> Path:
    helper_path = review_dir / "open_review_in_napari.py"

    movie_line = f"movie_path = r'{movie_path}'" if movie_path else "movie_path = None"

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

has_z = "position_z" in df.columns and df["position_z"].notna().any()

if has_z:
    points = df[["position_z", "position_y", "position_x"]].to_numpy()
else:
    points = df[["position_y", "position_x"]].to_numpy()

features = df.drop(
    columns=["position_x", "position_y", "position_z"],
    errors="ignore",
)

viewer.add_points(
    points,
    features=features,
    size=2,
    name="SMLM localizations",
)

napari.run()
'''

    helper_path.write_text(helper_code, encoding="utf-8")

    return helper_path


# =============================================================================
# Main review runner
# =============================================================================

def run_review(
    run_dir: Optional[str | Path] = None,
    input_csv: Optional[str | Path] = None,
    movie_path: Optional[str | Path] = None,
    coord_units: str = "nm",
    analysis_units: str = "nm",
    pixel_size_nm: Optional[float] = None,
    locan_review: bool = True,
    open_napari_viewer: bool = False,
    napari_3d: bool = False,
    point_size: float = 2.0,
    max_napari_points: int = 300_000,
    render_bin_size: float = 20.0,
    max_nn_points: int = 100_000,
) -> Dict[str, Any]:
    """
    Importable function for run_pipeline.py.

    Use this only when optional review is requested.
    """
    if run_dir is None and input_csv is None:
        raise ValueError("Either run_dir or input_csv must be provided.")

    run_dir_path = Path(run_dir).expanduser().resolve() if run_dir else None

    if input_csv is None:
        input_path = find_default_localization_file(run_dir_path)
    else:
        input_path = Path(input_csv).expanduser().resolve()

    if run_dir_path is None:
        run_dir_path = input_path.parent

    review_dir = run_dir_path / "review" / "napari_locan"
    review_dir.mkdir(parents=True, exist_ok=True)

    movie_path_obj = Path(movie_path).expanduser().resolve() if movie_path else None

    raw_df = read_csv_checked(input_path)

    standardized, detected_format = standardize_localizations(
        df=raw_df,
        coord_units=coord_units,
        target_units=analysis_units,
        pixel_size_nm=pixel_size_nm,
    )

    review_table_path = review_dir / "locan_review_table.csv"
    standardized.to_csv(review_table_path, index=False)

    locan_summary: Dict[str, Any] = {}

    if locan_review:
        locan_summary = run_locan_review(
            standardized=standardized,
            review_dir=review_dir,
            units=analysis_units,
            render_bin_size=render_bin_size,
            max_nn_points=max_nn_points,
        )

    helper_path = write_napari_helper(
        review_dir=review_dir,
        review_table_path=review_table_path,
        movie_path=movie_path_obj,
        units=analysis_units,
        pixel_size_nm=pixel_size_nm,
    )

    summary = {
        "created_at": now_iso(),
        "stage": "napari_locan_review",
        "input_csv": str(input_path),
        "detected_input_format": detected_format,
        "run_dir": str(run_dir_path),
        "review_dir": str(review_dir),
        "movie_path": str(movie_path_obj) if movie_path_obj else None,
        "coord_units_input": coord_units,
        "analysis_units": analysis_units,
        "pixel_size_nm": pixel_size_nm,
        "n_localizations": int(len(standardized)),
        "review_table": str(review_table_path),
        "napari_helper_script": str(helper_path),
        "locan_review_enabled": bool(locan_review),
        "napari_opened": bool(open_napari_viewer),
        "locan_summary": locan_summary,
    }

    summary_path = review_dir / "napari_locan_review_summary.json"
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
        )

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional napari + Locan-style review for SMLM pipeline outputs."
    )

    parser.add_argument(
        "--run",
        default=None,
        help="Run or batch directory containing post_inference outputs.",
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Specific localization CSV to review. Overrides --run auto-detection.",
    )

    parser.add_argument(
        "--movie",
        default=None,
        help="Optional raw TIFF/OME-TIFF movie for napari max-projection overlay.",
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
        help="Units used for Locan-style analysis and napari points.",
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
        help="Open napari GUI viewer.",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("napari + Locan-style review")
    print("=" * 70)
    print(f"Run dir:        {args.run}")
    print(f"Input CSV:      {args.input}")
    print(f"Movie:          {args.movie}")
    print(f"Coord units:    {args.coord_units}")
    print(f"Analysis units: {args.analysis_units}")
    print(f"Pixel size nm:  {args.pixel_size_nm}")
    print("=" * 70)

    summary = run_review(
        run_dir=args.run,
        input_csv=args.input,
        movie_path=args.movie,
        coord_units=args.coord_units,
        analysis_units=args.analysis_units,
        pixel_size_nm=args.pixel_size_nm,
        locan_review=args.locan_review,
        open_napari_viewer=args.open_napari,
        napari_3d=args.napari_3d,
        point_size=args.point_size,
        max_napari_points=args.max_napari_points,
        render_bin_size=args.render_bin_size,
        max_nn_points=args.max_nn_points,
    )

    print("[review] Saved:")
    print(f"  - review table: {summary['review_table']}")
    print(f"  - helper script: {summary['napari_helper_script']}")
    print(f"  - summary: {summary['review_dir']}/napari_locan_review_summary.json")

    if summary["locan_review_enabled"]:
        artifacts = summary.get("locan_summary", {}).get("artifacts", {})
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")

    print(f"[review] Localizations: {summary['n_localizations']}")
    print("[review] Done.")


if __name__ == "__main__":
    main()