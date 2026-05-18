#!/usr/bin/env python3
"""
liteloc_secondary_tools.py

One safe wrapper to expose LiteLoc secondary utilities for a lab-facing SMLM pipeline.

It does NOT replace LiteLoc inference. It runs around it:
- inventory/discover LiteLoc helper modules
- TIFF QC with optional LiteLoc helper_utils usage
- localization CSV QC/rendering/FRC/grid artefact checks
- prediction-vs-ground-truth evaluation
- infer YAML generation through LiteLoc helper_utils when available
- ADU-to-photon conversion through LiteLoc helper_utils when available

Typical use after inference:
python liteloc_secondary_tools.py all \
  --liteloc-root /path/to/LiteLoc \
  --input-movie movie.tif \
  --localizations results/run_001/canonical_localizations.csv \
  --out results/run_001/liteloc_secondary
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tifffile import TiffFile

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def safe_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def add_liteloc_to_syspath(liteloc_root: Optional[Path]) -> Optional[Path]:
    if liteloc_root is None:
        env = os.environ.get("LITELOC_ROOT")
        liteloc_root = Path(env).expanduser().resolve() if env else None
    else:
        liteloc_root = liteloc_root.expanduser().resolve()

    if liteloc_root is None:
        return None
    if not liteloc_root.exists():
        raise FileNotFoundError(f"LiteLoc root not found: {liteloc_root}")
    if str(liteloc_root) not in sys.path:
        sys.path.insert(0, str(liteloc_root))
    return liteloc_root


def try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def try_get_liteloc_function(
    liteloc_root: Optional[Path], module_name: str, func_name: str
) -> Optional[Any]:
    add_liteloc_to_syspath(liteloc_root)
    module = try_import(module_name)
    if module is None:
        return None
    return getattr(module, func_name, None)


def call_optional_function(
    func: Optional[Any], *args: Any, **kwargs: Any
) -> Tuple[bool, Any, Optional[str]]:
    if func is None:
        return False, None, "function_not_found"
    try:
        return True, func(*args, **kwargs), None
    except Exception:
        return False, None, traceback.format_exc(limit=5)


def normalize_image_for_png(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = np.nanpercentile(arr, [1, 99.7])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def save_image_png(image: np.ndarray, path: Path, title: Optional[str] = None) -> Path:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 7), dpi=160)
    plt.imshow(normalize_image_for_png(image), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return path


def save_hist(
    values: np.ndarray, path: Path, title: str, xlabel: str, bins: int = 80
) -> Path:
    ensure_dir(path.parent)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    plt.figure(figsize=(7, 4), dpi=160)
    if values.size:
        plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# -----------------------------------------------------------------------------
# LiteLoc source inventory
# -----------------------------------------------------------------------------


@dataclass
class PythonSymbolInventory:
    file: str
    module_guess: str
    functions: List[str]
    classes: List[str]
    import_error: Optional[str] = None


def module_guess_from_path(liteloc_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(liteloc_root).with_suffix("")
    return ".".join(part for part in rel.parts if part != "__init__")


def discover_liteloc_tools(liteloc_root: Path, out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    scan_dirs = [
        "utils",
        "network",
        "vector_psf",
        "spline_psf",
        "demo",
        "PSF Modeling",
        "calibrate_mat",
    ]
    inventories: List[PythonSymbolInventory] = []
    non_python_files: List[str] = []

    for folder in scan_dirs:
        root = liteloc_root / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(liteloc_root))
            if path.suffix == ".py":
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
                    functions = [
                        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                    ]
                    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                    inventories.append(
                        PythonSymbolInventory(
                            file=rel,
                            module_guess=module_guess_from_path(liteloc_root, path),
                            functions=functions,
                            classes=classes,
                        )
                    )
                except Exception as exc:
                    inventories.append(
                        PythonSymbolInventory(
                            file=rel,
                            module_guess=module_guess_from_path(liteloc_root, path),
                            functions=[],
                            classes=[],
                            import_error=str(exc),
                        )
                    )
            else:
                non_python_files.append(rel)

    report = {
        "liteloc_root": str(liteloc_root),
        "python_modules": [asdict(x) for x in inventories],
        "non_python_files": non_python_files,
        "notes": [
            "This is an inventory, not proof that every function is safe to call directly.",
            "Use wrappers below for stable post-inference QC, rendering and evaluation.",
        ],
    }
    write_json(report, out_dir / "liteloc_tools_inventory.json")

    md_lines = [
        "# LiteLoc secondary-tools inventory",
        "",
        f"Root: `{liteloc_root}`",
        "",
    ]
    for inv in inventories:
        md_lines.append(f"## `{inv.file}`")
        md_lines.append(f"Module guess: `{inv.module_guess}`")
        if inv.functions:
            md_lines.append("Functions: " + ", ".join(f"`{f}`" for f in inv.functions))
        if inv.classes:
            md_lines.append("Classes: " + ", ".join(f"`{c}`" for c in inv.classes))
        if inv.import_error:
            md_lines.append(f"Parse/import error: `{inv.import_error}`")
        md_lines.append("")
    (out_dir / "liteloc_tools_inventory.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    return report


# -----------------------------------------------------------------------------
# TIFF / movie QC
# -----------------------------------------------------------------------------


def read_tiff_sample(
    path: Path, sample_frames: int = 256
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with TiffFile(str(path)) as tif:
        series = tif.series[0]
        meta = {
            "path": str(path),
            "series_shape": tuple(int(x) for x in series.shape),
            "series_dtype": str(series.dtype),
            "num_pages": len(tif.pages),
            "is_ome": bool(getattr(tif, "is_ome", False)),
        }

        if len(tif.pages) > 1:
            n = min(sample_frames, len(tif.pages))
            try:
                arr = tif.asarray(key=list(range(n)))
            except Exception:
                arr = np.stack([tif.pages[i].asarray() for i in range(n)], axis=0)
        else:
            arr = series.asarray()

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-2:])
    arr = arr[:sample_frames]
    meta["sample_shape"] = tuple(int(x) for x in arr.shape)
    meta["sample_dtype"] = str(arr.dtype)
    return arr, meta


def run_movie_qc(
    input_movie: Path,
    out_dir: Path,
    liteloc_root: Optional[Path] = None,
    sample_frames: int = 256,
    peak_threshold: float = 0.3,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    images, meta = read_tiff_sample(input_movie, sample_frames=sample_frames)
    images_float = images.astype(np.float64, copy=False)

    stats = {
        "min": safe_float(np.nanmin(images_float)),
        "max": safe_float(np.nanmax(images_float)),
        "mean": safe_float(np.nanmean(images_float)),
        "std": safe_float(np.nanstd(images_float)),
        "median": safe_float(np.nanmedian(images_float)),
        "p01": safe_float(np.nanpercentile(images_float, 1)),
        "p99": safe_float(np.nanpercentile(images_float, 99)),
        "nonzero_fraction": safe_float(
            np.count_nonzero(images_float) / images_float.size
        ),
    }

    preview = (
        np.max(images_float, axis=0) if images_float.shape[0] > 1 else images_float[0]
    )
    save_image_png(
        preview, out_dir / "input_preview_max_projection.png", "Max projection"
    )
    save_hist(
        images_float.ravel(),
        out_dir / "input_intensity_histogram.png",
        "Input intensity histogram",
        "ADU/intensity",
    )

    results: Dict[str, Any] = {"metadata": meta, "stats": stats, "liteloc_helpers": {}}

    get_bg_stats = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "get_bg_stats"
    )
    ok, bg_result, bg_err = call_optional_function(
        get_bg_stats, images_float.copy(), percentile=10, plot=False
    )
    if ok:
        results["liteloc_helpers"]["get_bg_stats"] = {
            "status": "ok",
            "background_mean_estimate": safe_float(bg_result[0]),
            "background_gamma_scale": safe_float(bg_result[1]),
        }
    else:
        results["liteloc_helpers"]["get_bg_stats"] = {
            "status": "skipped_or_failed",
            "reason": bg_err,
        }

    extract_peaks = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "extract_smlm_peaks"
    )
    frame_index = int(min(images_float.shape[0] // 2, images_float.shape[0] - 1))
    frame = images_float[frame_index]
    bg = (
        np.median(images_float, axis=0)
        if images_float.shape[0] > 1
        else np.median(frame)
    )
    frame_nobg = np.clip(frame - bg, a_min=0, a_max=None)
    ok, peaks, peak_err = call_optional_function(
        extract_peaks,
        frame_nobg,
        dog_sigma=None,
        find_max_thre=peak_threshold,
        find_max_kernel=(3, 3),
    )
    if ok:
        peaks = np.asarray(peaks)
        results["liteloc_helpers"]["extract_smlm_peaks"] = {
            "status": "ok",
            "frame_index": frame_index,
            "n_peaks": int(len(peaks)),
            "threshold_rel": peak_threshold,
        }
        plt.figure(figsize=(7, 7), dpi=160)
        plt.imshow(normalize_image_for_png(frame), cmap="gray")
        if peaks.size:
            plt.scatter(
                peaks[:, 1],
                peaks[:, 0],
                s=12,
                facecolors="none",
                edgecolors="r",
                linewidths=0.5,
            )
        plt.title(f"Detected peaks on frame {frame_index}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            out_dir / "input_peak_overlay.png", bbox_inches="tight", pad_inches=0.02
        )
        plt.close()
    else:
        results["liteloc_helpers"]["extract_smlm_peaks"] = {
            "status": "skipped_or_failed",
            "reason": peak_err,
        }

    write_json(results, out_dir / "movie_qc_secondary.json")
    return results


# -----------------------------------------------------------------------------
# Localization table standardization and QC
# -----------------------------------------------------------------------------


COLUMN_ALIASES = {
    "frame": ["frame", "Frame", "t", "frame_index", "frame_idx"],
    "x_nm": ["x_nm", "xnm", "x", "X", "Xnm", "x [nm]", "xnm_rescale", "x_rescale"],
    "y_nm": ["y_nm", "ynm", "y", "Y", "Ynm", "y [nm]", "ynm_rescale", "y_rescale"],
    "z_nm": ["z_nm", "znm", "z", "Z", "Znm", "z [nm]"],
    "photons": ["photons", "photon", "intensity", "I", "phot", "n_photons"],
    "prob": ["prob", "confidence", "score", "p", "integrated_prob"],
    "x_sig": ["x_sig", "xsigma", "sigma_x", "x_uncertainty"],
    "y_sig": ["y_sig", "ysigma", "sigma_y", "y_uncertainty"],
    "z_sig": ["z_sig", "zsigma", "sigma_z", "z_uncertainty"],
}


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        key = c.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def load_localizations_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def standardize_localizations(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for standard, aliases in COLUMN_ALIASES.items():
        col = find_column(df, aliases)
        if col is not None:
            out[standard] = pd.to_numeric(df[col], errors="coerce")
    if "z_nm" not in out.columns:
        out["z_nm"] = np.nan
    if "frame" not in out.columns:
        out["frame"] = 0
    out = out.dropna(subset=["x_nm", "y_nm"], how="any")
    return out.reset_index(drop=True)


def summarize_locs(df: pd.DataFrame, std: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows_raw": int(len(df)),
        "n_rows_standardized": int(len(std)),
        "raw_columns": [str(c) for c in df.columns],
        "standardized_columns": [str(c) for c in std.columns],
    }
    for col in [
        "frame",
        "x_nm",
        "y_nm",
        "z_nm",
        "photons",
        "prob",
        "x_sig",
        "y_sig",
        "z_sig",
    ]:
        if col in std.columns:
            values = pd.to_numeric(std[col], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                summary[col] = {
                    "min": safe_float(np.min(values)),
                    "max": safe_float(np.max(values)),
                    "mean": safe_float(np.mean(values)),
                    "median": safe_float(np.median(values)),
                    "std": safe_float(np.std(values)),
                    "p01": safe_float(np.percentile(values, 1)),
                    "p99": safe_float(np.percentile(values, 99)),
                }
    if "frame" in std.columns and len(std):
        summary["n_frames_with_locs"] = int(std["frame"].nunique())
        summary["locs_per_frame_mean"] = safe_float(std.groupby("frame").size().mean())
        summary["locs_per_frame_max"] = int(std.groupby("frame").size().max())
    return summary


def render_density(
    std: pd.DataFrame,
    out_path: Path,
    bin_nm: float = 20.0,
    max_bins: int = 2500,
) -> Dict[str, Any]:
    x = std["x_nm"].to_numpy(dtype=float)
    y = std["y_nm"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return {"status": "empty"}

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    nx = max(8, int(np.ceil((xmax - xmin) / bin_nm)))
    ny = max(8, int(np.ceil((ymax - ymin) / bin_nm)))
    scale = max(nx / max_bins, ny / max_bins, 1.0)
    nx = int(nx / scale)
    ny = int(ny / scale)

    hist, xedges, yedges = np.histogram2d(
        x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]]
    )
    plt.figure(figsize=(7, 7), dpi=180)
    plt.imshow(hist.T, origin="lower", cmap="hot", extent=[xmin, xmax, ymin, ymax])
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title("Localization density")
    plt.colorbar(label="localizations / bin")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {
        "status": "ok",
        "path": str(out_path),
        "bin_nm_requested": bin_nm,
        "nx": nx,
        "ny": ny,
        "x_range_nm": [xmin, xmax],
        "y_range_nm": [ymin, ymax],
    }


def render_scatter(
    std: pd.DataFrame, out_path: Path, max_points: int = 200_000
) -> Dict[str, Any]:
    if std.empty:
        return {"status": "empty"}
    plot_df = std
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=1)
    plt.figure(figsize=(7, 7), dpi=180)
    plt.scatter(plot_df["x_nm"], plot_df["y_nm"], s=0.2, alpha=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title("Localization scatter preview")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {"status": "ok", "path": str(out_path), "n_plotted": int(len(plot_df))}


def fallback_fft_grid_index(
    std: pd.DataFrame,
    out_path: Path,
    pixel_size_nm: float,
    super_res_factor: int = 10,
) -> Dict[str, Any]:
    x = std["x_nm"].to_numpy(dtype=float)
    y = std["y_nm"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10:
        return {"status": "too_few_localizations"}

    bin_nm = pixel_size_nm / float(super_res_factor)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    nx = max(8, int(np.ceil((xmax - xmin) / bin_nm)))
    ny = max(8, int(np.ceil((ymax - ymin) / bin_nm)))
    if nx * ny > 30_000_000:
        scale = math.sqrt((nx * ny) / 30_000_000)
        nx = int(nx / scale)
        ny = int(ny / scale)
        bin_nm *= scale

    hist, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    projection = hist.sum(axis=1)
    spectrum = np.abs(np.fft.fft(projection))
    if spectrum[0] != 0:
        spectrum = spectrum / spectrum[0]
    freqs = np.fft.fftfreq(len(projection), d=bin_nm)
    target_freq = 1.0 / pixel_size_nm
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    amplitude = safe_float(spectrum[idx])

    shifted_freqs = np.fft.fftshift(freqs)
    shifted_spectrum = np.fft.fftshift(spectrum)
    plt.figure(figsize=(8, 4), dpi=180)
    plt.plot(shifted_freqs, shifted_spectrum)
    plt.axvline(target_freq, linestyle="--", linewidth=1)
    plt.xlabel("spatial frequency [nm^-1]")
    plt.ylabel("normalized amplitude")
    plt.title("Pixel-grid FFT artefact check")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {
        "status": "ok",
        "path": str(out_path),
        "pixel_size_nm": pixel_size_nm,
        "target_frequency_nm^-1": target_freq,
        "closest_frequency_nm^-1": safe_float(freqs[idx]),
        "normalized_amplitude_at_pixel_frequency": amplitude,
        "bin_nm_used": bin_nm,
    }


def run_frc_curve(
    std: pd.DataFrame,
    out_path: Path,
    bin_nm: float = 20.0,
    max_bins: int = 1024,
) -> Dict[str, Any]:
    if len(std) < 20:
        return {"status": "too_few_localizations"}

    if "frame" in std.columns and std["frame"].nunique() > 1:
        a = std[std["frame"].astype(int) % 2 == 0]
        b = std[std["frame"].astype(int) % 2 != 0]
    else:
        a = std.iloc[::2]
        b = std.iloc[1::2]

    x_all = std["x_nm"].to_numpy(dtype=float)
    y_all = std["y_nm"].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    nx = max(16, int(np.ceil((xmax - xmin) / bin_nm)))
    ny = max(16, int(np.ceil((ymax - ymin) / bin_nm)))
    scale = max(nx / max_bins, ny / max_bins, 1.0)
    nx = int(nx / scale)
    ny = int(ny / scale)
    actual_bin_nm = bin_nm * scale

    h1, _, _ = np.histogram2d(
        a["x_nm"], a["y_nm"], bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]]
    )
    h2, _, _ = np.histogram2d(
        b["x_nm"], b["y_nm"], bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]]
    )
    f1 = np.fft.fftshift(np.fft.fft2(h1))
    f2 = np.fft.fftshift(np.fft.fft2(h2))

    yy, xx = np.indices(f1.shape)
    cy, cx = (np.array(f1.shape) - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    max_r = int(r.max())

    frc_vals: List[float] = []
    freq_vals: List[float] = []
    for radius in range(1, max_r + 1):
        mask = r == radius
        if not np.any(mask):
            continue
        num = np.sum(f1[mask] * np.conj(f2[mask]))
        den = math.sqrt(
            float(np.sum(np.abs(f1[mask]) ** 2) * np.sum(np.abs(f2[mask]) ** 2))
        )
        if den > 0:
            frc_vals.append(float(np.real(num) / den))
            freq_vals.append(float(radius / (max(nx, ny) * actual_bin_nm)))

    plt.figure(figsize=(7, 4), dpi=180)
    plt.plot(freq_vals, frc_vals)
    plt.axhline(1 / 7, linestyle="--", linewidth=1)
    plt.xlabel("spatial frequency [nm^-1]")
    plt.ylabel("FRC")
    plt.title("Odd/even FRC diagnostic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    resolution_nm = None
    for f, v in zip(freq_vals, frc_vals):
        if v < 1 / 7 and f > 0:
            resolution_nm = 1 / f
            break

    return {
        "status": "ok",
        "path": str(out_path),
        "bin_nm_used": actual_bin_nm,
        "frc_threshold": 1 / 7,
        "resolution_nm_first_crossing": safe_float(resolution_nm),
    }


def run_localization_qc(
    localizations: Path,
    out_dir: Path,
    liteloc_root: Optional[Path] = None,
    pixel_size_nm: float = 100.0,
    render_bin_nm: float = 20.0,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    df = load_localizations_csv(localizations)
    std = standardize_localizations(df)
    std_path = out_dir / "standardized_localizations_for_review.csv"
    std.to_csv(std_path, index=False)

    summary = summarize_locs(df, std)
    summary["standardized_csv"] = str(std_path)
    summary["density_render"] = render_density(
        std, out_dir / "localization_density.png", bin_nm=render_bin_nm
    )
    summary["scatter_preview"] = render_scatter(
        std, out_dir / "localization_scatter_preview.png"
    )

    if "photons" in std.columns:
        save_hist(
            std["photons"].to_numpy(),
            out_dir / "photons_histogram.png",
            "Photon histogram",
            "photons",
        )
    if "prob" in std.columns:
        save_hist(
            std["prob"].to_numpy(),
            out_dir / "probability_histogram.png",
            "Probability/confidence histogram",
            "probability/confidence",
        )
    if "frame" in std.columns and len(std):
        per_frame = std.groupby("frame").size().rename("n_localizations").reset_index()
        per_frame.to_csv(out_dir / "localizations_per_frame.csv", index=False)
        save_hist(
            per_frame["n_localizations"].to_numpy(),
            out_dir / "localizations_per_frame_histogram.png",
            "Localizations per frame",
            "count",
        )

    # Try LiteLoc's FFT helper when available; fallback is always used for JSON consistency.
    calc_fft = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "calculate_fft_grid"
    )
    helper_status: Dict[str, Any] = {}
    if calc_fft is not None and len(std):
        try:
            liteloc_arr = (
                std[["frame", "x_nm", "y_nm", "z_nm", "photons"]]
                .fillna(0)
                .to_numpy(dtype=float)
            )
            # The helper shows/saves its own figure if fig_save_path is passed.
            result = calc_fft(
                molecule_list=liteloc_arr,
                image_size=[32, 32],
                pixel_size=pixel_size_nm,
                fig_save_path=str(out_dir / "liteloc_helper_fft_grid.png"),
            )
            helper_status["calculate_fft_grid"] = {
                "status": "ok",
                "result": str(result),
            }
        except Exception:
            helper_status["calculate_fft_grid"] = {
                "status": "failed",
                "traceback": traceback.format_exc(limit=5),
            }
    else:
        helper_status["calculate_fft_grid"] = {"status": "not_found"}

    summary["liteloc_helpers"] = helper_status
    summary["fallback_fft_grid"] = fallback_fft_grid_index(
        std, out_dir / "fallback_fft_grid_check.png", pixel_size_nm=pixel_size_nm
    )
    summary["fallback_frc"] = run_frc_curve(
        std, out_dir / "fallback_frc_curve.png", bin_nm=render_bin_nm
    )
    write_json(summary, out_dir / "localization_qc_secondary.json")
    return summary


# -----------------------------------------------------------------------------
# Evaluation against ground truth
# -----------------------------------------------------------------------------


def greedy_frame_match(
    pred: pd.DataFrame, gt: pd.DataFrame, radius_nm: float
) -> Dict[str, Any]:
    if len(pred) == 0 or len(gt) == 0:
        return {
            "tp": 0,
            "fp": int(len(pred)),
            "fn": int(len(gt)),
            "errors_xy_nm": [],
            "errors_z_nm": [],
        }

    pred_xy = pred[["x_nm", "y_nm"]].to_numpy(dtype=float)
    gt_xy = gt[["x_nm", "y_nm"]].to_numpy(dtype=float)
    matches: List[Tuple[int, int, float]] = []

    if cKDTree is not None:
        tree = cKDTree(gt_xy)
        distances, gt_indices = tree.query(pred_xy, k=1, distance_upper_bound=radius_nm)
        for pi, (dist, gi) in enumerate(zip(distances, gt_indices)):
            if np.isfinite(dist) and gi < len(gt):
                matches.append((pi, int(gi), float(dist)))
    else:
        for pi, p in enumerate(pred_xy):
            dist = np.sqrt(np.sum((gt_xy - p) ** 2, axis=1))
            gi = int(np.argmin(dist))
            if float(dist[gi]) <= radius_nm:
                matches.append((pi, gi, float(dist[gi])))

    matches.sort(key=lambda x: x[2])
    used_pred = set()
    used_gt = set()
    xy_err: List[float] = []
    z_err: List[float] = []
    for pi, gi, dist in matches:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        xy_err.append(dist)
        if "z_nm" in pred.columns and "z_nm" in gt.columns:
            pz = pred.iloc[pi]["z_nm"]
            gz = gt.iloc[gi]["z_nm"]
            if np.isfinite(pz) and np.isfinite(gz):
                z_err.append(float(pz - gz))

    tp = len(used_pred)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "errors_xy_nm": xy_err, "errors_z_nm": z_err}


def run_prediction_eval(
    predictions: Path, ground_truth: Path, out_dir: Path, radius_nm: float = 250.0
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    pred = standardize_localizations(load_localizations_csv(predictions))
    gt = standardize_localizations(load_localizations_csv(ground_truth))

    frames = sorted(
        set(pred["frame"].dropna().astype(int)).union(
            set(gt["frame"].dropna().astype(int))
        )
    )
    totals = {"tp": 0, "fp": 0, "fn": 0}
    all_xy: List[float] = []
    all_z: List[float] = []
    per_frame: List[Dict[str, Any]] = []

    for frame in frames:
        p = pred[pred["frame"].astype(int) == frame]
        g = gt[gt["frame"].astype(int) == frame]
        res = greedy_frame_match(p, g, radius_nm=radius_nm)
        for key in totals:
            totals[key] += int(res[key])
        all_xy.extend(res["errors_xy_nm"])
        all_z.extend(res["errors_z_nm"])
        per_frame.append(
            {"frame": int(frame), "tp": res["tp"], "fp": res["fp"], "fn": res["fn"]}
        )

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    rmse_xy = math.sqrt(float(np.mean(np.square(all_xy)))) if all_xy else None
    mae_xy = float(np.mean(np.abs(all_xy))) if all_xy else None
    rmse_z = math.sqrt(float(np.mean(np.square(all_z)))) if all_z else None

    pd.DataFrame(per_frame).to_csv(out_dir / "evaluation_per_frame.csv", index=False)
    if all_xy:
        save_hist(
            np.asarray(all_xy),
            out_dir / "evaluation_xy_error_histogram.png",
            "Matched XY errors",
            "XY error [nm]",
        )
    if all_z:
        save_hist(
            np.asarray(all_z),
            out_dir / "evaluation_z_error_histogram.png",
            "Matched Z errors",
            "Z error [nm]",
        )

    summary = {
        "matching_radius_nm": radius_nm,
        "n_predictions": int(len(pred)),
        "n_ground_truth": int(len(gt)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "jaccard": safe_float(jaccard),
        "rmse_xy_nm": safe_float(rmse_xy),
        "mae_xy_nm": safe_float(mae_xy),
        "rmse_z_nm": safe_float(rmse_z),
    }
    write_json(summary, out_dir / "prediction_vs_ground_truth_eval.json")
    return summary


# -----------------------------------------------------------------------------
# LiteLoc YAML helper and camera helper
# -----------------------------------------------------------------------------


def generate_infer_yaml(
    liteloc_root: Optional[Path],
    train_yaml: Path,
    out_yaml: Path,
    model_path: Optional[Path] = None,
    image_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    time_block_gb: Optional[float] = None,
    sub_fov_size: Optional[int] = None,
    over_cut: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_dir(out_yaml.parent)
    load_yaml_train = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "load_yaml_train"
    )
    create_infer_yaml = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "create_infer_yaml"
    )

    helper_status = "not_used"
    if load_yaml_train is not None and create_infer_yaml is not None:
        try:
            params = load_yaml_train(str(train_yaml))
            create_infer_yaml(params, str(out_yaml))
            helper_status = "created_with_liteloc_utils.help_utils.create_infer_yaml"
        except Exception:
            helper_status = "liteloc_helper_failed_fallback_used"

    if not out_yaml.exists():
        with train_yaml.open("r", encoding="utf-8") as f:
            train = yaml.safe_load(f) or {}
        training = train.get("Training", {}) if isinstance(train, dict) else {}
        inferred_result_path = training.get("result_path", "../results/")
        inferred_infer_data = training.get("infer_data")
        infer_dir = (
            str(Path(inferred_infer_data).parent) + "/"
            if inferred_infer_data
            else "../results/"
        )
        data = {
            "Loc_Model": {
                "model_path": str(Path(inferred_result_path) / "checkpoint.pkl")
            },
            "Multi_Process": {
                "image_path": infer_dir,
                "save_path": str(Path(infer_dir) / "result.csv"),
                "time_block_gb": 1,
                "batch_size": 64,
                "sub_fov_size": 256,
                "over_cut": 8,
                "multi_gpu": True,
                "num_producers": 1,
            },
        }
        out_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with out_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("Loc_Model", {})
    data.setdefault("Multi_Process", {})
    if model_path is not None:
        data["Loc_Model"]["model_path"] = str(model_path)
    if image_path is not None:
        data["Multi_Process"]["image_path"] = str(image_path)
    if save_path is not None:
        data["Multi_Process"]["save_path"] = str(save_path)
    if batch_size is not None:
        data["Multi_Process"]["batch_size"] = int(batch_size)
    if time_block_gb is not None:
        data["Multi_Process"]["time_block_gb"] = float(time_block_gb)
    if sub_fov_size is not None:
        data["Multi_Process"]["sub_fov_size"] = int(sub_fov_size)
    if over_cut is not None:
        data["Multi_Process"]["over_cut"] = int(over_cut)

    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {"out_yaml": str(out_yaml), "helper_status": helper_status, "content": data}


def run_adu_to_photon(
    input_movie: Path,
    camera_json: Path,
    out_dir: Path,
    liteloc_root: Optional[Path] = None,
    sample_frames: int = 64,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    adu2photon = try_get_liteloc_function(
        liteloc_root, "utils.help_utils", "adu2photon"
    )
    if adu2photon is None:
        raise RuntimeError(
            "Could not import utils.help_utils.adu2photon from LiteLoc root."
        )

    camera_params = SimpleNamespace(
        **json.loads(camera_json.read_text(encoding="utf-8"))
    )
    images, meta = read_tiff_sample(input_movie, sample_frames=sample_frames)
    photons = adu2photon(camera_params, images.astype(np.float64))
    photons = np.asarray(photons, dtype=np.float64)

    save_image_png(
        np.max(photons, axis=0),
        out_dir / "photon_preview_max_projection.png",
        "Photon max projection",
    )
    save_hist(
        photons.ravel(),
        out_dir / "photon_histogram_from_movie.png",
        "Photon histogram from movie",
        "photons",
    )
    summary = {
        "metadata": meta,
        "camera_params": json.loads(camera_json.read_text(encoding="utf-8")),
        "photon_stats": {
            "min": safe_float(np.nanmin(photons)),
            "max": safe_float(np.nanmax(photons)),
            "mean": safe_float(np.nanmean(photons)),
            "std": safe_float(np.nanstd(photons)),
        },
    }
    write_json(summary, out_dir / "adu_to_photon_summary.json")
    return summary


# -----------------------------------------------------------------------------
# Full run orchestration
# -----------------------------------------------------------------------------


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    liteloc_root = add_liteloc_to_syspath(
        Path(args.liteloc_root) if args.liteloc_root else None
    )
    out_dir = ensure_dir(Path(args.out))
    summary: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "liteloc_root": str(liteloc_root) if liteloc_root else None,
    }

    if liteloc_root is not None:
        summary["discover"] = discover_liteloc_tools(
            liteloc_root, out_dir / "00_inventory"
        )

    if args.input_movie:
        summary["movie_qc"] = run_movie_qc(
            Path(args.input_movie),
            out_dir / "01_movie_qc",
            liteloc_root=liteloc_root,
            sample_frames=args.sample_frames,
            peak_threshold=args.peak_threshold,
        )

    if args.localizations:
        summary["localization_qc"] = run_localization_qc(
            Path(args.localizations),
            out_dir / "02_localization_qc",
            liteloc_root=liteloc_root,
            pixel_size_nm=args.pixel_size_nm,
            render_bin_nm=args.render_bin_nm,
        )

    if args.localizations and args.ground_truth:
        summary["evaluation"] = run_prediction_eval(
            Path(args.localizations),
            Path(args.ground_truth),
            out_dir / "03_evaluation",
            radius_nm=args.match_radius_nm,
        )

    write_json(summary, out_dir / "liteloc_secondary_summary.json")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use LiteLoc secondary utilities safely around an existing SMLM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--liteloc-root",
        default=os.environ.get("LITELOC_ROOT"),
        help="Path to cloned LiteLoc repository.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "discover",
        help="Inventory functions/classes/files from LiteLoc utils/network/PSF modules.",
    )
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "qc-movie",
        help="Run TIFF/movie QC and optional LiteLoc peak/background helpers.",
    )
    p.add_argument("--input-movie", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-frames", type=int, default=256)
    p.add_argument("--peak-threshold", type=float, default=0.3)

    p = sub.add_parser(
        "locs-qc",
        help="Run localization CSV QC, rendering, FFT grid and FRC diagnostics.",
    )
    p.add_argument("--localizations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pixel-size-nm", type=float, default=100.0)
    p.add_argument("--render-bin-nm", type=float, default=20.0)

    p = sub.add_parser(
        "eval",
        help="Evaluate prediction CSV against ground truth CSV using frame-wise nearest-neighbour matching.",
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--match-radius-nm", type=float, default=250.0)

    p = sub.add_parser(
        "make-infer-yaml",
        help="Generate/patch LiteLoc inference YAML from a training YAML.",
    )
    p.add_argument("--train-yaml", required=True)
    p.add_argument("--out-yaml", required=True)
    p.add_argument("--model-path")
    p.add_argument("--image-path")
    p.add_argument("--save-path")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--time-block-gb", type=float)
    p.add_argument("--sub-fov-size", type=int)
    p.add_argument("--over-cut", type=int)

    p = sub.add_parser(
        "adu2photon",
        help="Convert a movie sample from ADU to photons through LiteLoc helper_utils.adu2photon.",
    )
    p.add_argument("--input-movie", required=True)
    p.add_argument(
        "--camera-json",
        required=True,
        help="JSON with baseline, e_per_adu, em_gain, spurious_c, qe.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--sample-frames", type=int, default=64)

    p = sub.add_parser(
        "all", help="Run discovery + movie QC + localization QC + optional evaluation."
    )
    p.add_argument("--input-movie")
    p.add_argument("--localizations")
    p.add_argument("--ground-truth")
    p.add_argument("--out", required=True)
    p.add_argument("--sample-frames", type=int, default=256)
    p.add_argument("--peak-threshold", type=float, default=0.3)
    p.add_argument("--pixel-size-nm", type=float, default=100.0)
    p.add_argument("--render-bin-nm", type=float, default=20.0)
    p.add_argument("--match-radius-nm", type=float, default=250.0)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    liteloc_root = add_liteloc_to_syspath(
        Path(args.liteloc_root) if args.liteloc_root else None
    )

    if args.command == "discover":
        if liteloc_root is None:
            raise ValueError("--liteloc-root or LITELOC_ROOT is required for discover")
        result = discover_liteloc_tools(liteloc_root, Path(args.out))
    elif args.command == "qc-movie":
        result = run_movie_qc(
            Path(args.input_movie),
            Path(args.out),
            liteloc_root=liteloc_root,
            sample_frames=args.sample_frames,
            peak_threshold=args.peak_threshold,
        )
    elif args.command == "locs-qc":
        result = run_localization_qc(
            Path(args.localizations),
            Path(args.out),
            liteloc_root=liteloc_root,
            pixel_size_nm=args.pixel_size_nm,
            render_bin_nm=args.render_bin_nm,
        )
    elif args.command == "eval":
        result = run_prediction_eval(
            Path(args.predictions),
            Path(args.ground_truth),
            Path(args.out),
            radius_nm=args.match_radius_nm,
        )
    elif args.command == "make-infer-yaml":
        result = generate_infer_yaml(
            liteloc_root,
            train_yaml=Path(args.train_yaml),
            out_yaml=Path(args.out_yaml),
            model_path=Path(args.model_path) if args.model_path else None,
            image_path=Path(args.image_path) if args.image_path else None,
            save_path=Path(args.save_path) if args.save_path else None,
            batch_size=args.batch_size,
            time_block_gb=args.time_block_gb,
            sub_fov_size=args.sub_fov_size,
            over_cut=args.over_cut,
        )
    elif args.command == "adu2photon":
        result = run_adu_to_photon(
            Path(args.input_movie),
            Path(args.camera_json),
            Path(args.out),
            liteloc_root=liteloc_root,
            sample_frames=args.sample_frames,
        )
    elif args.command == "all":
        result = run_all(args)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
