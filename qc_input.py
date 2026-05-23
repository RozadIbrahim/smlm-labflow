#!/usr/bin/env python3
"""
qc_input.py

Standalone + importable QC script for SMLM TIFF / OME-TIFF movies.

Two modes:

1. Standalone debug:
    python qc_input.py --input movie.tif --out results/debug_qc

2. Automatic pipeline use:
    from qc_input import qc_one_movie
    qc_result = qc_one_movie(movie_path, movie_out_dir)

Outputs per movie:
    input_qc.json
    input_preview.png
    input_histogram.png
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile


TIFF_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
)


def is_tiff(path: Path) -> bool:
    """Check whether a file looks like a TIFF / OME-TIFF."""
    name = path.name.lower()
    return path.is_file() and name.endswith(TIFF_EXTENSIONS)


def safe_stem(path: Path) -> str:
    """Create a clean folder name from a TIFF filename."""
    name = path.name

    for ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "movie"


def discover_tiffs(input_path: Path) -> List[Path]:
    """
    Accept either:
        - one TIFF file
        - one folder containing TIFF files recursively
    """
    input_path = input_path.resolve()

    if input_path.is_file():
        if not is_tiff(input_path):
            raise ValueError(f"Input is not a TIFF / OME-TIFF file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    movies = [p.resolve() for p in input_path.rglob("*") if is_tiff(p)]

    return sorted(movies, key=lambda p: str(p).lower())


def inspect_tiff(path: Path) -> Dict[str, Any]:
    """
    Read TIFF metadata without assuming too much.

    Important:
        A shape like [3, 512, 512] is not automatically 3 frames.
        If axes are CYX, then 3 means channels, not time.
    """
    info: Dict[str, Any] = {
        "input_path": str(path),
        "input_name": path.name,
        "file_size_gb": round(path.stat().st_size / (1024**3), 6),
        "shape": None,
        "axes": None,
        "dtype": None,
        "pages": None,
        "series_count": None,
        "is_ome": False,
        "is_imagej": False,
        "n_frames_guess": None,
        "frame_guess_confidence": "none",
        "frame_guess_reason": "",
        "inspect_status": "ok",
        "inspect_error": "",
    }

    try:
        with tifffile.TiffFile(path, is_ome=False) as tif:
            page_count = int(len(tif.pages))
            if page_count > 1:
                first_shape = tuple(int(x) for x in tif.pages[0].shape)
                shape = (page_count, *first_shape)
                axes = "TYX" if len(first_shape) == 2 else None
                dtype = str(tif.pages[0].dtype)
            else:
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                axes = getattr(series, "axes", None)
                dtype = str(series.dtype)

            info["shape"] = list(shape)
            info["axes"] = axes
            info["dtype"] = dtype
            info["pages"] = page_count
            info["series_count"] = int(len(tif.series))
            info["is_ome"] = ".ome." in path.name.lower()
            info["is_imagej"] = bool(tif.imagej_metadata)

            frame_info = guess_frames(shape, axes)
            info.update(frame_info)

    except Exception as exc:
        info["inspect_status"] = "failed"
        info["inspect_error"] = repr(exc)

    return info


def guess_frames(shape: Tuple[int, ...], axes: Optional[str]) -> Dict[str, Any]:
    """Conservative frame-count guessing."""
    axes = axes or ""

    if "T" in axes:
        t_index = axes.index("T")
        return {
            "n_frames_guess": int(shape[t_index]),
            "frame_guess_confidence": "high",
            "frame_guess_reason": f"Found T axis in axes={axes}",
        }

    if axes == "TYX":
        return {
            "n_frames_guess": int(shape[0]),
            "frame_guess_confidence": "high",
            "frame_guess_reason": "Axes are TYX",
        }

    if axes in {"CYX", "ZYX"}:
        return {
            "n_frames_guess": None,
            "frame_guess_confidence": "low",
            "frame_guess_reason": f"Axes are {axes}; first dimension is not safely time",
        }

    if len(shape) == 2:
        return {
            "n_frames_guess": 1,
            "frame_guess_confidence": "medium",
            "frame_guess_reason": "2D image treated as one frame",
        }

    if len(shape) == 3:
        return {
            "n_frames_guess": int(shape[0]),
            "frame_guess_confidence": "medium",
            "frame_guess_reason": "3D stack without reliable axes; first axis guessed as time",
        }

    return {
        "n_frames_guess": None,
        "frame_guess_confidence": "low",
        "frame_guess_reason": f"Shape has {len(shape)} dimensions; not safely interpreted",
    }


def read_sample_array(path: Path, max_frames: int = 200) -> np.ndarray:
    """
    Read enough data for QC without trying to be clever.

    For huge movies:
        - Multi-page TIFFs are sampled page-by-page.
        - Single-page arrays are sliced on the first axis when it looks like frames.

    This is QC, not inference.
    """
    try:
        with tifffile.TiffFile(path, is_ome=False) as tif:
            if len(tif.pages) > 1:
                n_pages = len(tif.pages)
                if n_pages <= max_frames:
                    indices = list(range(n_pages))
                else:
                    indices = np.linspace(0, n_pages - 1, max_frames).astype(int).tolist()
                return np.asarray([tif.pages[int(idx)].asarray() for idx in indices])

            if len(tif.pages) == 1:
                arr = tif.pages[0].asarray()
            else:
                arr = np.asarray([])
    except Exception:
        arr = tifffile.imread(path)

    arr = np.asarray(arr)

    if arr.ndim >= 3 and arr.shape[0] > max_frames:
        arr = arr[:max_frames]

    return arr


def make_preview_image(arr: np.ndarray) -> np.ndarray:
    """
    Make a simple 2D preview.

    Rules:
        2D -> image
        3D+ -> max projection over first axis until 2D
    """
    preview = np.asarray(arr)

    preview = np.squeeze(preview)

    while preview.ndim > 2:
        preview = np.max(preview, axis=0)

    return preview


def robust_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute useful numeric QC stats."""
    arr_float = arr.astype(np.float64, copy=False)

    finite = arr_float[np.isfinite(arr_float)]

    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "nonzero_fraction": None,
        }

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "nonzero_fraction": float(np.mean(finite != 0)),
    }


def save_preview(preview: np.ndarray, out_path: Path) -> None:
    """Save preview image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(preview, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_histogram(arr: np.ndarray, out_path: Path) -> None:
    """Save intensity histogram."""
    values = arr.astype(np.float64, copy=False).ravel()
    values = values[np.isfinite(values)]

    if values.size > 1_000_000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=1_000_000, replace=False)

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=100)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.title("Input intensity histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def qc_one_movie(
    input_path: str | Path,
    out_dir: str | Path,
    max_frames_for_qc: int = 200,
) -> Dict[str, Any]:
    """
    QC one movie.

    This is the function run_pipeline.py should call automatically.
    """
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_json = out_dir / "input_qc.json"
    preview_png = out_dir / "input_preview.png"
    histogram_png = out_dir / "input_histogram.png"

    result: Dict[str, Any] = {
        "qc_status": "started",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "qc_json": str(qc_json),
        "preview_png": str(preview_png),
        "histogram_png": str(histogram_png),
    }

    try:
        metadata = inspect_tiff(input_path)
        arr = read_sample_array(input_path, max_frames=max_frames_for_qc)

        stats = robust_stats(arr)
        preview = make_preview_image(arr)

        save_preview(preview, preview_png)
        save_histogram(arr, histogram_png)

        result.update(metadata)
        result.update(
            {
                "qc_status": "passed",
                "sampled_shape": list(arr.shape),
                "sampled_dtype": str(arr.dtype),
                "stats": stats,
                "preview_shape": list(preview.shape),
            }
        )

    except Exception as exc:
        result.update(
            {
                "qc_status": "failed",
                "error": repr(exc),
            }
        )

    qc_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def qc_path(
    input_path: str | Path,
    out_dir: str | Path,
    max_frames_for_qc: int = 200,
) -> List[Dict[str, Any]]:
    """
    QC either:
        - one TIFF
        - or all TIFFs inside a folder

    This makes qc_input.py automatic even when used standalone.
    """
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    movies = discover_tiffs(input_path)

    if not movies:
        raise RuntimeError(f"No TIFF / OME-TIFF files found in: {input_path}")

    results: List[Dict[str, Any]] = []

    single_file_mode = input_path.is_file()

    for i, movie in enumerate(movies, start=1):
        if single_file_mode:
            movie_out = out_dir
        else:
            movie_out = out_dir / f"{i:04d}_{safe_stem(movie)}"

        print(f"[QC {i}/{len(movies)}] {movie.name}")
        result = qc_one_movie(
            input_path=movie,
            out_dir=movie_out,
            max_frames_for_qc=max_frames_for_qc,
        )
        results.append(result)

    summary_path = out_dir / "qc_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nQC complete: {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="QC TIFF / OME-TIFF input movies.")

    parser.add_argument("--input", required=True, help="Input TIFF file or folder.")
    parser.add_argument("--out", required=True, help="Output QC folder.")
    parser.add_argument(
        "--max-frames-for-qc",
        type=int,
        default=200,
        help="Maximum frames sampled for QC statistics and plots.",
    )

    args = parser.parse_args()

    qc_path(
        input_path=args.input,
        out_dir=args.out,
        max_frames_for_qc=args.max_frames_for_qc,
    )


if __name__ == "__main__":
    main()
