#!/usr/bin/env python3
"""
run_pipeline.py

Main SMLM wrapper pipeline.

User-facing CLI.

Architecture:
    QC
    → LiteLoc adapter
    → post_inference.run_post_inference()
        → canonical CSV
        → localization QC
        → SMAP/Picasso/napari/Locan adapted exports
    → combined run-level exports
    → supervisor-friendly report

Important:
    This script does NOT open napari and does NOT run Locan analysis.
    napari/Locan review should be run separately in napari_locan_env using
    napari_locan_review.py.

Normal command:
    python run_pipeline.py \
        --input data/raw_movies \
        --out results/run_001 \
        --profile profiles/dna_paint_standard.yaml \
        --backend liteloc \
        --coord-units nm \
        --pixel-size-nm 65

Quick test:
    python run_pipeline.py \
        --input data/raw_movies \
        --out results/test_run \
        --profile profiles/dna_paint_standard.yaml \
        --backend liteloc \
        --coord-units nm \
        --pixel-size-nm 65 \
        --max-files 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from runtime_benchmark import RuntimeBenchmark


TIFF_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
)


# =============================================================================
# Basic utilities
# =============================================================================

def is_tiff(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and name.endswith(TIFF_EXTENSIONS)


def safe_stem(path: Path) -> str:
    name = path.name

    for ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._-")

    return name or "movie"


def make_run_id(path: Path, index: int) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{index:04d}_{safe_stem(path)}_{digest}"


def display_path(path: Path | str | None, base: Optional[Path] = None) -> str:
    """
    Return a clean relative path for terminal output when possible.
    Internal paths remain absolute.
    """
    if path is None:
        return ""

    path = Path(path)

    if str(path).strip() == "":
        return ""

    try:
        path = path.expanduser().resolve()
    except Exception:
        return str(path)

    if base is None:
        base = Path.cwd().resolve()
    else:
        base = Path(base).expanduser().resolve()

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def discover_tiff_movies(input_path: Path) -> List[Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if not is_tiff(input_path):
            raise ValueError(f"Input file is not TIFF/OME-TIFF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    movies = [p.resolve() for p in input_path.rglob("*") if is_tiff(p)]

    return sorted(movies, key=lambda p: str(p).lower())


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_numbered_out_dir(out_dir: Path, force: bool = False) -> Path:
    """
    Return a safe output directory.

    Behavior:
        - If out_dir does not exist: use it.
        - If out_dir exists and is empty: use it.
        - If out_dir exists and is non-empty:
            - with --force: reuse it.
            - without --force: create numbered sibling folder.
    """
    out_dir = out_dir.expanduser().resolve()

    if force:
        return out_dir

    if not out_dir.exists():
        return out_dir

    if out_dir.is_dir() and not any(out_dir.iterdir()):
        return out_dir

    parent = out_dir.parent
    base_name = out_dir.name

    match = re.match(r"^(.*?)(?:_(\d{3,}))$", base_name)

    if match:
        prefix = match.group(1)
        start_number = int(match.group(2)) + 1
    else:
        prefix = base_name
        start_number = 1

    for number in range(start_number, 10000):
        candidate = parent / f"{prefix}_{number:03d}"

        if not candidate.exists():
            return candidate

        if candidate.is_dir() and not any(candidate.iterdir()):
            return candidate

    raise RuntimeError(f"Could not find available numbered output directory for: {out_dir}")


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}

    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = value

    return clean


def write_manifest_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(flatten_for_csv(row))


# =============================================================================
# Profile loading
# =============================================================================

def load_profile(profile_path: Path) -> Dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    try:
        import yaml
    except ImportError:
        return {
            "profile_path": str(profile_path),
            "profile_loaded": False,
            "profile_warning": "PyYAML not installed; profile was not parsed.",
            "backend": {"name": "liteloc"},
        }

    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    if not isinstance(profile, dict):
        raise ValueError(f"Profile is not a YAML dictionary: {profile_path}")

    profile["profile_path"] = str(profile_path)
    profile["profile_loaded"] = True

    return profile


def get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def get_backend_name(profile: Dict[str, Any], cli_backend: Optional[str]) -> str:
    if cli_backend:
        return cli_backend

    backend_block = profile.get("backend", {})

    if isinstance(backend_block, dict):
        return str(backend_block.get("name", "liteloc"))

    return "liteloc"


def infer_pixel_size_nm(
    profile: Dict[str, Any],
    cli_pixel_size_nm: Optional[float],
) -> Optional[float]:
    """
    Resolve pixel size from CLI first, then profile.

    Supported profile locations:
        pixel_size_nm
        data.pixel_size_nm
        input.pixel_size_nm
        camera.pixel_size_nm
        acquisition.pixel_size_nm
        microscope.pixel_size_nm
        smlm.pixel_size_nm
    """
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


# =============================================================================
# Dynamic imports
# =============================================================================

def get_qc_function() -> Callable[..., Dict[str, Any]]:
    try:
        from qc_input import qc_one_movie
    except Exception as exc:
        raise ImportError(
            "Could not import qc_one_movie from qc_input.py. "
            "Make sure qc_input.py exists in the project root."
        ) from exc

    return qc_one_movie


def get_optional_liteloc_function() -> Tuple[Optional[Callable[..., Any]], str]:
    try:
        module = importlib.import_module("adapters.liteloc_adapter")
    except ModuleNotFoundError:
        return None, "adapters.liteloc_adapter not found"
    except Exception as exc:
        return None, f"adapters.liteloc_adapter import failed: {repr(exc)}"

    for name in [
        "run_liteloc_one_movie",
        "run_inference_one_movie",
        "run_liteloc",
    ]:
        fn = getattr(module, name, None)

        if callable(fn):
            return fn, f"using adapters.liteloc_adapter.{name}"

    return None, "No supported LiteLoc adapter function found"


def get_post_inference_function() -> Callable[..., Dict[str, Any]]:
    try:
        from post_inference import run_post_inference
    except Exception as exc:
        raise ImportError(
            "Could not import run_post_inference from post_inference.py. "
            "Make sure post_inference.py exists in the project root."
        ) from exc

    return run_post_inference


# =============================================================================
# Backend execution
# =============================================================================

def run_backend_if_available(
    backend_name: str,
    movie_path: Path,
    movie_out_dir: Path,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    backend_name = backend_name.lower().strip()

    if backend_name != "liteloc":
        return {
            "backend_status": "skipped_unsupported_backend",
            "backend_name": backend_name,
            "backend_message": f"Unsupported backend: {backend_name}",
            "raw_output_path": "",
        }

    fn, message = get_optional_liteloc_function()

    if fn is None:
        return {
            "backend_status": "pending_adapter_missing",
            "backend_name": "liteloc",
            "backend_message": message,
            "raw_output_path": "",
        }

    try:
        raw_output = fn(
            input_path=movie_path,
            out_dir=movie_out_dir,
            profile=profile,
        )

        if raw_output is None:
            return {
                "backend_status": "pending_no_raw_output",
                "backend_name": "liteloc",
                "backend_message": message + " but no raw output was returned",
                "raw_output_path": "",
            }

        return {
            "backend_status": "passed",
            "backend_name": "liteloc",
            "backend_message": message,
            "raw_output_path": str(raw_output),
        }

    except TypeError:
        try:
            raw_output = fn(movie_path, movie_out_dir, profile)

            if raw_output is None:
                return {
                    "backend_status": "pending_no_raw_output",
                    "backend_name": "liteloc",
                    "backend_message": message + " with positional fallback but no raw output was returned",
                    "raw_output_path": "",
                }

            return {
                "backend_status": "passed",
                "backend_name": "liteloc",
                "backend_message": message + " with positional fallback",
                "raw_output_path": str(raw_output),
            }

        except Exception as exc:
            return {
                "backend_status": "failed",
                "backend_name": "liteloc",
                "backend_message": repr(exc),
                "raw_output_path": "",
            }

    except Exception as exc:
        return {
            "backend_status": "failed",
            "backend_name": "liteloc",
            "backend_message": repr(exc),
            "raw_output_path": "",
        }


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(
    input_path: Path,
    out_dir: Path,
    profile_path: Path,
    backend_override: Optional[str] = None,
    max_files: Optional[int] = None,
    force: bool = False,
    coord_units: str = "auto",
    pixel_size_nm: Optional[float] = None,
    export_smap_enabled: Optional[bool] = None,
    export_picasso_enabled: Optional[bool] = None,
    export_napari_enabled: Optional[bool] = None,
    export_locan_enabled: Optional[bool] = None,
    default_locprec_nm: float = 20.0,
    default_lpx_px: float = 1.0,
    napari_units: str = "nm",
    locan_units: str = "nm",
) -> Dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    profile_path = profile_path.expanduser().resolve()

    requested_out_dir = out_dir
    out_dir = get_numbered_out_dir(out_dir, force=force)

    if out_dir != requested_out_dir:
        print(
            f"Output directory already exists and is not empty.\n"
            f"Using new numbered output directory:\n"
            f"{display_path(out_dir)}\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    bench = RuntimeBenchmark(out_dir=out_dir)

    profile = load_profile(profile_path)
    backend_name = get_backend_name(profile, backend_override)
    resolved_pixel_size_nm = infer_pixel_size_nm(profile, pixel_size_nm)

    movies = discover_tiff_movies(input_path)

    if max_files is not None:
        movies = movies[:max_files]

    if not movies:
        raise RuntimeError(f"No TIFF/OME-TIFF files found in: {input_path}")

    qc_one_movie = get_qc_function()
    run_post_inference = get_post_inference_function()

    batches_dir = out_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print("=" * 70)
    print("SMLM wrapper pipeline")
    print("=" * 70)
    print(f"Input:          {display_path(input_path)}")
    print(f"Output:         {display_path(out_dir)}")
    print(f"Profile:        {display_path(profile_path)}")
    print(f"Backend:        {backend_name}")
    print(f"Movies:         {len(movies)}")
    print(f"Coord units:    {coord_units}")
    print(f"Pixel size nm:  {resolved_pixel_size_nm}")
    print("Review:         external/manual")
    print("=" * 70)
    print()

    for index, movie_path in enumerate(movies, start=1):
        run_id = make_run_id(movie_path, index)
        movie_out_dir = batches_dir / run_id
        movie_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(movies)}] {movie_path.name}")

        base_row: Dict[str, Any] = {
            "batch_index": index,
            "run_id": run_id,
            "input_path": str(movie_path),
            "input_name": movie_path.name,
            "input_parent": str(movie_path.parent),
            "run_dir": str(movie_out_dir),
            "profile_path": str(profile_path),
            "backend_name": backend_name,
            "coord_units_requested": coord_units,
            "pixel_size_nm": resolved_pixel_size_nm,
            "review_mode": "external_manual",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        # -----------------------------------------------------------------
        # QC stage
        # -----------------------------------------------------------------

        try:
            with bench.stage(
                "qc",
                batch_index=index,
                input_path=movie_path,
                out_dir=movie_out_dir,
            ):
                qc_result = qc_one_movie(
                    input_path=movie_path,
                    out_dir=movie_out_dir,
                )

        except TypeError:
            try:
                with bench.stage(
                    "qc",
                    batch_index=index,
                    input_path=movie_path,
                    out_dir=movie_out_dir,
                ):
                    qc_result = qc_one_movie(movie_path, movie_out_dir)

            except Exception as exc:
                qc_result = {
                    "qc_status": "failed",
                    "qc_error": repr(exc),
                }

        except Exception as exc:
            qc_result = {
                "qc_status": "failed",
                "qc_error": repr(exc),
            }

        qc_status = qc_result.get("qc_status", "unknown")
        print(f"    QC: {qc_status}")

        # -----------------------------------------------------------------
        # Backend + post-inference stages
        # -----------------------------------------------------------------

        if qc_status != "passed":
            backend_result = {
                "backend_status": "skipped_qc_failed",
                "backend_name": backend_name,
                "backend_message": "QC failed; backend skipped.",
                "raw_output_path": "",
            }

            canonical_result = {
                "canonical_status": "skipped_qc_failed",
                "canonical_message": "QC failed; post-inference skipped.",
                "canonical_output_path": "",
                "post_inference_summary": "",
                "localization_qc": "",
            }

            export_result = {
                "status": "skipped_qc_failed",
                "message": "QC failed; downstream export skipped.",
            }

            print("    Backend: skipped_qc_failed")
            print("    Post-inference: skipped_qc_failed")
            print("    Review: external_manual")

        else:
            with bench.stage(
                "backend_liteloc",
                batch_index=index,
                input_path=movie_path,
                out_dir=movie_out_dir,
            ):
                backend_result = run_backend_if_available(
                    backend_name=backend_name,
                    movie_path=movie_path,
                    movie_out_dir=movie_out_dir,
                    profile=profile,
                )

            print(f"    Backend: {backend_result.get('backend_status')}")

            raw_output_path = backend_result.get("raw_output_path", "")

            if raw_output_path:
                try:
                    with bench.stage(
                        "post_inference",
                        batch_index=index,
                        input_path=raw_output_path,
                        out_dir=movie_out_dir,
                    ):
                        post_summary = run_post_inference(
                            input_path=raw_output_path,
                            out_dir=movie_out_dir,
                            profile=profile,
                            backend_name=backend_name,
                            source_file=str(movie_path),
                            coord_units=coord_units,
                            pixel_size_nm=resolved_pixel_size_nm,
                            default_locprec_nm=default_locprec_nm,
                            default_lpx_px=default_lpx_px,
                            napari_units=napari_units,
                            locan_units=locan_units,
                            export_smap_enabled=export_smap_enabled,
                            export_picasso_enabled=export_picasso_enabled,
                            export_napari_enabled=export_napari_enabled,
                            export_locan_enabled=export_locan_enabled,
                        )

                    canonical_output_path = post_summary.get("canonical_csv", "")

                    canonical_result = {
                        "canonical_status": "passed" if canonical_output_path else "failed",
                        "canonical_message": "post_inference.run_post_inference completed",
                        "canonical_output_path": canonical_output_path,
                        "post_inference_summary": post_summary.get(
                            "post_inference_summary",
                            str(movie_out_dir / "post_inference_summary.json"),
                        ),
                        "localization_qc": post_summary.get("localization_qc", ""),
                    }

                    export_result = {
                        "status": post_summary.get("status", "unknown"),
                        "exports": post_summary.get("exports", {}),
                        "plots": post_summary.get("plots", {}),
                        "quality_flags": post_summary.get("quality_flags", []),
                        "coord_units_detected": post_summary.get(
                            "coord_units_detected",
                            coord_units,
                        ),
                        "pixel_size_nm": post_summary.get(
                            "pixel_size_nm",
                            resolved_pixel_size_nm,
                        ),
                    }

                except Exception as exc:
                    post_summary = {}

                    canonical_result = {
                        "canonical_status": "failed",
                        "canonical_message": repr(exc),
                        "canonical_output_path": "",
                        "post_inference_summary": "",
                        "localization_qc": "",
                    }

                    export_result = {
                        "status": "failed",
                        "error": repr(exc),
                    }

            else:
                post_summary = {}

                canonical_result = {
                    "canonical_status": "skipped_no_raw_output",
                    "canonical_message": "No raw backend output available for post-inference.",
                    "canonical_output_path": "",
                    "post_inference_summary": "",
                    "localization_qc": "",
                }

                export_result = {
                    "status": "skipped_no_raw_output",
                }

            print(f"    Post-inference: {export_result.get('status')}")
            print("    Review: external_manual")

        # -----------------------------------------------------------------
        # Manifest row
        # -----------------------------------------------------------------

        row: Dict[str, Any] = {}
        row.update(base_row)

        row["qc_status"] = qc_result.get("qc_status", "")
        row["qc_json"] = qc_result.get("qc_json", str(movie_out_dir / "input_qc.json"))
        row["qc_preview"] = qc_result.get("preview_png", str(movie_out_dir / "input_preview.png"))
        row["qc_histogram"] = qc_result.get("histogram_png", str(movie_out_dir / "input_histogram.png"))
        row["shape"] = qc_result.get("shape", "")
        row["axes"] = qc_result.get("axes", "")
        row["dtype"] = qc_result.get("dtype", "")
        row["n_frames_guess"] = qc_result.get("n_frames_guess", "")
        row["frame_guess_confidence"] = qc_result.get("frame_guess_confidence", "")
        row["qc_full_result"] = qc_result

        row.update(backend_result)
        row.update(canonical_result)

        row["post_inference_status"] = export_result.get("status", "")
        row["downstream_export_status"] = export_result.get("status", "")
        row["downstream_export_result"] = export_result
        row["post_inference_summary"] = canonical_result.get("post_inference_summary", "")
        row["localization_qc"] = canonical_result.get("localization_qc", "")
        row["review_status"] = "external_manual"
        row["review_result"] = {
            "status": "external_manual",
            "message": (
                "napari/Locan review is not run inside run_pipeline.py. "
                "Use napari_locan_review.py separately in napari_locan_env."
            ),
        }

        rows.append(row)
        print()

    # =========================================================================
    # Save manifests and benchmark
    # =========================================================================

    manifest_csv = out_dir / "batch_manifest.csv"
    manifest_json = out_dir / "batch_manifest.json"
    summary_json = out_dir / "run_summary.json"

    write_manifest_csv(rows, manifest_csv)
    write_json(rows, manifest_json)

    benchmark_summary = bench.finalize()

    # =========================================================================
    # Create top-level combined exports
    # =========================================================================

    try:
        from combine_run_outputs import combine_run_outputs

        combined_exports = combine_run_outputs(out_dir)

    except Exception as exc:
        combined_exports = {
            "status": "failed",
            "error": repr(exc),
            "combined_dir": "",
            "outputs": {},
        }

    # =========================================================================
    # Save summary
    # =========================================================================

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "requested_out_dir": str(requested_out_dir),
        "out_dir": str(out_dir),
        "profile_path": str(profile_path),
        "backend_name": backend_name,
        "coord_units_requested": coord_units,
        "pixel_size_nm": resolved_pixel_size_nm,
        "review_mode": "external_manual",
        "n_movies": len(rows),

        "qc_passed": sum(row.get("qc_status") == "passed" for row in rows),
        "qc_failed": sum(row.get("qc_status") == "failed" for row in rows),

        "backend_passed": sum(row.get("backend_status") == "passed" for row in rows),
        "backend_failed": sum(row.get("backend_status") == "failed" for row in rows),
        "backend_pending_adapter_missing": sum(
            row.get("backend_status") == "pending_adapter_missing"
            for row in rows
        ),

        "canonical_passed": sum(
            row.get("canonical_status") == "passed"
            for row in rows
        ),
        "canonical_failed": sum(
            row.get("canonical_status") == "failed"
            for row in rows
        ),

        "post_inference_passed": sum(
            row.get("post_inference_status") in {"passed", "warning"}
            for row in rows
        ),
        "post_inference_failed": sum(
            row.get("post_inference_status") == "failed"
            for row in rows
        ),

        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "summary_json": str(summary_json),
        "benchmark": benchmark_summary,
        "combined_exports": combined_exports,
    }

    write_json(summary, summary_json)

    # =========================================================================
    # Generate supervisor-friendly report
    # =========================================================================

    try:
        from generate_run_report import generate_run_report

        report_outputs = generate_run_report(out_dir)

        summary["report"] = {
            "status": "passed",
            "markdown_report": report_outputs.get("markdown_report", ""),
            "html_report": report_outputs.get("html_report", ""),
            "assets_dir": report_outputs.get("assets_dir", ""),
        }

    except Exception as exc:
        summary["report"] = {
            "status": "failed",
            "error": repr(exc),
        }

    write_json(summary, summary_json)

    # =========================================================================
    # Final terminal output
    # =========================================================================

    print("=" * 70)
    print("Pipeline complete")
    print("=" * 70)

    print(f"Manifest CSV:      {display_path(manifest_csv)}")
    print(f"Manifest JSON:     {display_path(manifest_json)}")
    print(f"Summary JSON:      {display_path(summary_json)}")

    if isinstance(benchmark_summary, dict):
        benchmark_csv = benchmark_summary.get("benchmark_csv", "")
        benchmark_json = benchmark_summary.get("benchmark_json", "")

        print(f"Benchmark CSV:     {display_path(benchmark_csv) if benchmark_csv else ''}")
        print(f"Benchmark JSON:    {display_path(benchmark_json) if benchmark_json else ''}")
    else:
        print("Benchmark:         unavailable")

    report = summary.get("report", {})

    if report.get("status") == "passed":
        markdown_report = report.get("markdown_report", "")
        html_report = report.get("html_report", "")
        assets_dir = report.get("assets_dir", "")

        print(f"Report Markdown:   {display_path(markdown_report) if markdown_report else ''}")
        print(f"Report HTML:       {display_path(html_report) if html_report else ''}")
        print(f"Report assets:     {display_path(assets_dir) if assets_dir else ''}")
    else:
        print("Report generation: failed")
        print(f"Report error:      {report.get('error', '')}")

    combined_exports_safe = summary.get("combined_exports", {})
    combined_outputs = combined_exports_safe.get("outputs", {})

    if combined_outputs:
        combined_dir = combined_exports_safe.get("combined_dir", "")
        print(f"Combined exports:  {display_path(combined_dir) if combined_dir else ''}")

        known_keys = [
            "canonical_all_localizations",
            "smap_all_localizations",
            "picasso_all_localizations",
            "napari_all_points",
            "locan_all_localizations",
            "downstream_exports_index",
            "combined_export_report",
        ]

        printed = set()

        for key in known_keys:
            value = combined_outputs.get(key, "")

            if value:
                label = key.replace("_", " ").title()
                print(f"{label + ':':<25} {display_path(value)}")
                printed.add(key)

        for key, value in combined_outputs.items():
            if key in printed:
                continue

            label = key.replace("_", " ").title()

            if isinstance(value, str) and value:
                print(f"{label + ':':<25} {display_path(value)}")
            else:
                print(f"{label + ':':<25} {value}")
    else:
        print("Combined exports:  none")

    print("=" * 70)

    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SMLM wrapper pipeline with automatic QC, LiteLoc backend, "
            "post-inference adapted exports, combined outputs, and runtime benchmark."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input TIFF/OME-TIFF file or folder.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output run folder.",
    )

    parser.add_argument(
        "--profile",
        required=True,
        help="YAML profile path.",
    )

    parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override. Default comes from profile.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional test limit.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow reuse of non-empty output folder.",
    )

    parser.add_argument(
        "--coord-units",
        choices=["auto", "nm", "pixel"],
        default="auto",
        help="Units of localization coordinates after inference.",
    )

    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        default=None,
        help="Camera pixel size in nm for nm/pixel downstream conversion.",
    )

    parser.add_argument(
        "--default-locprec-nm",
        type=float,
        default=20.0,
        help="Default localization precision in nm for SMAP export when lpx/lpy are missing.",
    )

    parser.add_argument(
        "--default-lpx-px",
        type=float,
        default=1.0,
        help="Default localization precision in pixels for Picasso export when lpx/lpy are missing.",
    )

    parser.add_argument(
        "--napari-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Coordinate units written to napari CSV export from post_inference.",
    )

    parser.add_argument(
        "--locan-units",
        choices=["nm", "pixel"],
        default="nm",
        help="Coordinate units written to Locan CSV export from post_inference.",
    )

    smap_group = parser.add_mutually_exclusive_group()
    smap_group.add_argument(
        "--export-smap",
        dest="export_smap",
        action="store_true",
        help="Force SMAP-adapted CSV export.",
    )
    smap_group.add_argument(
        "--no-smap",
        dest="export_smap",
        action="store_false",
        help="Disable SMAP-adapted CSV export.",
    )

    picasso_group = parser.add_mutually_exclusive_group()
    picasso_group.add_argument(
        "--export-picasso",
        dest="export_picasso",
        action="store_true",
        help="Force Picasso-adapted CSV export.",
    )
    picasso_group.add_argument(
        "--no-picasso",
        dest="export_picasso",
        action="store_false",
        help="Disable Picasso-adapted CSV export.",
    )

    napari_group = parser.add_mutually_exclusive_group()
    napari_group.add_argument(
        "--export-napari",
        dest="export_napari",
        action="store_true",
        help="Force napari points CSV export.",
    )
    napari_group.add_argument(
        "--no-napari",
        dest="export_napari",
        action="store_false",
        help="Disable napari points CSV export.",
    )

    locan_group = parser.add_mutually_exclusive_group()
    locan_group.add_argument(
        "--export-locan",
        dest="export_locan",
        action="store_true",
        help="Force Locan-style CSV export.",
    )
    locan_group.add_argument(
        "--no-locan",
        dest="export_locan",
        action="store_false",
        help="Disable Locan-style CSV export.",
    )

    parser.set_defaults(
        export_smap=None,
        export_picasso=None,
        export_napari=None,
        export_locan=None,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=Path(args.input),
            out_dir=Path(args.out),
            profile_path=Path(args.profile),
            backend_override=args.backend,
            max_files=args.max_files,
            force=args.force,
            coord_units=args.coord_units,
            pixel_size_nm=args.pixel_size_nm,
            export_smap_enabled=args.export_smap,
            export_picasso_enabled=args.export_picasso,
            export_napari_enabled=args.export_napari,
            export_locan_enabled=args.export_locan,
            default_locprec_nm=args.default_locprec_nm,
            default_lpx_px=args.default_lpx_px,
            napari_units=args.napari_units,
            locan_units=args.locan_units,
        )

    except Exception as exc:
        print("\nPipeline failed.")
        print(f"Error: {repr(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()