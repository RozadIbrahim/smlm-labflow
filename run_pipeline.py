#!/usr/bin/env python3
"""
run_pipeline.py

Main SMLM wrapper pipeline.

Automatic behavior:
    - Accept one TIFF/OME-TIFF file OR a folder of TIFF/OME-TIFF files.
    - Discover all movies automatically.
    - Run qc_input.py automatically through qc_one_movie().
    - Run LiteLoc adapter automatically if adapters/liteloc_adapter.py exists.
    - Run canonical conversion automatically if raw backend output exists.
    - Record runtime benchmark automatically through runtime_benchmark.py.

Normal command:
    python run_pipeline.py \
        --input data/raw_movies \
        --out results/run_001 \
        --profile profiles/dna_paint_standard.yaml
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


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------


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

    Example:
        results/run_real_liteloc
        results/run_real_liteloc_001
        results/run_real_liteloc_002
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

    # If user already gave run_001, make next as run_002, not run_001_001.
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

    raise RuntimeError(
        f"Could not find available numbered output directory for: {out_dir}"
    )

def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}

    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = value

    return clean


def write_manifest_csv(rows: List[Dict[str, Any]], path: Path) -> None:
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


# ---------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------


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


def get_backend_name(profile: Dict[str, Any], cli_backend: Optional[str]) -> str:
    if cli_backend:
        return cli_backend

    backend_block = profile.get("backend", {})

    if isinstance(backend_block, dict):
        return str(backend_block.get("name", "liteloc"))

    return "liteloc"


# ---------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------


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


def get_optional_converter_function() -> Tuple[Optional[Callable[..., Any]], str]:
    try:
        module = importlib.import_module("convert_to_canonical")
    except ModuleNotFoundError:
        return None, "post_inference.py not found"
    except Exception as exc:
        return None, f"convert_to_canonical import failed: {repr(exc)}"

    for name in [
        "convert_one",
        "convert_liteloc_to_canonical",
        "convert_to_canonical",
    ]:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, f"using convert_to_canonical.{name}"

    return None, "No supported converter function found"


# ---------------------------------------------------------------------
# Backend and conversion
# ---------------------------------------------------------------------


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


def convert_if_available(
    raw_output_path: str,
    movie_out_dir: Path,
    profile: Dict[str, Any],
    source_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert backend raw output to canonical localization CSV.

    Supports multiple possible converter signatures:
        1. convert_liteloc_to_canonical(raw_csv=..., output_csv=..., source_file=...)
        2. convert_liteloc_to_canonical(input_path=..., out_path=..., profile=...)
        3. convert_liteloc_to_canonical(raw_output, canonical_path)
    """
    if not raw_output_path:
        return {
            "canonical_status": "pending_no_raw_output",
            "canonical_message": "No raw backend output available.",
            "canonical_output_path": "",
        }

    raw_output = Path(raw_output_path).expanduser().resolve()

    if not raw_output.exists():
        return {
            "canonical_status": "failed_raw_output_missing",
            "canonical_message": f"Raw output path does not exist: {raw_output}",
            "canonical_output_path": "",
        }

    fn, message = get_optional_converter_function()

    if fn is None:
        return {
            "canonical_status": "pending_converter_missing",
            "canonical_message": message,
            "canonical_output_path": "",
        }

    canonical_name = "canonical_localizations.csv"

    output_block = profile.get("output", {})
    if isinstance(output_block, dict):
        canonical_name = str(
            output_block.get("canonical_output_name", canonical_name)
        )

    canonical_path = movie_out_dir / canonical_name

    # -------------------------------------------------------------
    # Try modern explicit signature first:
    # convert_liteloc_to_canonical(raw_csv=..., output_csv=..., source_file=...)
    # -------------------------------------------------------------
    try:
        output = fn(
            raw_csv=raw_output,
            output_csv=canonical_path,
            source_file=source_file,
        )

        if output is not None:
            canonical_path = Path(output)

        return {
            "canonical_status": "passed",
            "canonical_message": message + " with raw_csv/output_csv/source_file signature",
            "canonical_output_path": str(canonical_path),
        }

    except TypeError:
        pass

    except Exception as exc:
        return {
            "canonical_status": "failed",
            "canonical_message": repr(exc),
            "canonical_output_path": "",
        }

    # -------------------------------------------------------------
    # Try older keyword signature:
    # convert_one(input_path=..., out_path=..., profile=...)
    # -------------------------------------------------------------
    try:
        output = fn(
            input_path=raw_output,
            out_path=canonical_path,
            profile=profile,
        )

        if output is not None:
            canonical_path = Path(output)

        return {
            "canonical_status": "passed",
            "canonical_message": message + " with input_path/out_path/profile signature",
            "canonical_output_path": str(canonical_path),
        }

    except TypeError:
        pass

    except Exception as exc:
        return {
            "canonical_status": "failed",
            "canonical_message": repr(exc),
            "canonical_output_path": "",
        }

    # -------------------------------------------------------------
    # Try simple positional fallback:
    # convert_liteloc_to_canonical(raw_output, canonical_path)
    # -------------------------------------------------------------
    try:
        output = fn(raw_output, canonical_path)

        if output is not None:
            canonical_path = Path(output)

        return {
            "canonical_status": "passed",
            "canonical_message": message + " with positional fallback",
            "canonical_output_path": str(canonical_path),
        }

    except Exception as exc:
        return {
            "canonical_status": "failed",
            "canonical_message": repr(exc),
            "canonical_output_path": "",
        }
# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def run_pipeline(
    input_path: Path,
    out_dir: Path,
    profile_path: Path,
    backend_override: Optional[str] = None,
    max_files: Optional[int] = None,
    force: bool = False,
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
            f"{out_dir}\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    bench = RuntimeBenchmark(out_dir=out_dir)

    profile = load_profile(profile_path)
    backend_name = get_backend_name(profile, backend_override)

    movies = discover_tiff_movies(input_path)

    if max_files is not None:
        movies = movies[:max_files]

    if not movies:
        raise RuntimeError(f"No TIFF/OME-TIFF files found in: {input_path}")

    qc_one_movie = get_qc_function()

    batches_dir = out_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print("=" * 70)
    print("SMLM wrapper pipeline")
    print("=" * 70)
    print(f"Input:   {input_path}")
    print(f"Output:  {out_dir}")
    print(f"Profile: {profile_path}")
    print(f"Backend: {backend_name}")
    print(f"Movies:  {len(movies)}")
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
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        # -------------------------------------------------------------
        # QC stage
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Backend + canonical stages
        # -------------------------------------------------------------

        if qc_status != "passed":
            backend_result = {
                "backend_status": "skipped_qc_failed",
                "backend_name": backend_name,
                "backend_message": "QC failed; backend skipped.",
                "raw_output_path": "",
            }

            canonical_result = {
                "canonical_status": "skipped_qc_failed",
                "canonical_message": "QC failed; canonical conversion skipped.",
                "canonical_output_path": "",
            }

            export_result = {
                "status": "skipped_qc_failed",
                "message": "QC failed; downstream export skipped.",
            }

            print("    Backend: skipped_qc_failed")
            print("    Canonical: skipped_qc_failed")
            print("    Downstream export: skipped_qc_failed")

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

            with bench.stage(
                "canonical_conversion",
                batch_index=index,
                input_path=backend_result.get("raw_output_path", ""),
                out_dir=movie_out_dir,
            ):
                canonical_result = convert_if_available(
                    raw_output_path=backend_result.get("raw_output_path", ""),
                    movie_out_dir=movie_out_dir,
                    profile=profile,
                    source_file=movie_path,
                )
            print(f"    Canonical: {canonical_result.get('canonical_status')}")
            if canonical_result.get("canonical_status") == "passed":
                try:
                    from export_downstream import export_one

                    export_result = export_one(
                        canonical_path=canonical_result.get("canonical_output_path"),
                        out_dir=movie_out_dir,
                        profile=profile,
                    )

                except Exception as exc:
                    export_result = {
                        "status": "failed",
                        "error": repr(exc),
                    }
            else:
                export_result = {
                    "status": "skipped_no_canonical",
                }

            print(f"    Downstream export: {export_result.get('status')}")
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
        row["downstream_export_status"] = export_result.get("status", "")
        row["downstream_export_result"] = export_result
        rows.append(row)
        print()

    # -----------------------------------------------------------------
    # Save manifests and summary
    # -----------------------------------------------------------------

    manifest_csv = out_dir / "batch_manifest.csv"
    manifest_json = out_dir / "batch_manifest.json"
    summary_json = out_dir / "run_summary.json"

    write_manifest_csv(rows, manifest_csv)
    write_json(rows, manifest_json)

    benchmark_summary = bench.finalize()

    # -----------------------------------------------------------------
    # Create top-level combined exports
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Save manifests and summary
    # -----------------------------------------------------------------

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "out_dir": str(out_dir),
        "profile_path": str(profile_path),
        "backend_name": backend_name,
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
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "summary_json": str(summary_json),
        "benchmark": benchmark_summary,
        "combined_exports": combined_exports,
    }

    # Write summary once before report generation.
    # The report generator reads run_summary.json.
    write_json(summary, summary_json)

    # -----------------------------------------------------------------
    # Generate supervisor-friendly report
    # -----------------------------------------------------------------

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

    # Write final summary again after report generation.
    write_json(summary, summary_json)

    # -----------------------------------------------------------------
    # Final terminal output
    # -----------------------------------------------------------------

    print("=" * 70)
    print("Pipeline complete")
    print("=" * 70)
    print(f"Manifest CSV:      {manifest_csv}")
    print(f"Manifest JSON:     {manifest_json}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Benchmark CSV:     {benchmark_summary['benchmark_csv']}")
    print(f"Benchmark JSON:    {benchmark_summary['benchmark_json']}")

    report = summary.get("report", {})

    if report.get("status") == "passed":
        print(f"Report Markdown:   {report.get('markdown_report', '')}")
        print(f"Report HTML:       {report.get('html_report', '')}")
        print(f"Report assets:     {report.get('assets_dir', '')}")
    else:
        print("Report generation: failed")
        print(f"Report error:      {report.get('error', '')}")

    combined_outputs = combined_exports.get("outputs", {})

    if combined_outputs:
        combined_dir = combined_exports.get("combined_dir", "")
        canonical_all = combined_outputs.get("canonical_all_localizations", "")
        napari_all = combined_outputs.get("napari_all_points", "")
        export_index = combined_outputs.get("downstream_exports_index", "")
        combined_report = combined_outputs.get("combined_export_report", "")

        print(f"Combined exports:  {combined_dir}")
        print(f"Canonical all:     {canonical_all}")
        print(f"Napari all:        {napari_all}")
        print(f"Export index:      {export_index}")
        print(f"Combined report:   {combined_report}")
    else:
        print("Combined exports:  none")

    print("=" * 70)

    return summary

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMLM wrapper pipeline with automatic QC, backend, canonical conversion, and runtime benchmark."
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
        "--review-locan",
        action="store_true",
        help="Generate Locan-style downstream review after post-inference.",
    )

    parser.add_argument(
        "--open-napari",
        action="store_true",
        help="Open napari viewer after post-inference. GUI/blocking.",
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
        )

    except Exception as exc:
        print("\nPipeline failed.")
        print(f"Error: {repr(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()