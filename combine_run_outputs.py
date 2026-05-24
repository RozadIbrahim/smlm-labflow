#!/usr/bin/env python3
"""
combine_run_outputs.py

Create top-level combined exports for one pipeline run.

This script keeps all per-batch files inside:

    results/<run>/batches/<batch_id>/

and creates only top-level convenience files inside:

    results/<run>/combined_exports/

Outputs:
    combined_exports/canonical_all_localizations.csv
    combined_exports/napari_all_points.csv
    combined_exports/downstream_exports_index.csv
    combined_exports/combined_export_report.json

Standalone usage:
    python combine_run_outputs.py --run results/test_run

Pipeline usage:
    from combine_run_outputs import combine_run_outputs
    combined_exports = combine_run_outputs(out_dir)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def write_json(data: Any, path: Path) -> None:
    """Write JSON safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV. Return empty DataFrame if missing or unreadable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_csv_in(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    candidates = sorted(path for path in folder.glob("*.csv") if path.is_file())
    return candidates[0] if candidates else None


def find_batch_dirs(run_dir: Path) -> List[Path]:
    """Return sorted batch folders from run_dir/batches."""
    batches_dir = run_dir / "batches"

    if not batches_dir.exists():
        return []

    return sorted(
        [path for path in batches_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )


def load_manifest(run_dir: Path) -> pd.DataFrame:
    """Load top-level batch_manifest.csv if present."""
    return read_csv_safe(run_dir / "batch_manifest.csv")


def get_batch_metadata(
    manifest: pd.DataFrame,
    batch_dir: Path,
    fallback_index: int,
) -> Dict[str, Any]:
    """
    Get batch metadata using batch_manifest.csv when possible.

    Fallback:
        batch_index = folder order
        run_id = batch folder name
    """
    metadata = {
        "batch_index": fallback_index,
        "run_id": batch_dir.name,
        "input_name": "",
        "input_path": "",
        "run_dir": str(batch_dir),
    }

    if manifest.empty:
        return metadata

    candidates = manifest.copy()

    # Match by run_dir if that column exists.
    if "run_dir" in candidates.columns:
        run_dir_text = candidates["run_dir"].astype(str)
        match = candidates[run_dir_text == str(batch_dir)]

        if not match.empty:
            row = match.iloc[0]
            metadata["batch_index"] = row.get("batch_index", fallback_index)
            metadata["run_id"] = row.get("run_id", batch_dir.name)
            metadata["input_name"] = row.get("input_name", "")
            metadata["input_path"] = row.get("input_path", "")
            metadata["run_dir"] = row.get("run_dir", str(batch_dir))
            return metadata

    # Match by run_id if that column exists.
    if "run_id" in candidates.columns:
        run_id_text = candidates["run_id"].astype(str)
        match = candidates[run_id_text == batch_dir.name]

        if not match.empty:
            row = match.iloc[0]
            metadata["batch_index"] = row.get("batch_index", fallback_index)
            metadata["run_id"] = row.get("run_id", batch_dir.name)
            metadata["input_name"] = row.get("input_name", "")
            metadata["input_path"] = row.get("input_path", "")
            metadata["run_dir"] = row.get("run_dir", str(batch_dir))
            return metadata

    return metadata


def add_column_front(
    df: pd.DataFrame,
    name: str,
    value: Any,
    position: int,
) -> pd.DataFrame:
    """Insert a metadata column safely, replacing it if it already exists."""
    if name in df.columns:
        df = df.drop(columns=[name])

    position = max(0, min(position, len(df.columns)))
    df.insert(position, name, value)

    return df


def load_one_canonical(batch_dir: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Load one batch canonical_localizations.csv and add metadata columns.

    Returns an empty DataFrame if the canonical file is missing or empty.
    """
    canonical_path = batch_dir / "canonical_localizations.csv"
    df = read_csv_safe(canonical_path)

    if df.empty:
        return pd.DataFrame()

    df = add_column_front(df, "input_path", metadata.get("input_path", ""), 0)
    df = add_column_front(df, "input_name", metadata.get("input_name", ""), 0)
    df = add_column_front(df, "run_id", metadata.get("run_id", batch_dir.name), 0)
    df = add_column_front(df, "batch_index", metadata.get("batch_index", ""), 0)

    return df


def make_napari_all_points(canonical_all: pd.DataFrame) -> pd.DataFrame:
    """
    Create top-level napari-friendly points table from combined canonical table.

    Coordinate columns:
        y, x for 2D
        z, y, x if z is present and non-empty

    Metadata columns are kept as properties.
    """
    if canonical_all.empty:
        return pd.DataFrame()

    required = ["x", "y"]
    missing = [col for col in required if col not in canonical_all.columns]

    if missing:
        return pd.DataFrame()

    napari = pd.DataFrame()

    has_z = "z" in canonical_all.columns and canonical_all["z"].notna().any()

    if has_z:
        napari["z"] = pd.to_numeric(
            canonical_all["z"],
            errors="coerce",
        ).fillna(0.0)

    napari["y"] = pd.to_numeric(
        canonical_all["y"],
        errors="coerce",
    ).fillna(0.0)

    napari["x"] = pd.to_numeric(
        canonical_all["x"],
        errors="coerce",
    ).fillna(0.0)

    property_columns = [
        "batch_index",
        "run_id",
        "input_name",
        "input_path",
        "frame",
        "photons",
        "background",
        "confidence",
        "backend",
        "source_file",
    ]

    for col in property_columns:
        if col in canonical_all.columns:
            napari[col] = canonical_all[col]

    return napari


def collect_downstream_index(
    batch_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create one row for downstream_exports_index.csv."""
    downstream_dir = batch_dir / "downstream_exports"
    exports_dir = batch_dir / "exports"

    canonical_path = batch_dir / "canonical_localizations.csv"
    raw_path = batch_dir / "liteloc_raw_output.csv"
    vanilla_path = first_csv_in(exports_dir / "vanilla")
    picasso_path = first_existing(
        [
            exports_dir / "picasso" / "picasso_localizations.csv",
            downstream_dir / "picasso_thunderstorm.csv",
        ]
    )
    napari_path = first_existing(
        [
            exports_dir / "napari" / "napari_points.csv",
            downstream_dir / "napari_points.csv",
        ]
    )
    smap_path = first_existing([exports_dir / "smap" / "smap_localizations.csv"])
    locan_path = first_existing([exports_dir / "locan" / "locan_localizations.csv"])
    generic_path = first_existing(
        [exports_dir / "generic" / "smlm_generic_localizations.csv"]
    )
    post_summary_path = batch_dir / "post_inference_summary.json"
    export_report_path = downstream_dir / "downstream_export_report.json"

    return {
        "batch_index": metadata.get("batch_index", ""),
        "run_id": metadata.get("run_id", batch_dir.name),
        "input_name": metadata.get("input_name", ""),
        "input_path": metadata.get("input_path", ""),
        "run_dir": str(batch_dir),
        "canonical_localizations": str(canonical_path)
        if canonical_path.exists()
        else "",
        "vanilla_backend_csv": str(vanilla_path) if vanilla_path else "",
        "raw_backend_csv": str(raw_path) if raw_path.exists() else "",
        "picasso_thunderstorm_csv": str(picasso_path) if picasso_path else "",
        "napari_points_csv": str(napari_path) if napari_path else "",
        "smap_csv": str(smap_path) if smap_path else "",
        "locan_csv": str(locan_path) if locan_path else "",
        "generic_smlm_csv": str(generic_path) if generic_path else "",
        "post_inference_summary": str(post_summary_path)
        if post_summary_path.exists()
        else "",
        "downstream_export_report": str(export_report_path)
        if export_report_path.exists()
        else "",
    }


def combine_run_outputs(run_dir: str | Path) -> Dict[str, Any]:
    """
    Create top-level combined exports for one run.

    Per-batch files are not moved or deleted.
    """
    run_dir = Path(run_dir).expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    combined_dir = run_dir / "combined_exports"
    combined_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_dir)
    batch_dirs = find_batch_dirs(run_dir)

    canonical_tables: List[pd.DataFrame] = []
    index_rows: List[Dict[str, Any]] = []

    for fallback_index, batch_dir in enumerate(batch_dirs, start=1):
        metadata = get_batch_metadata(
            manifest=manifest,
            batch_dir=batch_dir,
            fallback_index=fallback_index,
        )

        canonical_df = load_one_canonical(batch_dir, metadata)

        if not canonical_df.empty:
            canonical_tables.append(canonical_df)

        index_rows.append(
            collect_downstream_index(
                batch_dir=batch_dir,
                metadata=metadata,
            )
        )

    canonical_all_path = combined_dir / "canonical_all_localizations.csv"
    napari_all_path = combined_dir / "napari_all_points.csv"
    export_index_path = combined_dir / "downstream_exports_index.csv"
    report_path = combined_dir / "combined_export_report.json"

    if canonical_tables:
        canonical_all = pd.concat(canonical_tables, ignore_index=True)
    else:
        canonical_all = pd.DataFrame()

    canonical_all.to_csv(canonical_all_path, index=False)

    napari_all = make_napari_all_points(canonical_all)
    napari_all.to_csv(napari_all_path, index=False)

    export_index = pd.DataFrame(index_rows)
    export_index.to_csv(export_index_path, index=False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed",
        "run_dir": str(run_dir),
        "combined_dir": str(combined_dir),
        "n_batches_found": len(batch_dirs),
        "n_batches_with_canonical": len(canonical_tables),
        "n_total_localizations": int(len(canonical_all)),
        "outputs": {
            "canonical_all_localizations": str(canonical_all_path),
            "napari_all_points": str(napari_all_path),
            "downstream_exports_index": str(export_index_path),
            "combined_export_report": str(report_path),
        },
        "note": (
            "Per-batch files stay inside batches/*/. "
            "combined_exports/ only contains top-level convenience files."
        ),
    }

    write_json(report, report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create top-level combined exports for one SMLM pipeline run."
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Pipeline run folder, e.g. results/test_run.",
    )

    args = parser.parse_args()

    report = combine_run_outputs(args.run)

    print("Combined exports created.")
    print(f"Status:           {report.get('status', '')}")
    print(f"Combined dir:     {report.get('combined_dir', '')}")
    print(
        f"Canonical all:    {report['outputs'].get('canonical_all_localizations', '')}"
    )
    print(f"Napari all:       {report['outputs'].get('napari_all_points', '')}")
    print(f"Export index:     {report['outputs'].get('downstream_exports_index', '')}")
    print(f"Combined report:  {report['outputs'].get('combined_export_report', '')}")


if __name__ == "__main__":
    main()
