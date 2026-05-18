#!/usr/bin/env python3
"""
export_downstream.py

Export canonical SMLM localizations to downstream-friendly formats.

Input:
    canonical_localizations.csv

Outputs:
    downstream_exports/
        picasso_thunderstorm.csv
        napari_points.csv
        downstream_export_report.json

Standalone usage:
    python export_downstream.py \
        --canonical results/test_run/batches/0001_movie/canonical_localizations.csv

Pipeline usage:
    from export_downstream import export_one
    export_one(canonical_path, batch_out_dir, profile)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def get_nested(
    profile: Optional[Dict[str, Any]], *keys: str, default: Any = None
) -> Any:
    if profile is None:
        return default

    current: Any = profile

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(default)


def load_canonical(canonical_path: str | Path) -> pd.DataFrame:
    canonical_path = Path(canonical_path).expanduser().resolve()

    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {canonical_path}")

    df = pd.read_csv(canonical_path)

    required = ["frame", "x", "y"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Canonical CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def export_picasso_thunderstorm(
    canonical_df: pd.DataFrame,
    out_path: Path,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a ThunderSTORM-style CSV that Picasso can convert with csv2hdf.

    Picasso 2D csv2hdf expected columns:
        frame
        x_nm
        y_nm
        sigma_nm
        intensity_photon
        offset_photon
        uncertainty_xy_nm

    We map:
        canonical frame      -> frame
        canonical x          -> x_nm
        canonical y          -> y_nm
        canonical photons    -> intensity_photon
        canonical background -> offset_photon

    Missing values get conservative defaults.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_sigma_nm = float(
        get_nested(profile, "downstream", "picasso", "default_sigma_nm", default=120.0)
    )

    default_uncertainty_nm = float(
        get_nested(
            profile, "downstream", "picasso", "default_uncertainty_xy_nm", default=20.0
        )
    )

    picasso = pd.DataFrame()

    picasso["frame"] = safe_numeric(canonical_df["frame"], default=0).astype(int)
    picasso["x_nm"] = safe_numeric(canonical_df["x"], default=0.0)
    picasso["y_nm"] = safe_numeric(canonical_df["y"], default=0.0)

    if "z" in canonical_df.columns and canonical_df["z"].notna().any():
        picasso["z_nm"] = safe_numeric(canonical_df["z"], default=0.0)

    picasso["sigma_nm"] = default_sigma_nm

    if "photons" in canonical_df.columns:
        picasso["intensity_photon"] = safe_numeric(canonical_df["photons"], default=0.0)
    else:
        picasso["intensity_photon"] = 0.0

    if "background" in canonical_df.columns:
        picasso["offset_photon"] = safe_numeric(canonical_df["background"], default=0.0)
    else:
        picasso["offset_photon"] = 0.0

    picasso["uncertainty_xy_nm"] = default_uncertainty_nm

    picasso.to_csv(out_path, index=False)

    return str(out_path)


def export_napari_points(
    canonical_df: pd.DataFrame,
    out_path: Path,
) -> str:
    """
    Export a simple napari-friendly points CSV.

    For 2D visualization:
        y, x are the main point coordinates.

    For 3D-like visualization:
        z, y, x are included if z exists.

    Frame is kept as a property column, not as a coordinate by default.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    napari = pd.DataFrame()

    has_z = "z" in canonical_df.columns and canonical_df["z"].notna().any()

    if has_z:
        napari["z"] = safe_numeric(canonical_df["z"], default=0.0)

    napari["y"] = safe_numeric(canonical_df["y"], default=0.0)
    napari["x"] = safe_numeric(canonical_df["x"], default=0.0)

    napari["frame"] = safe_numeric(canonical_df["frame"], default=0).astype(int)

    for col in ["photons", "background", "confidence", "backend", "source_file"]:
        if col in canonical_df.columns:
            napari[col] = canonical_df[col]

    napari.to_csv(out_path, index=False)

    return str(out_path)


def export_one(
    canonical_path: str | Path,
    out_dir: str | Path,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Export one canonical localization CSV to downstream formats.

    Parameters
    ----------
    canonical_path:
        Path to canonical_localizations.csv.

    out_dir:
        Batch output folder.

    profile:
        Optional parsed YAML profile.

    Returns
    -------
    dict
        Export report.
    """
    canonical_path = Path(canonical_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    export_dir = out_dir / "downstream_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    report_path = export_dir / "downstream_export_report.json"

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_path": str(canonical_path),
        "out_dir": str(out_dir),
        "export_dir": str(export_dir),
        "status": "started",
        "outputs": {},
        "errors": {},
    }

    canonical_df = load_canonical(canonical_path)

    report["n_localizations"] = int(len(canonical_df))
    report["canonical_columns"] = list(canonical_df.columns)

    # -----------------------------------------------------------------
    # Picasso / ThunderSTORM CSV
    # -----------------------------------------------------------------

    try:
        picasso_path = export_dir / "picasso_thunderstorm.csv"

        report["outputs"]["picasso_thunderstorm_csv"] = export_picasso_thunderstorm(
            canonical_df=canonical_df,
            out_path=picasso_path,
            profile=profile,
        )

    except Exception as exc:
        report["errors"]["picasso_thunderstorm_csv"] = repr(exc)

    # -----------------------------------------------------------------
    # Napari points CSV
    # -----------------------------------------------------------------

    try:
        napari_path = export_dir / "napari_points.csv"

        report["outputs"]["napari_points_csv"] = export_napari_points(
            canonical_df=canonical_df,
            out_path=napari_path,
        )

    except Exception as exc:
        report["errors"]["napari_points_csv"] = repr(exc)

    if report["errors"]:
        report["status"] = "partial_failed"
    else:
        report["status"] = "passed"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report["report_path"] = str(report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export canonical SMLM localizations to downstream formats."
    )

    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to canonical_localizations.csv.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output batch folder. Default: parent folder of canonical CSV. "
            "Exports are written to out/downstream_exports/."
        ),
    )

    args = parser.parse_args()

    canonical_path = Path(args.canonical).expanduser().resolve()

    if args.out is None:
        out_dir = canonical_path.parent
    else:
        out_dir = Path(args.out).expanduser().resolve()

    report = export_one(
        canonical_path=canonical_path,
        out_dir=out_dir,
        profile=None,
    )

    print("Downstream export complete.")
    print(f"Status: {report['status']}")
    print(f"Export dir: {report['export_dir']}")

    for name, path in report.get("outputs", {}).items():
        print(f"{name}: {path}")

    if report.get("errors"):
        print("\nErrors:")
        for name, error in report["errors"].items():
            print(f"{name}: {error}")


if __name__ == "__main__":
    main()
