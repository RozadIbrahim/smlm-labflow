#!/usr/bin/env python3
"""
generate_run_report.py

Generate a human-readable report for one SMLM pipeline run.

Usage:
    python generate_run_report.py --run results/test_run

Outputs:
    run_report.md
    run_report.html
    report_assets/
        runtime_by_stage.png
        localizations_per_movie.png
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def image_to_base64(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""

    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def find_batch_dirs(run_dir: Path) -> List[Path]:
    batches_dir = run_dir / "batches"

    if not batches_dir.exists():
        return []

    return sorted(
        [p for p in batches_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def count_canonical_localizations(batch_dir: Path) -> Optional[int]:
    csv_path = batch_dir / "canonical_localizations.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        return int(len(df))
    except Exception:
        return None


def summarize_batches(run_dir: Path) -> pd.DataFrame:
    manifest_path = run_dir / "batch_manifest.csv"
    manifest = read_csv_safe(manifest_path)

    rows: List[Dict[str, Any]] = []

    if not manifest.empty:
        for _, row in manifest.iterrows():
            batch_dir_raw = row.get("run_dir", "")
            batch_dir = Path(str(batch_dir_raw)) if batch_dir_raw else None

            if batch_dir is not None and not batch_dir.is_absolute():
                batch_dir = (run_dir / batch_dir).resolve()

            if batch_dir is None or not batch_dir.exists():
                batch_dir = None

            n_locs = count_canonical_localizations(batch_dir) if batch_dir else None

            rows.append(
                {
                    "batch_index": row.get("batch_index", ""),
                    "run_id": row.get("run_id", ""),
                    "input_name": row.get("input_name", ""),
                    "qc_status": row.get("qc_status", ""),
                    "backend_status": row.get("backend_status", ""),
                    "canonical_status": row.get("canonical_status", ""),
                    "shape": row.get("shape", ""),
                    "axes": row.get("axes", ""),
                    "dtype": row.get("dtype", ""),
                    "n_frames_guess": row.get("n_frames_guess", ""),
                    "n_localizations": n_locs,
                    "run_dir": str(batch_dir) if batch_dir else "",
                }
            )

    else:
        for i, batch_dir in enumerate(find_batch_dirs(run_dir), start=1):
            qc = read_json(batch_dir / "input_qc.json")
            n_locs = count_canonical_localizations(batch_dir)

            rows.append(
                {
                    "batch_index": i,
                    "run_id": batch_dir.name,
                    "input_name": qc.get("input_name", ""),
                    "qc_status": qc.get("qc_status", ""),
                    "backend_status": "",
                    "canonical_status": (
                        "passed"
                        if (batch_dir / "canonical_localizations.csv").exists()
                        else ""
                    ),
                    "shape": qc.get("shape", ""),
                    "axes": qc.get("axes", ""),
                    "dtype": qc.get("dtype", ""),
                    "n_frames_guess": qc.get("n_frames_guess", ""),
                    "n_localizations": n_locs,
                    "run_dir": str(batch_dir),
                }
            )

    return pd.DataFrame(rows)


def select_existing_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    existing = [col for col in columns if col in df.columns]
    return df[existing] if existing else df


def make_runtime_plot(runtime_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if runtime_df.empty:
        return None

    if "stage" not in runtime_df.columns or "elapsed_sec" not in runtime_df.columns:
        return None

    runtime_df = runtime_df.copy()
    runtime_df["elapsed_sec"] = pd.to_numeric(
        runtime_df["elapsed_sec"],
        errors="coerce",
    )

    grouped = (
        runtime_df.groupby("stage", dropna=False)["elapsed_sec"]
        .sum()
        .sort_values(ascending=False)
    )

    if grouped.empty:
        return None

    plt.figure(figsize=(8, 4.5))
    grouped.plot(kind="bar")
    plt.ylabel("Total runtime (sec)")
    plt.xlabel("Pipeline stage")
    plt.title("Runtime by pipeline stage")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def make_localization_plot(batch_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if batch_df.empty or "n_localizations" not in batch_df.columns:
        return None

    df = batch_df.copy()
    df["n_localizations"] = pd.to_numeric(
        df["n_localizations"],
        errors="coerce",
    )

    df = df.dropna(subset=["n_localizations"])

    if df.empty:
        return None

    labels = df["input_name"].fillna(df["run_id"]).astype(str)
    values = df["n_localizations"].astype(int)

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Number of localizations")
    plt.xlabel("Input movie")
    plt.title("Canonical localizations per movie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"

    shown = df.head(max_rows).copy()

    try:
        return shown.to_markdown(index=False)
    except Exception:
        return shown.to_string(index=False)


def make_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"

    return df.head(max_rows).to_html(index=False, escape=True)


def collect_preview_cards(batch_df: pd.DataFrame, max_cards: int = 12) -> str:
    if batch_df.empty or "run_dir" not in batch_df.columns:
        return "<p><em>No previews available.</em></p>"

    cards = []

    for _, row in batch_df.head(max_cards).iterrows():
        batch_dir = Path(str(row.get("run_dir", "")))

        preview = batch_dir / "input_preview.png"
        histogram = batch_dir / "input_histogram.png"

        preview_b64 = image_to_base64(preview)
        hist_b64 = image_to_base64(histogram)

        title = str(row.get("input_name", row.get("run_id", "movie")))
        qc_status = str(row.get("qc_status", ""))
        canonical_status = str(row.get("canonical_status", ""))
        n_locs = row.get("n_localizations", "")

        preview_img = (
            f'<img src="{preview_b64}" alt="preview" />'
            if preview_b64
            else "<p><em>No preview image.</em></p>"
        )

        hist_img = (
            f'<img src="{hist_b64}" alt="histogram" />'
            if hist_b64
            else "<p><em>No histogram image.</em></p>"
        )

        cards.append(
            f"""
            <div class="card">
                <h3>{title}</h3>
                <p>
                    <strong>QC:</strong> {qc_status}<br>
                    <strong>Canonical:</strong> {canonical_status}<br>
                    <strong>Localizations:</strong> {n_locs}
                </p>
                <div class="image-row">
                    <div>{preview_img}</div>
                    <div>{hist_img}</div>
                </div>
            </div>
            """
        )

    return "\n".join(cards)


def generate_markdown_report(
    run_dir: Path,
    summary: Dict[str, Any],
    batch_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_plot: Optional[Path],
    loc_plot: Optional[Path],
) -> str:
    created_at = datetime.now().isoformat(timespec="seconds")

    n_movies = len(batch_df)

    qc_passed = 0
    canonical_passed = 0

    if not batch_df.empty and "qc_status" in batch_df.columns:
        qc_passed = int((batch_df["qc_status"] == "passed").sum())

    if not batch_df.empty and "canonical_status" in batch_df.columns:
        canonical_passed = int((batch_df["canonical_status"] == "passed").sum())

    runtime_total = "NA"

    if not runtime_df.empty and "elapsed_sec" in runtime_df.columns:
        runtime_total = round(
            float(pd.to_numeric(runtime_df["elapsed_sec"], errors="coerce").sum()),
            3,
        )

    batch_cols = [
        "batch_index",
        "input_name",
        "qc_status",
        "backend_status",
        "canonical_status",
        "shape",
        "axes",
        "dtype",
        "n_localizations",
    ]

    runtime_cols = [
        "stage",
        "batch_index",
        "elapsed_sec",
        "status",
        "rss_mb",
        "process_cpu_percent",
        "gpu_peak_memory_allocated_mb",
    ]

    batch_md = dataframe_to_markdown(select_existing_columns(batch_df, batch_cols))

    runtime_md = dataframe_to_markdown(
        select_existing_columns(runtime_df, runtime_cols)
    )

    lines = [
        "# SMLM Pipeline Run Report",
        "",
        f"**Generated:** {created_at}",
        "",
        "## Run overview",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Input: `{summary.get('input', '')}`",
        f"- Profile: `{summary.get('profile_path', '')}`",
        f"- Backend: `{summary.get('backend_name', '')}`",
        f"- Movies processed: **{n_movies}**",
        f"- QC passed: **{qc_passed}/{n_movies}**",
        f"- Canonical conversion passed: **{canonical_passed}/{n_movies}**",
        f"- Total timed runtime: **{runtime_total} sec**",
        "",
        "## Batch summary",
        "",
        batch_md,
        "",
        "## Runtime summary",
        "",
        runtime_md,
        "",
    ]

    if runtime_plot is not None:
        lines.extend(
            [
                "## Runtime plot",
                "",
                f"![Runtime by stage]({runtime_plot.relative_to(run_dir)})",
                "",
            ]
        )

    if loc_plot is not None:
        lines.extend(
            [
                "## Localization count plot",
                "",
                f"![Localizations per movie]({loc_plot.relative_to(run_dir)})",
                "",
            ]
        )

    lines.extend(
        [
            "## Files produced",
            "",
            "- `batch_manifest.csv`",
            "- `batch_manifest.json`",
            "- `run_summary.json`",
            "- `runtime_benchmark.csv`",
            "- `runtime_benchmark.json`",
            "",
        ]
    )

    return "\n".join(lines)


def generate_html_report(
    run_dir: Path,
    summary: Dict[str, Any],
    batch_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_plot: Optional[Path],
    loc_plot: Optional[Path],
) -> str:
    created_at = datetime.now().isoformat(timespec="seconds")

    n_movies = len(batch_df)

    qc_passed = 0
    canonical_passed = 0

    if not batch_df.empty and "qc_status" in batch_df.columns:
        qc_passed = int((batch_df["qc_status"] == "passed").sum())

    if not batch_df.empty and "canonical_status" in batch_df.columns:
        canonical_passed = int((batch_df["canonical_status"] == "passed").sum())

    runtime_total = "NA"

    if not runtime_df.empty and "elapsed_sec" in runtime_df.columns:
        runtime_total = str(
            round(
                float(pd.to_numeric(runtime_df["elapsed_sec"], errors="coerce").sum()),
                3,
            )
        )

    runtime_plot_b64 = image_to_base64(runtime_plot) if runtime_plot else ""
    loc_plot_b64 = image_to_base64(loc_plot) if loc_plot else ""

    runtime_plot_html = (
        f'<img class="plot" src="{runtime_plot_b64}" alt="Runtime plot" />'
        if runtime_plot_b64
        else "<p><em>No runtime plot available.</em></p>"
    )

    loc_plot_html = (
        f'<img class="plot" src="{loc_plot_b64}" alt="Localization plot" />'
        if loc_plot_b64
        else "<p><em>No localization plot available.</em></p>"
    )

    batch_cols = [
        "batch_index",
        "input_name",
        "qc_status",
        "backend_status",
        "canonical_status",
        "shape",
        "axes",
        "dtype",
        "n_localizations",
    ]

    runtime_cols = [
        "stage",
        "batch_index",
        "elapsed_sec",
        "status",
        "rss_mb",
        "process_cpu_percent",
        "gpu_peak_memory_allocated_mb",
    ]

    batch_table = make_html_table(select_existing_columns(batch_df, batch_cols))

    runtime_table = make_html_table(select_existing_columns(runtime_df, runtime_cols))

    preview_cards = collect_preview_cards(batch_df)

    input_text = summary.get("input", "")
    profile_text = summary.get("profile_path", "")
    backend_text = summary.get("backend_name", "")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SMLM Pipeline Run Report</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 32px;
        line-height: 1.45;
        color: #222;
        background: #fafafa;
    }}

    h1, h2, h3 {{
        color: #111;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }}

    .metric {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 14px;
    }}

    .metric .label {{
        font-size: 0.85rem;
        color: #666;
    }}

    .metric .value {{
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 4px;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
        background: white;
        margin: 14px 0 28px 0;
        font-size: 0.9rem;
    }}

    th, td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        vertical-align: top;
    }}

    th {{
        background: #f0f0f0;
    }}

    .plot {{
        max-width: 100%;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: white;
        padding: 8px;
    }}

    .card {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
    }}

    .image-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
    }}

    .image-row img {{
        max-width: 100%;
        border: 1px solid #ddd;
        border-radius: 8px;
    }}

    code {{
        background: #eee;
        padding: 2px 5px;
        border-radius: 4px;
    }}
</style>
</head>
<body>

<h1>SMLM Pipeline Run Report</h1>

<p><strong>Generated:</strong> {created_at}</p>
<p><strong>Run folder:</strong> <code>{run_dir}</code></p>
<p><strong>Input:</strong> <code>{input_text}</code></p>
<p><strong>Profile:</strong> <code>{profile_text}</code></p>
<p><strong>Backend:</strong> <code>{backend_text}</code></p>

<div class="summary-grid">
    <div class="metric">
        <div class="label">Movies processed</div>
        <div class="value">{n_movies}</div>
    </div>
    <div class="metric">
        <div class="label">QC passed</div>
        <div class="value">{qc_passed}/{n_movies}</div>
    </div>
    <div class="metric">
        <div class="label">Canonical passed</div>
        <div class="value">{canonical_passed}/{n_movies}</div>
    </div>
    <div class="metric">
        <div class="label">Timed runtime</div>
        <div class="value">{runtime_total} sec</div>
    </div>
</div>

<h2>Batch summary</h2>
{batch_table}

<h2>Runtime summary</h2>
{runtime_table}

<h2>Runtime plot</h2>
{runtime_plot_html}

<h2>Localization count plot</h2>
{loc_plot_html}

<h2>QC previews</h2>
{preview_cards}

</body>
</html>
"""

    return html


def generate_run_report(run_dir: str | Path) -> Dict[str, str]:
    run_dir = Path(run_dir).expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(run_dir / "run_summary.json")
    batch_df = summarize_batches(run_dir)
    runtime_df = read_csv_safe(run_dir / "runtime_benchmark.csv")

    runtime_plot = make_runtime_plot(
        runtime_df=runtime_df,
        out_path=assets_dir / "runtime_by_stage.png",
    )

    loc_plot = make_localization_plot(
        batch_df=batch_df,
        out_path=assets_dir / "localizations_per_movie.png",
    )

    markdown = generate_markdown_report(
        run_dir=run_dir,
        summary=summary,
        batch_df=batch_df,
        runtime_df=runtime_df,
        runtime_plot=runtime_plot,
        loc_plot=loc_plot,
    )

    html = generate_html_report(
        run_dir=run_dir,
        summary=summary,
        batch_df=batch_df,
        runtime_df=runtime_df,
        runtime_plot=runtime_plot,
        loc_plot=loc_plot,
    )

    md_path = run_dir / "run_report.md"
    html_path = run_dir / "run_report.html"

    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    return {
        "markdown_report": str(md_path),
        "html_report": str(html_path),
        "assets_dir": str(assets_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SMLM pipeline run report.")
    parser.add_argument("--run", required=True, help="Pipeline run directory.")
    args = parser.parse_args()

    outputs = generate_run_report(args.run)

    print("Report generated:")
    print(f"Markdown: {outputs['markdown_report']}")
    print(f"HTML:     {outputs['html_report']}")
    print(f"Assets:   {outputs['assets_dir']}")


if __name__ == "__main__":
    main()
