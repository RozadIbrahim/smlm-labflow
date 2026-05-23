#!/usr/bin/env python3
"""
Combine comparison-ready benchmark rows from many SMLM LabFlow runs.

Each run writes:
    benchmarks/comparison_ready_summary.csv

This script recursively finds those rows and writes one combined table for
method-to-method, machine-to-machine, or lab-to-lab comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_PATTERN = "comparison_ready_summary.csv"


def read_rows_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def find_comparison_csvs(roots: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file() and root.name == DEFAULT_PATTERN:
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob(DEFAULT_PATTERN))
    return sorted(set(found))


def combine_comparison_rows(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in find_comparison_csvs(roots):
        for row in read_rows_csv(csv_path):
            enriched = dict(row)
            enriched["comparison_source_csv"] = str(csv_path)
            try:
                enriched["run_folder"] = str(csv_path.parents[1])
            except Exception:
                enriched["run_folder"] = str(csv_path.parent)
            rows.append(enriched)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine SMLM LabFlow comparison_ready_summary.csv files."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="Run folders, benchmark folders, or parent folders to search recursively.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="comparison_summary_all_runs.csv",
        help="Combined CSV output path.",
    )
    parser.add_argument(
        "--json",
        default="",
        help="Optional combined JSON output path.",
    )
    args = parser.parse_args()

    rows = combine_comparison_rows([Path(root) for root in args.roots])
    out_csv = Path(args.out).expanduser().resolve()
    write_rows_csv(rows, out_csv)

    if args.json:
        out_json = Path(args.json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print(f"Found runs: {len(rows)}")
    print(f"CSV: {out_csv}")
    if args.json:
        print(f"JSON: {Path(args.json).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
