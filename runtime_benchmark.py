#!/usr/bin/env python3
"""
runtime_benchmark.py

Separate but integrated runtime benchmark utility for the SMLM pipeline.

It records:
    - elapsed time per stage
    - RAM usage if psutil is installed
    - CPU usage if psutil is installed
    - GPU/CUDA memory if torch.cuda is available

Used automatically by run_pipeline.py.

Outputs:
    runtime_benchmark.csv
    runtime_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 ** 2), 3)


def safe_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def safe_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


class RuntimeBenchmark:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / "runtime_benchmark.csv"
        self.json_path = self.out_dir / "runtime_benchmark.json"

        self.rows: List[Dict[str, Any]] = []

        self.psutil = safe_import_psutil()
        self.torch = safe_import_torch()

        self.process = None

        if self.psutil is not None:
            try:
                self.process = self.psutil.Process(os.getpid())
                self.psutil.cpu_percent(interval=None)
                self.process.cpu_percent(interval=None)
            except Exception:
                self.process = None

        self.run_metadata = {
            "created_at": now_iso(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "psutil_available": self.psutil is not None,
            "torch_available": self.torch is not None,
            "cuda_available": self.cuda_available(),
        }

        self.write()

    def cuda_available(self) -> bool:
        if self.torch is None:
            return False

        try:
            return bool(self.torch.cuda.is_available())
        except Exception:
            return False

    def cuda_sync(self) -> None:
        if not self.cuda_available():
            return

        try:
            self.torch.cuda.synchronize()
        except Exception:
            pass

    def reset_cuda_peak(self) -> None:
        if not self.cuda_available():
            return

        try:
            self.torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def get_ram_mb(self) -> Dict[str, Optional[float]]:
        if self.process is None:
            return {
                "rss_mb": None,
                "vms_mb": None,
            }

        try:
            mem = self.process.memory_info()
            return {
                "rss_mb": bytes_to_mb(mem.rss),
                "vms_mb": bytes_to_mb(mem.vms),
            }
        except Exception:
            return {
                "rss_mb": None,
                "vms_mb": None,
            }

    def get_cpu_percent(self) -> Dict[str, Optional[float]]:
        if self.psutil is None or self.process is None:
            return {
                "process_cpu_percent": None,
                "system_cpu_percent": None,
            }

        try:
            return {
                "process_cpu_percent": self.process.cpu_percent(interval=None),
                "system_cpu_percent": self.psutil.cpu_percent(interval=None),
            }
        except Exception:
            return {
                "process_cpu_percent": None,
                "system_cpu_percent": None,
            }

    def get_cuda_memory_mb(self) -> Dict[str, Optional[float]]:
        if not self.cuda_available():
            return {
                "gpu_memory_allocated_mb": None,
                "gpu_memory_reserved_mb": None,
                "gpu_peak_memory_allocated_mb": None,
            }

        try:
            return {
                "gpu_memory_allocated_mb": bytes_to_mb(
                    self.torch.cuda.memory_allocated()
                ),
                "gpu_memory_reserved_mb": bytes_to_mb(
                    self.torch.cuda.memory_reserved()
                ),
                "gpu_peak_memory_allocated_mb": bytes_to_mb(
                    self.torch.cuda.max_memory_allocated()
                ),
            }
        except Exception:
            return {
                "gpu_memory_allocated_mb": None,
                "gpu_memory_reserved_mb": None,
                "gpu_peak_memory_allocated_mb": None,
            }

    @contextmanager
    def stage(
        self,
        stage_name: str,
        batch_index: Optional[int] = None,
        input_path: Optional[str | Path] = None,
        out_dir: Optional[str | Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.cuda_sync()
        self.reset_cuda_peak()

        start = time.perf_counter()
        start_time = now_iso()

        status = "passed"
        error = ""

        try:
            yield

        except Exception as exc:
            status = "failed"
            error = repr(exc)
            raise

        finally:
            self.cuda_sync()

            end = time.perf_counter()
            end_time = now_iso()

            row: Dict[str, Any] = {
                "stage": stage_name,
                "batch_index": batch_index,
                "input_path": "" if input_path is None else str(input_path),
                "out_dir": "" if out_dir is None else str(out_dir),
                "status": status,
                "error": error,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_sec": round(end - start, 6),
            }

            row.update(self.get_ram_mb())
            row.update(self.get_cpu_percent())
            row.update(self.get_cuda_memory_mb())

            if extra:
                row.update(extra)

            self.rows.append(row)
            self.write()

    def write(self) -> None:
        payload = {
            "run_metadata": self.run_metadata,
            "stages": self.rows,
        }

        self.json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if not self.rows:
            self.csv_path.write_text("", encoding="utf-8")
            return

        fieldnames: List[str] = []

        for row in self.rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def summarize(self) -> Dict[str, Any]:
        total = 0.0
        by_stage: Dict[str, float] = {}

        for row in self.rows:
            elapsed = float(row.get("elapsed_sec", 0.0))
            stage = str(row.get("stage", "unknown"))

            total += elapsed
            by_stage[stage] = by_stage.get(stage, 0.0) + elapsed

        return {
            "benchmark_csv": str(self.csv_path),
            "benchmark_json": str(self.json_path),
            "n_timed_stages": len(self.rows),
            "total_timed_sec": round(total, 6),
            "time_by_stage_sec": {
                stage: round(value, 6)
                for stage, value in sorted(by_stage.items())
            },
        }

    def finalize(self) -> Dict[str, Any]:
        self.write()
        return self.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a runtime benchmark JSON file.")
    parser.add_argument("--input", required=True, help="Path to runtime_benchmark.json.")
    args = parser.parse_args()

    path = Path(args.input).expanduser().resolve()

    data = json.loads(path.read_text(encoding="utf-8"))
    stages = data.get("stages", [])

    print(f"Benchmark file: {path}")
    print(f"Timed stages: {len(stages)}")
    print()

    for row in stages:
        print(
            f"{row.get('stage', ''):24s} "
            f"batch={str(row.get('batch_index', '')):>4s} "
            f"time={row.get('elapsed_sec', '')} sec "
            f"status={row.get('status', '')}"
        )


if __name__ == "__main__":
    main()