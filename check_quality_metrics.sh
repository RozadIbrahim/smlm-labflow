#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT="${1:-quality_metrics.py}"
PY="${PYTHON:-python}"

ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
fail() { echo "❌ $*"; exit 1; }

echo "======================================================================"
echo "Checking quality_metrics.py"
echo "======================================================================"
echo "Script: $SCRIPT"
echo "Python: $($PY --version 2>&1)"
echo

[[ -f "$SCRIPT" ]] || fail "File not found: $SCRIPT"

echo "[1/7] Syntax compilation"
"$PY" -m py_compile "$SCRIPT" || fail "Python syntax compilation failed."
ok "quality_metrics.py compiles"

echo
echo "[2/7] Import + API + runtime smoke tests"

"$PY" - "$SCRIPT" <<'PY'
import csv
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import traceback

script_path = pathlib.Path(sys.argv[1]).resolve()

def die(msg: str) -> None:
    print(f"❌ {msg}")
    raise SystemExit(1)

def ok(msg: str) -> None:
    print(f"✅ {msg}")

def warn(msg: str) -> None:
    print(f"⚠️  {msg}")

# ---------------------------------------------------------------------
# 1. Source-level regression checks
# ---------------------------------------------------------------------
src = script_path.read_text(encoding="utf-8")

bad_patterns = [
    'value not in {None, "", []}',
    "value not in {None, '', []}",
    "value not in {None,\"\",[]}",
]

for pattern in bad_patterns:
    if pattern in src:
        die(
            "Old unhashable-list crash pattern still exists: "
            f"{pattern!r}. Replace it with a safe is_missing_value() helper."
        )

ok("Old {None, '', []} crash pattern not found")

# ---------------------------------------------------------------------
# 2. Import module from file
# ---------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("quality_metrics_under_test", script_path)
if spec is None or spec.loader is None:
    die("Could not create import spec for quality_metrics.py")

module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
except Exception:
    traceback.print_exc()
    die("Module import failed")

ok("quality_metrics.py imports cleanly")

# ---------------------------------------------------------------------
# 3. Public API expected by run_pipeline.py
# ---------------------------------------------------------------------
required_functions = [
    "run_quality_after_calibrate",
    "run_quality_after_train",
    "run_quality_after_infer",
    "run_quality_metrics",
]

missing = [name for name in required_functions if not callable(getattr(module, name, None))]
if missing:
    die(f"Missing required public functions: {missing}")

ok("Required public functions exist")

# Compatibility aliases are not mandatory forever, but they are useful now.
expected_aliases = [
    "after_calibrate",
    "after_train",
    "after_infer",
    "run_after_calibrate",
    "run_after_train",
    "run_after_infer",
]

missing_aliases = [name for name in expected_aliases if not callable(getattr(module, name, None))]
if missing_aliases:
    warn(f"Compatibility aliases missing: {missing_aliases}")
else:
    ok("Compatibility aliases exist")

# ---------------------------------------------------------------------
# 4. Direct Markdown generation regression test
# ---------------------------------------------------------------------
if callable(getattr(module, "make_quality_markdown", None)):
    try:
        md = module.make_quality_markdown(
            "infer",
            {
                "created_at": "test",
                "status": "passed",
                "metrics": [
                    {
                        "metric": "dummy_metric",
                        "status": "passed",
                        "message": "markdown smoke test",
                        "path": "",
                        "source_file": None,
                        "model_path": [],
                        "canonical_csv": "",
                        "grid_index": None,
                        "n_localizations": 10,
                    }
                ],
                "benchmark_files": {"files": {}},
            },
        )
        if "# Quality metrics after infer" not in md:
            die("Markdown output does not contain expected title")
        ok("Markdown generation works with None/empty/list values")
    except Exception:
        traceback.print_exc()
        die("Markdown generation failed; likely the old unhashable-list bug is still present")
else:
    warn("make_quality_markdown() not found; skipping direct Markdown regression test")

# ---------------------------------------------------------------------
# 5. Build fake run directory
# ---------------------------------------------------------------------
with tempfile.TemporaryDirectory(prefix="qm_smoke_") as tmp:
    root = pathlib.Path(tmp)
    results = root / "results"
    benchmarks = root / "benchmarks"
    reports = root / "reports"
    registry = root / "registry"

    for d in [results, benchmarks, reports, registry]:
        d.mkdir(parents=True, exist_ok=True)

    canonical = results / "canonical_localizations.csv"

    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "x",
                "y",
                "z",
                "photons",
                "background",
                "confidence",
                "backend",
            ],
        )
        writer.writeheader()
        for i in range(100):
            writer.writerow(
                {
                    "frame": i // 10,
                    "x": 100.0 + i * 0.2,
                    "y": 200.0 + i * 0.3,
                    "z": 0.0,
                    "photons": 1500 + i,
                    "background": 20,
                    "confidence": 0.95,
                    "backend": "test",
                }
            )

    checkpoint = results / "checkpoint.pkl"
    checkpoint.write_bytes(b"dummy checkpoint smoke test")

    calibration = results / "psf_calibration_dummy.yaml"
    calibration.write_text("dummy: true\n", encoding="utf-8")

    profile = {
        "quality_control": {
            "automatic": True,
        }
    }

    backend_config = {}

    # -----------------------------------------------------------------
    # 6. Run public functions
    # -----------------------------------------------------------------
    try:
        infer_payload = module.run_quality_after_infer(
            run_parent=root,
            profile=profile,
            backend_config=backend_config,
            step_result={"canonical_csv": str(canonical)},
        )
    except Exception:
        traceback.print_exc()
        die("run_quality_after_infer() crashed")

    if infer_payload.get("status") not in {"passed", "warning"}:
        die(f"infer quality returned bad status: {infer_payload.get('status')!r}")

    ok(f"run_quality_after_infer() executed with status={infer_payload.get('status')}")

    try:
        train_payload = module.run_quality_after_train(
            run_parent=root,
            profile=profile,
            backend_config=backend_config,
            step_result={"checkpoint_path": str(checkpoint)},
        )
    except Exception:
        traceback.print_exc()
        die("run_quality_after_train() crashed")

    if train_payload.get("status") not in {"passed", "warning"}:
        die(f"train quality returned bad status: {train_payload.get('status')!r}")

    ok(f"run_quality_after_train() executed with status={train_payload.get('status')}")

    try:
        calibrate_payload = module.run_quality_after_calibrate(
            run_parent=root,
            profile=profile,
            backend_config=backend_config,
            step_result={"calibration_file": str(calibration)},
        )
    except Exception:
        traceback.print_exc()
        die("run_quality_after_calibrate() crashed")

    if calibrate_payload.get("status") not in {"passed", "warning"}:
        die(f"calibrate quality returned bad status: {calibrate_payload.get('status')!r}")

    ok(f"run_quality_after_calibrate() executed with status={calibrate_payload.get('status')}")

    # -----------------------------------------------------------------
    # 7. Check output files
    # -----------------------------------------------------------------
    expected_outputs = [
        benchmarks / "quality_metrics_after_infer.json",
        benchmarks / "quality_metrics_after_infer.csv",
        reports / "quality_metrics_after_infer.md",
        benchmarks / "quality_metrics_after_train.json",
        benchmarks / "quality_metrics_after_train.csv",
        reports / "quality_metrics_after_train.md",
        benchmarks / "quality_metrics_after_calibrate.json",
        benchmarks / "quality_metrics_after_calibrate.csv",
        reports / "quality_metrics_after_calibrate.md",
    ]

    missing_outputs = [str(p) for p in expected_outputs if not p.exists()]
    if missing_outputs:
        die(f"Missing expected output files: {missing_outputs}")

    ok("Expected JSON/CSV/Markdown outputs were created")

    # -----------------------------------------------------------------
    # 8. CLI smoke test
    # -----------------------------------------------------------------
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "infer",
            "--run-parent",
            str(root),
        ],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        die("Standalone CLI smoke test failed")

    ok("Standalone CLI works")

print("✅ FULL SMOKE TEST PASSED")
PY

echo
echo "[3/7] Optional ruff check"
if command -v ruff >/dev/null 2>&1; then
  ruff check "$SCRIPT" || fail "ruff found issues"
  ok "ruff passed"
else
  warn "ruff not installed; skipping"
fi

echo
echo "[4/7] Optional black format check"
if command -v black >/dev/null 2>&1; then
  black --check "$SCRIPT" || fail "black formatting check failed"
  ok "black passed"
else
  warn "black not installed; skipping"
fi

echo
echo "======================================================================"
echo "✅ quality_metrics.py looks structurally correct and executable"
echo "======================================================================"
