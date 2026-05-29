# SMLM LabFlow

A modular pipeline wrapper for **Single-Molecule Localization Microscopy (SMLM)** analysis.

It handles the engineering overhead — QC, calibration, training, inference, export, benchmarking, and reporting — so you can focus on the science.

> **Status:** ready for lab use — actively developed  
> **Current backend:** LiteLoc  
> **Architecture:** backend-agnostic (DECODE, DeepSTORM, FD-DeepLoc adapters can be added)

---

## What it does

A typical SMLM run has many fragile manual steps. LabFlow wraps them into a single reproducible CLI:

```
calibrate → train → infer → QC → export → benchmark → report
```

---

## Quick start

### 1. Install dependencies

```bash
conda env create -f env_yamls/liteloc_env_base.yml
conda activate liteloc_env
```

### 2. Configure your machine

Edit `adapters/backend_paths.yml` to point at your local LiteLoc install:

```yaml
liteloc:
  root: /path/to/LiteLoc
```

### 3. Choose a profile

Copy and adapt `profiles/liteloc_unified_example.yaml` for your microscope setup.

### 4. Run the pipeline

```bash
# Step 1 — calibrate PSF from bead z-stack
python run_pipeline.py calibrate \
  -i /data/beads \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/my_calibration \
  -b liteloc

# Step 2 — train localization model
python run_pipeline.py train \
  -i /data/training_frames \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/my_training \
  -b liteloc

# Step 3 — run inference
python run_pipeline.py infer \
  -i /data/raw_movies \
  -p profiles/liteloc_unified_example.yaml \
  -o outputs/my_inference \
  -b liteloc
```

---

## Features

| Feature | Description |
|---|---|
| Input QC | Shape, dtype, axes, intensity stats, preview image, histogram |
| Calibration | `vector_beads`, `spline_file`, `analytic`, or `none` |
| Backend execution | Adapter-based: add new backends without touching the core |
| Canonical output | Unified localization CSV across all backends |
| Export formats | SMAP, Picasso, napari, Locan |
| Benchmarking | Runtime, memory, resolution, drift, CRLB/RMSE metrics |
| Reports | Markdown + HTML run reports |
| Registry | Tracks latest calibration/model/results for reuse across runs |
| Review helper | Separate napari/Locan viewer (`napari_locan_review.py`) |

---

## Project structure

```
smlm-labflow/
├── run_pipeline.py              # Main CLI: calibrate / train / infer
├── run_folders.py               # Run folder layout
├── qc_input.py                  # Input movie quality control
├── schema.py                    # Canonical localization schema
├── post_inference.py            # Backend output → canonical CSV
├── export_downstream.py         # Export to SMAP / Picasso / napari / Locan
├── benchmark.py                 # Runtime + scientific metrics
├── quality_metrics.py           # Automatic QC metrics
├── generate_run_report.py       # Markdown/HTML reports
├── combine_run_outputs.py       # Merge batch outputs
├── combine_benchmark_comparisons.py
├── napari_locan_review.py       # Manual review helper
│
├── adapters/
│   ├── backend_paths.yml        # ⚠ Machine-specific — not committed
│   ├── backend_paths.example.yml
│   ├── resolver.py              # Merges profile + paths + registry
│   └── liteloc_adapter.py      # LiteLoc calibrate/train/infer
│
├── profiles/                    # Scientific workflow profiles
├── env_yamls/                   # Conda environments
└── results/                     # Run outputs (git-ignored)
```

---

## Configuration: two files, two roles

| File | What goes in it |
|---|---|
| `adapters/backend_paths.yml` | **Where** software is installed on your machine |
| `profiles/*.yaml` | **What** scientific workflow to run (PSF type, pixel size, etc.) |

These are intentionally separate so that the same profile works on any machine that has the backend installed.

---

## Profiles

Profiles define the scientific configuration for a run:

- PSF type and dimensionality
- Calibration mode
- Pixel size and z-step
- LiteLoc YAML sections (calibration / training / inference)
- Export and QC options

Start from `profiles/liteloc_unified_example.yaml` — it uses `auto` placeholders for paths that LabFlow fills at runtime (bead file, model path, movie paths).

You can extend it for your specific setup:

```yaml
extends: liteloc_unified_example.yaml
profile_name: my_lab_condition_A

microscope:
  pixel_size_nm: 100

calibration:
  z_step_nm: 50
```

---

## Outputs

Each run creates a structured folder:

```
outputs/my_run/
├── results/
│   └── batches/<movie_id>/
│       ├── input_qc.json
│       ├── input_preview.png
│       ├── canonical_localizations.csv
│       ├── smap_localizations.csv
│       ├── picasso_localizations.csv
│       ├── napari_points.csv
│       └── locan_localizations.csv
├── benchmarks/
│   ├── runtime_benchmark.csv
│   ├── quality_metrics_benchmark.csv
│   └── comparison_ready_summary.csv
├── reports/
│   ├── run_report.md
│   └── run_report.html
└── registry/
    └── resolved_runtime_config.json
```

---

## Combining and comparing runs

```bash
# Merge outputs from multiple runs
python combine_run_outputs.py

# Compare benchmarks across runs
python combine_benchmark_comparisons.py results -o comparison_summary_all_runs.csv
```

---

## Syntax checks

```bash
python -m py_compile run_pipeline.py
python -m py_compile adapters/resolver.py
python -m py_compile adapters/liteloc_adapter.py
python -m py_compile qc_input.py post_inference.py benchmark.py
```

---

## Roadmap

- Additional backend adapters (DECODE, DeepSTORM, FD-DeepLoc)
- CRLB/RMSE reporting improvements
- PSF diagnostics
- Grid artifact analysis
- Profile templates for common SMLM setups
- Notebook-based demos
- Registry compatibility checks

---

## Limitations

- LiteLoc must be installed separately
- Profiles and backend YAMLs must match your microscope and PSF setup
- Some QC metrics are experimental
- Scientific validation is the responsibility of the user

---

## License

**MIT License** — see [`LICENSE`](LICENSE).

External tools (LiteLoc, downstream SMLM tools) remain governed by their own licenses.

---

## Citation

If you use this pipeline, please cite:

1. **SMLM LabFlow** (this repository)
2. The **backend localization tool** used (e.g., LiteLoc)
3. Any **downstream analysis tools** applied to results

```bibtex
@software{smlm_labflow_2026,
  author  = {Ibrahim, Rozad},
  title   = {SMLM LabFlow: a modular wrapper pipeline for Single-Molecule Localization Microscopy workflows},
  year    = {2026},
  url     = {https://github.com/rozadibrahim/smlm-labflow}
}
```

---

## Maintainer

**Rozad Ibrahim** — ESBS, University of Strasbourg
