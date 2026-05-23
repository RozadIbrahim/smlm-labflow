
# SMLM LabFlow      

**SMLM LabFlow** is an early-stage wrapper pipeline for **Single-Molecule Localization Microscopy (SMLM)** analysis.

The project currently integrates **LiteLoc** as the first backend, but the architecture is intentionally backend-agnostic: future adapters can be added for other localization engines such as DECODE, DeepSTORM-style tools, FD-DeepLoc-style tools, or other lab-specific backends.

> Status: **alpha / beta research prototype**  
> Current backend: **LiteLoc**  
> Design goal: **calibrate → train → infer → QC → export → benchmark → report**

---

## Purpose

SMLM workflows often require many fragile manual steps: checking raw TIFF files, configuring PSF calibration, training models, running inference, converting localization tables, benchmarking runtime, and preparing outputs for downstream tools.

**SMLM LabFlow** provides a reproducible orchestration layer around those steps.

It is not a replacement for expert microscopy validation. It is a lab-oriented engineering wrapper designed to make SMLM runs easier to launch, inspect, compare, and document.

---

## Current features

- TIFF / OME-TIFF input discovery
- Input quality control:
  - shape
  - dtype
  - axes guess
  - intensity statistics
  - preview image
  - histogram
- Backend execution through adapters
- Current LiteLoc support:
  - calibration
  - training
  - inference
- Calibration modes:
  - `vector_beads`
  - `spline_file`
  - `none`
  - `analytic`
- Canonical localization CSV output
- Downstream exports:
  - SMAP-style CSV
  - Picasso-style CSV
  - napari points CSV
  - Locan-compatible CSV
- Runtime/resource benchmarking
- Human-readable Markdown/HTML run reports
- Registry tracking for latest calibration/model/results
- Separate napari/Locan review helper

---

## Basic CLI

The user-facing interface is intentionally small:

```bash
python run_pipeline.py calibrate -i INPUT -p PROFILE -o RUN_FOLDER -b BACKEND
python run_pipeline.py train     -i INPUT -p PROFILE -o RUN_FOLDER -b BACKEND
python run_pipeline.py infer     -i INPUT -p PROFILE -o RUN_FOLDER -b BACKEND
```

`INPUT` can be a single file or, where noted, a parent folder:

* `infer` searches parent folders recursively and runs one inference batch per
  discovered TIFF/OME-TIFF
* `train` runs one training job for one training input; a folder is treated as
  one dataset/context, not expanded into separate jobs
* `calibrate` runs one calibration job for one PSF/profile condition; if a
  folder contains multiple candidate TIFF/.mat/.h5 files, it refuses to guess

For calibration, point `-i` to the exact bead stack or calibration artifact you
want to use:

```bash
python run_pipeline.py calibrate -i BEAD_STACK.ome.tif -p PROFILE -o RUN_FOLDER
python run_pipeline.py calibrate -i PSF_MODEL.h5       -p PROFILE -o RUN_FOLDER
```

If a bead dataset contains multiple independent bead stacks for the same
profile condition, use candidate mode:

```bash
python run_pipeline.py calibrate \
  -i BEAD_STACK_PARENT \
  -p PROFILE \
  -o RUN_FOLDER \
  --multi-files
```

Candidate mode runs calibration once per discovered bead stack/artifact and
writes a manifest for comparison. It deliberately does **not** update
`latest_calibration.json`; choose the final calibration before training.

Use `--max-files N` with inference or `calibrate --multi-files` for a quick
recursive smoke test.

For now, the main backend is:

```bash
-b liteloc
```

Example inference run:

```bash
python run_pipeline.py infer \
  -i /path/to/raw_movies \
  -p profiles/astigmatic_3d_vector_beads.yaml \
  -o results/infer_test_001 \
  -b liteloc
```

Quick test on one discovered input:

```bash
python run_pipeline.py infer \
  -i /path/to/raw_movies \
  -p profiles/astigmatic_3d_vector_beads.yaml \
  -o results/test_run \
  -b liteloc \
  --max-files 1
```

---

## Project structure

```text
smlm-labflow/
│
├── run_pipeline.py              # Main CLI: calibrate / train / infer
├── run_folders.py               # Run folder creation and organization
├── benchmark.py                 # Runtime/resource/scientific benchmarking
├── qc_input.py                  # Raw input movie QC
├── schema.py                    # Canonical localization schema
├── post_inference.py            # Raw backend output → canonical output
├── export_downstream.py         # SMAP / Picasso / napari / Locan exports
├── combine_run_outputs.py       # Combine batch outputs
├── combine_benchmark_comparisons.py # Combine comparison rows across runs
├── generate_run_report.py       # Markdown/HTML run reports
├── quality_metrics.py           # Automatic scientific QC metrics
├── napari_locan_review.py       # Separate manual review helper
│
├── adapters/
│   ├── backend_paths.yml        # Machine-specific backend paths
│   ├── resolver.py              # Resolves profile + backend paths + registry
│   └── liteloc_adapter.py       # LiteLoc calibrate/train/infer adapter
│
├── profiles/                    # Scientific workflow profiles
├── env_yamls/                   # Conda environment files
└── results/                     # Generated runs, ignored by Git
```

---

## Design philosophy

The project separates machine configuration, scientific configuration, backend execution, and reporting.

| File / Layer                 | Role                                             |
| ---------------------------- | ------------------------------------------------ |
| `run_pipeline.py`            | Main orchestrator                                |
| `adapters/backend_paths.yml` | Machine-specific backend paths                   |
| `profiles/*.yaml`            | Scientific workflow settings                     |
| `adapters/resolver.py`       | Merges backend paths, profile, and registry      |
| `adapters/*_adapter.py`      | Backend-specific calibrate/train/infer execution |
| `qc_input.py`                | Raw input QC                                     |
| `post_inference.py`          | Canonicalization and export coordination         |
| `quality_metrics.py`         | Automatic scientific QC metrics                  |
| `benchmark.py`               | Runtime/resources plus comparison-grade metrics  |
| `generate_run_report.py`     | Human-readable reports                           |

Important rule:

```text
backend_paths.yml = where software is installed
profiles/*.yaml  = what scientific workflow to run
resolver.py      = connects paths + profile + registry
adapter.py       = executes backend steps only
benchmark.py     = measures runtime/resources and comparison metrics
```

---

## Backend configuration

Machine-specific backend configuration lives in:

```text
adapters/backend_paths.yml
```

Example for LiteLoc:

```yaml
liteloc:
  root: /path/to/LiteLoc

  modules:
    vector_calibration: utils/vectorpsf_fit.py
    spline_calibration_io: spline_psf/calibration_io.py
    train: network/loc_model.py
    infer: network/multi_process.py

  functions:
    vector_calibration: beads_psf_calibrate
    spline_loader_class: SMAPSplineCoefficient
    train_class: LocModel
    infer_class: CompetitiveSmlmDataAnalyzer_multi_producer

  execution:
    vector_calibration: function
    spline_calibration_io: module
    train: module
    infer: module

  supported_calibration_modes:
    - vector_beads
    - spline_file
    - none
    - analytic
```

This file should not contain experiment-specific science settings.

---

## Profiles

Scientific workflow settings live in:

```text
profiles/*.yaml
```

Profiles define things like:

* backend name
* PSF type
* calibration mode
* pixel size
* LiteLoc calibration/training/inference runtime YAML sections
* inference parameters
* downstream export options
* quality-control options
* report options

Changing the profile changes the scientific route.

For LiteLoc, prefer starting from:

```text
profiles/liteloc_unified_example.yaml
```

That profile is intentionally path-free. It contains the LiteLoc YAML sections
that would otherwise live in separate demo files, and uses `auto` placeholders
for values LabFlow fills at runtime:

* calibration input path from the `calibrate -i` argument
* training output folder from the current training run
* latest calibration artifact from `outputs/registry/latest_calibration.json`
* inference input/output paths from each movie batch
* latest model artifact from `outputs/registry/latest_model.json`

The normal three-step workflow stays as three independent shell commands:

```bash
python run_pipeline.py calibrate -i DATA_BEADS -p profiles/liteloc_unified_example.yaml -o outputs/my_calibration
python run_pipeline.py train     -i DATA_TRAIN -p profiles/liteloc_unified_example.yaml -o outputs/my_training
python run_pipeline.py infer     -i DATA_MOVIES -p profiles/liteloc_unified_example.yaml -o outputs/my_inference --max-files 1
```

`adapters/backend_paths.yml` is still the only place that should contain the
local LiteLoc install path for a lab machine.

Smaller lab-specific profiles can inherit from the unified profile and override
only changed values:

```yaml
extends: liteloc_unified_example.yaml
profile_name: my_lab_condition_A

microscope:
  pixel_size_nm: 100

calibration:
  z_step_nm: 50
```

This makes the workflow transferable across labs when each machine provides:

* a compatible LiteLoc installation in `adapters/backend_paths.yml`
* the same profile YAML committed or archived with the run
* the calibration/model artifacts recorded in the registry or regenerated by
  running the three steps again
* matching Python/CUDA/LiteLoc dependency versions for strict reproducibility

Example profile concept:

```yaml
profile_name: astigmatic_3d_vector_beads

backend:
  name: liteloc

experiment:
  psf_type: astigmatic
  dimensionality: 3d

calibration:
  mode: vector_beads

liteloc:
  runtime_yaml:
    calibration:
      psf_params_dict: {}
      camera_params_dict: {}
      calib_params_dict: {}
      beads_file_name: auto
    train:
      Camera: {}
      PSF_model: {}
      Training: {}
    infer:
      Loc_Model:
        model_path: auto
      Multi_Process:
        image_path: auto
        save_path: auto

downstream:
  export_smap: true
  export_picasso: true
  export_napari: true
  export_locan: true
```

---

## Calibration modes

| Mode           | Meaning                                                             |
| -------------- | ------------------------------------------------------------------- |
| `vector_beads` | Fit/register PSF calibration from bead z-stack data                 |
| `spline_file`  | Register an existing SMAP/MATLAB/DECODE `.mat` or `.h5` calibration |
| `none`         | No external calibration file required                               |
| `analytic`     | PSF defined analytically in the profile/backend YAML                |

---

## Output layout

Each run creates a parent folder:

```text
results/my_run/
├── results/
├── benchmarks/
├── reports/
└── registry/
```

Typical inference outputs:

```text
results/
  runtime_liteloc_calibration.yaml    # calibrate
  liteloc_calibration_adapter_status.json
  calibration_candidates_manifest.csv # calibrate --multi-files
  calibration_candidates/
    <candidate_id>/
      runtime_liteloc_calibration.yaml
      liteloc_calibration_adapter_status.json
  runtime_liteloc_train.yaml          # train
  liteloc_training_adapter_status.json
  batches/
    <movie_id>/
      input_qc.json
      input_preview.png
      input_histogram.png
      liteloc_raw_output.csv
      canonical_localizations.csv
      smap_localizations.csv
      picasso_localizations.csv
      napari_points.csv
      locan_localizations.csv

benchmarks/
  runtime_benchmark.csv
  resource_benchmark.csv
  machine_specs.json
  machine_specs.csv
  localization_qc_benchmark.csv
  resolution_benchmark.csv
  drift_benchmark.csv
  quality_metrics_benchmark.csv
  comparison_ready_summary.csv
  benchmark_summary.json

reports/
  run_report.md
  run_report.html

registry/
  resolved_runtime_config.json
  latest_results.json
```

---

## Registry

The registry stores reusable artifacts across runs:

```text
latest_calibration.json
latest_model.json
latest_results.json
```

This allows later runs to reuse the latest compatible calibration or model without forcing users to type long paths in the CLI.

---

## Development checks

Run basic syntax checks:

```bash
python -m py_compile run_pipeline.py
python -m py_compile adapters/resolver.py
python -m py_compile adapters/liteloc_adapter.py
python -m py_compile qc_input.py
python -m py_compile post_inference.py
python -m py_compile benchmark.py
python -m py_compile combine_benchmark_comparisons.py
```

Combine many runs into one comparison table:

```bash
python combine_benchmark_comparisons.py results -o comparison_summary_all_runs.csv
```

Optional:

```bash
pre-commit run --all-files
```

---

## Roadmap

Possible future extensions:

* additional backend adapters
* stronger `quality_metrics.py`
* better PSF diagnostics
* CRLB/RMSE reporting
* grid artifact analysis
* profile templates for common SMLM setups
* richer benchmark figures
* notebook-based demonstrations
* improved registry compatibility checks

---

## Limitations

This project is still in alpha/beta stage.

Known limitations:

* LiteLoc must currently be installed separately.
* Profiles and backend YAMLs must match the microscope and PSF setup.
* Some QC metrics are experimental.
* Interactive napari/Locan review is separate from the main pipeline.
* Scientific validation remains the responsibility of the user.

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the full license text.

External tools and dependencies, including LiteLoc and downstream SMLM tools, remain governed by their own licenses and citation requirements.

---

## Citation and attribution

This repository is an orchestration/wrapper project.

How to cite

If you use this repository, please cite both:

SMLM LabFlow, as the wrapper/orchestration pipeline.
The original scientific tools used by the run, especially the backend localization method and downstream analysis tools.

Suggested citation:

<Your Last Name>, <Your First Name>. SMLM LabFlow: a modular wrapper pipeline for Single-Molecule Localization Microscopy workflows. GitHub repository, 2026. Available at: <repository URL>.

BibTeX:

@software{smlm_labflow_2026,
  author       = {<Your Last Name>, <Your First Name>},
  title        = {SMLM LabFlow: a modular wrapper pipeline for Single-Molecule Localization Microscopy workflows},
  year         = {2026},
  publisher    = {GitHub},
  url          = {<repository URL>},
  note         = {Alpha/beta research prototype}
}

Please also cite LiteLoc and any downstream tools used to generate or analyze results.
---

## Maintainer

```text
Rozad Ibrahim
ESBS at Strasbourg's University
IbrahimLabs
```

[1]: https://www.wired.com/2013/07/github-licenses?utm_source=chatgpt.com "GitHub Helps Clueless Coders Go Open Source"
