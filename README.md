# SourceBridge

Research code for **SourceBridge: Precision-Constrained Calibration Acquisition for Robust Truth Discovery**.

Authors: Yubao Deng, Xuechi Chen, Tian Wang, Houbing Herbert Song, Mianxiong Dong, and Anfeng Liu.

SourceBridge studies the cost of purchasing calibration reports to meet a declared target precision. This repository contains its mathematical implementations, experimental generators, frozen evaluation records, reproducibility checks, and editable figure sources.

## Contents

| Directory | Contents |
| --- | --- |
| `code/analysis` | Error metrics and sufficient statistics for manuscript tables |
| `code/experiments` | Procurement, inference, attacks, timing, baselines, frozen inputs and outputs |
| `code/diagrams` | Editable conceptual figures and licensed icons |
| `code/plots` | Experimental plots and complete plotting data |
| `code/validation` | Integrity checks, imports, geometry checks and validation records |

See [README_CODE.txt](README_CODE.txt) for the full reproduction guide in Chinese. Scripts resolve their resources relative to their own locations. Keep the directory structure when copying the code.

## Quick checks

Use Python 3.12+ for the complete scientific workflow. Dependencies are listed in the corresponding `requirements.txt` files. The basic integrity and error-metric checks use only the Python standard library.

```console
python code/validation/verify_code.py
python code/analysis/compute_error_metrics.py --predictions-csv code/experiments/data/public_inference_cells.csv --output-dir code/experiments/generated/error_metrics
```

After installing NumPy and SciPy:

```console
python code/validation/check_imports.py
python code/experiments/replay.py
python code/experiments/replay_procurement.py
python code/experiments/analyze.py
python code/validation/check_geometry.py
```

The inference replay executes 64 frozen conditions across all 15 interfaces: 960 predictions and 64 rational certificates. Complete-table analysis uses all 102,240 stored predictions and the declared resampling units. It does not create new independent experimental evidence.

## Figures

```console
python code/diagrams/regenerate.py
python code/plots/plot_experiments.py
```

Figure generation requires the listed plotting dependencies and Times New Roman. On non-Windows systems, set `SOURCEBRIDGE_FONT_DIR` to a legally installed Times New Roman font directory. The scripts also create local `manuscript_en/figures` and `manuscript_zh/figures` export directories; manuscript text and author photographs are not distributed here.

## Evidence and data scope

- The primary public-reference evaluation contains 213 targets per reference profile. The declared finite library has 14 conditions; two whole-packet translation controls are retained separately.
- Pooled RMSE weights every target-condition cell equally. It is a descriptive summary of the declared library, not an estimated real-world attack distribution.
- SourceBridge and the matched proper-prior MLNI interface have identical stored predictions on all 6,816 conditions. Their prediction-error metrics coincide; procurement and conditional guarantees are separate contributions.
- All 15 evaluated interfaces and all favorable and unfavorable recorded comparisons are retained.
- Compact extracted public values, complete predictions, procurement records, and timing observations are included. Re-extracting the original public archive requires the MSRA Urban Air `Data-1.zip` with the protocol's recorded hash. The external archive and bulky solver traces are not redistributed.
- Existing timing values are frozen observations. New timing measurements depend on the machine and must be stored separately.

`PUBLICATION_PROVENANCE.json` records normalization of local-machine path metadata. Numerical data and all scientific Python implementations are unchanged. Archived hashes identify original records; distribution manifests identify the public files.

## Attribution and contact

Repository contact: **1728837604@qq.com**.

Use the author names and manuscript title above when citing this work. No publication DOI is claimed. Third-party notices, including the Lucide icon license, remain with their respective resources. A general software reuse license has not been selected by the authors.
