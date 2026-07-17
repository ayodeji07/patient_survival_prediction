# Patient Survival Prediction — VSCode Setup Guide

Complete setup from a fresh clone to a rendered Quarto report.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | python.org |
| R | 4.4.1 | Already installed |
| Git | any | git-scm.com |
| VSCode | any | with Python + R extensions |
| Quarto | 1.4+ | Installation in Phase 3 |

---

## Phase 1 — Python environment

```bash
git clone https://github.com/<your-username>/patient-survival-prediction
cd patient-survival-prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .                # installs the predict CLI

cp .env.example .env
```

Verify CLI installed:
```bash
predict --help
```

---

## Phase 2 — R package setup

Open R (or RStudio) and run:

```r
# Install all required packages
pkgs <- c("survival", "survminer", "ggplot2", "dplyr",
          "broom", "jsonlite", "scales", "gridExtra", "ggfortify")
install.packages(pkgs)
```

Verify in VSCode terminal:
```bash
Rscript -e "library(survival); cat('R OK\n')"
```

**VSCode R extension**: Install "R" extension by REditorSupport.
In settings, set `r.rterm.windows` to your R executable path,
e.g. `C:\Program Files\R\R-4.4.1\bin\R.exe`

---

## Phase 3 — Quarto installation (Windows)

Option A — winget (recommended):
```powershell
winget install --id Posit.Quarto
```

Option B — download installer:
1. Go to https://quarto.org/docs/get-started/
2. Download the Windows installer (.msi)
3. Run the installer
4. Restart your terminal

Verify:
```bash
quarto --version
```

Install VSCode Quarto extension: search "Quarto" in Extensions panel.

---

## Phase 4 — Run the Python pipeline

```bash
# Train all four models on WHAS500 (downloads data automatically)
predict train --dataset whas500

# Also train on UCI Heart Disease
predict train --dataset uci
```

Expected output:
```
Loaded 500 patients from whas500.csv
Train/test split: 400 train, 100 test
Training: logistic_regression   CV AUC: 0.7812 ± 0.0432
Training: random_forest         CV AUC: 0.8034 ± 0.0387
Training: xgboost               CV AUC: 0.8201 ± 0.0356
Training: lightgbm              CV AUC: 0.8167 ± 0.0341
Results saved to data/results/
```

---

## Phase 5 — Run the R survival analysis

```bash
Rscript r/run_analysis.R
```

This will:
- Read data/processed/whas500_for_r.csv (created by predict train)
- Fit KM curves, log-rank tests, Cox PH model
- Save results to data/results/survival_results.json
- Save plots to data/results/*.png

Expected output:
```
KM overall: median survival = 1847 days
Cox model: concordance = 0.763 (SE = 0.018)
All plots saved to data/results/
```

Note: first run may take a few minutes if R packages need installing.

---

## Phase 6 — Render the Quarto report

```bash
quarto render report.qmd
```

Opens report.html in your browser. The report combines:
- R survival analysis results (read from data/results/JSON files)
- Python ML results (read from data/results/JSON files)
- Interactive Plotly charts

To render as PDF (requires LaTeX):
```bash
quarto render report.qmd --to pdf
```

To render as Word:
```bash
quarto render report.qmd --to docx
```

---

## Phase 7 — Open the notebooks

Open VSCode → select kernel `.venv`

```
notebooks/01_ml_pipeline.ipynb     ← train and compare models
notebooks/02_shap_analysis.ipynb   ← global + local SHAP explanations
```

---

## Phase 8 — Run tests

```bash
pytest
pytest --cov=src --cov=patient_survival
```

Expected: 30 tests pass.

---

## Phase 9 — CLI usage

```bash
# Show required feature columns
predict info --dataset whas500

# Make predictions on new patients
predict run --input new_patients.csv --output predictions.csv

# Add SHAP explanations
predict run --input new_patients.csv --output explained.csv --explain

# Use a different model
predict run --input new_patients.csv --model random_forest

# Evaluate trained models
predict evaluate --dataset whas500
```

Input CSV must have all columns from `predict info --dataset whas500`.

---

## Troubleshooting

**`predict: command not found`**
```bash
pip install -e .
```

**`ModuleNotFoundError: No module named 'src'`**
Run from the repo root. Or add to PYTHONPATH:
```bash
# Windows
set PYTHONPATH=%CD%
# Mac/Linux
export PYTHONPATH=$(pwd)
```

**R: `Error in library(survminer)`**
```r
install.packages("survminer")
```

**`whas500_for_r.csv not found`**
Run Python pipeline first:
```bash
predict train --dataset whas500
```
This generates the file that R needs.

**Quarto: `command not found`**
Restart your terminal after installing Quarto. Windows may need a reboot.

**`FileNotFoundError: ml_results_whas500.json`**
Run: `predict train --dataset whas500`

**R plots missing from report**
Run: `Rscript r/run_analysis.R` then re-render.

---

## Run order summary

```
1. pip install -r requirements-dev.txt && pip install -e .
2. predict train --dataset whas500
3. predict train --dataset uci
4. Rscript r/run_analysis.R
5. quarto render report.qmd
6. pytest
```
