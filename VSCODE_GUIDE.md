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
          "broom", "jsonlite", "scales", "gridExtra", "ggfortify",
          "knitr", "rmarkdown", "reticulate")  # needed by Quarto (Phase 6)
install.packages(pkgs)
```

If `library(knitr)` fails with `object 'attr' is not exported by
'namespace:xfun'`, your installed `knitr`/`rmarkdown` predate the
installed `xfun` version — re-run
`install.packages(c("knitr", "rmarkdown"))` to update them.

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

Expected output (WHAS500 -- real numbers from this pipeline):
```
WHAS500 loaded (bundled via scikit-survival): 500 patients, 16 columns, 43.0% died during follow-up
Data ready: 400 train, 100 test

Training: logistic_regression   CV AUC: 0.8198 ± 0.0250
Training: random_forest         CV AUC: 0.8286 ± 0.0205
Training: lightgbm              CV AUC: 0.7992 ± 0.0270
Training: xgboost               CV AUC: 0.7973 ± 0.0208

Model Comparison — Test Set
  logistic_regression   AUC=0.8796  F1=0.7955
  random_forest         AUC=0.8752  F1=0.7727
  lightgbm              AUC=0.8531  F1=0.7556
  xgboost               AUC=0.8527  F1=0.7865

Done. Training complete.
```

Logistic Regression has the best test AUC here (and is the CLI's default
model) -- CV AUC ranks Random Forest slightly higher, which is a normal
cross-validation vs. held-out-test-set discrepancy on a dataset this size,
not a bug.

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

Expected output (real numbers from this pipeline):
```
KM overall: median survival = 1627 days, 1yr = 72.4%, 5yr = 49.4%
Log-rank test (age_group): p = 0.0000 — SIGNIFICANT
Log-rank test (sex_label): p = 0.0052 — SIGNIFICANT
Cox model: concordance = 0.780 (SE = 0.016), AIC = 2262.6
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
notebooks/00_eda.ipynb             ← data dictionary, missingness, VIF, significance tests
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

**`'Rscript' is not recognized as an internal or external command`**
R is installed but its `bin` folder isn't on your PATH. Add it via
System Properties > Environment Variables > User variables > `Path`:
```
C:\Program Files\R\R-4.4.1\bin\x64
```
Then fully close and reopen VSCode (a new terminal tab in an
already-running window won't pick up the change) and re-verify with
`Rscript -e "cat('R OK\n')"`.

**R: `Error in sys.frame(1) : not that many frames on the stack`**
This shouldn't happen anymore (fixed in `r/utils.R` and `r/run_analysis.R`
to use `commandArgs()` instead of the fragile `sys.frame(1)$ofile`
idiom), but if you see it, it means a script is trying to find its own
directory using a method that only works when `source()`'d, not when
run directly via `Rscript`.

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
