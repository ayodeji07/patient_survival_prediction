"""
src/utils/config.py
────────────────────────────────────────────────────────────────
Central configuration for the Patient Survival & Outcome
Prediction pipeline.

All paths, dataset settings, model hyperparameters, and runtime
constants live here.  Nothing else in the codebase hardcodes a
path or magic number.

Two datasets
────────────
  WHAS500 (Worcester Heart Attack Study)
    500 acute MI patients, real survival times (days to death),
    14 clinical features.  Used for:
      - R survival analysis (Kaplan-Meier, Cox PH)
      - Python ML mortality prediction
    Download: free, included in R survival package / direct URL

  UCI Heart Disease (Cleveland subset)
    303 patients, 13 clinical features, binary disease target.
    Used for:
      - Python ML binary classification benchmark
      - Model comparison across datasets
    Download: free, UCI ML Repository

R + Python integration
──────────────────────
  R scripts write results to data/results/ as JSON/CSV.
  Python reads those results for the combined Quarto report.
  This keeps the two runtimes completely independent.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Project root ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


class Paths:
    """All filesystem paths used by the pipeline."""

    # ── Data ──────────────────────────────────────────────────────
    raw:         Path = ROOT / "data" / "raw"
    processed:   Path = ROOT / "data" / "processed"
    results:     Path = ROOT / "data" / "results"

    # WHAS500 files
    whas500_csv:  Path = ROOT / "data" / "raw"  / "whas500.csv"
    whas500_proc: Path = ROOT / "data" / "processed" / "whas500_clean.parquet"

    # UCI Heart Disease files
    uci_csv:      Path = ROOT / "data" / "raw"  / "uci_heart.csv"
    uci_proc:     Path = ROOT / "data" / "processed" / "uci_heart_clean.parquet"

    # R output files (written by R, read by Python)
    r_survival_results: Path = ROOT / "data" / "results" / "survival_results.json"
    r_km_plot:          Path = ROOT / "data" / "results" / "km_curves.png"
    r_forest_plot:      Path = ROOT / "data" / "results" / "cox_forest.png"
    r_cox_coefficients: Path = ROOT / "data" / "results" / "cox_coefficients.csv"

    # ML model outputs
    ml_results:   Path = ROOT / "data" / "results" / "ml_results.json"
    shap_results: Path = ROOT / "data" / "results" / "shap_results.json"

    # Saved model artefacts
    models_dir:   Path = ROOT / "data" / "results" / "models"

    # Report
    report_qmd:   Path = ROOT / "report.qmd"
    report_html:  Path = ROOT / "report.html"

    @classmethod
    def ensure_all(cls) -> None:
        """Create all output directories if they do not exist."""
        for attr in ("raw", "processed", "results", "models_dir"):
            getattr(cls, attr).mkdir(parents=True, exist_ok=True)


class WHAS500Config:
    """WHAS500 dataset settings and feature definitions.

    WHAS500 = Worcester Heart Attack Study (500 patients, 1975–2001).
    All patients were admitted to hospital with acute myocardial infarction.
    Follow-up ranged from days to years; the outcome is all-cause mortality.

    Data source: bundled locally via scikit-survival
    (sksurv.datasets.load_whas500) — the standard Python distribution of
    this dataset. No network download required. Loaded in
    src/data/extract.py:load_whas500().

    Note: an earlier version of this config assumed a "glucose" feature
    and a 3-category "mitype" — neither exists in the real WHAS500 data.
    mitype is genuinely binary (0=non-Q-wave, 1=Q-wave) and there is no
    glucose measurement in this dataset.
    """

    # Time and event columns (survival analysis targets)
    time_col:   str = "lenfol"    # follow-up time in days
    event_col:  str = "fstat"     # 1 = died, 0 = censored (alive at last contact)

    # Binary mortality target (derived from fstat)
    mortality_col: str = "died"

    # Clinical feature columns
    numeric_features: list[str] = [
        "age",      # age at hospitalisation (years)
        "bmi",      # body mass index
        "sysbp",    # systolic blood pressure (mmHg)
        "diasbp",   # diastolic blood pressure (mmHg)
        "hr",       # initial heart rate (bpm)
        "los",      # length of hospital stay (days)
    ]

    categorical_features: list[str] = [
        "sex",      # 0=male, 1=female (source column "gender", renamed)
        "chf",      # congestive heart failure complication (0/1)
        "miord",    # MI order: 0=first, 1=recurrent
        "mitype",   # MI type: 0=non-Q-wave, 1=Q-wave
        "cvd",      # cardiovascular disease history (0/1)
        "afb",      # atrial fibrillation (0/1)
        "sho",      # cardiogenic shock complication (0/1)
        "av3",      # third-degree AV block (0/1)
    ]

    @classmethod
    def all_features(cls) -> list[str]:
        """Return all feature column names in a stable order."""
        return cls.numeric_features + cls.categorical_features


class UCIHeartConfig:
    """UCI Heart Disease (Cleveland) dataset settings."""

    download_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )

    # Column names (file has no header)
    columns: list[str] = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target",
    ]

    target_col: str = "target"   # 0 = no disease, 1-4 = disease (binarised to 0/1)

    numeric_features: list[str] = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]
    categorical_features: list[str] = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
    ]

    @classmethod
    def all_features(cls) -> list[str]:
        return cls.numeric_features + cls.categorical_features


class ModelConfig:
    """ML model hyperparameters.

    Defaults are conservative — they work well on small clinical
    datasets without overfitting.  Tune via .env or pass overrides
    to the training functions directly.
    """

    # Random seed — used everywhere for reproducibility
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    # Cross-validation folds
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))

    # Train / test split
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))

    # ── Logistic Regression ───────────────────────────────────────
    lr_C:       float  = float(os.getenv("LR_C",       "1.0"))
    lr_max_iter: int   = int(os.getenv("LR_MAX_ITER", "1000"))
    lr_solver:  str    = os.getenv("LR_SOLVER", "lbfgs")

    # ── Random Forest ─────────────────────────────────────────────
    rf_n_estimators: int   = int(os.getenv("RF_N_ESTIMATORS", "200"))
    rf_max_depth:    int   = int(os.getenv("RF_MAX_DEPTH",    "6"))
    rf_min_samples_leaf: int = int(os.getenv("RF_MIN_SAMPLES", "5"))

    # ── XGBoost ───────────────────────────────────────────────────
    xgb_n_estimators: int   = int(os.getenv("XGB_N_ESTIMATORS", "200"))
    xgb_max_depth:    int   = int(os.getenv("XGB_MAX_DEPTH",    "4"))
    xgb_learning_rate: float = float(os.getenv("XGB_LR",        "0.05"))
    xgb_subsample:    float  = float(os.getenv("XGB_SUBSAMPLE",  "0.8"))

    # ── LightGBM ──────────────────────────────────────────────────
    lgbm_n_estimators: int   = int(os.getenv("LGBM_N_ESTIMATORS", "200"))
    lgbm_max_depth:    int   = int(os.getenv("LGBM_MAX_DEPTH",    "4"))
    lgbm_learning_rate: float = float(os.getenv("LGBM_LR",        "0.05"))
    lgbm_num_leaves:   int   = int(os.getenv("LGBM_NUM_LEAVES",  "31"))


class SHAPConfig:
    """SHAP explainability settings."""

    # Number of background samples for SHAP KernelExplainer
    # (used for Logistic Regression — tree models use TreeExplainer)
    background_samples: int = int(os.getenv("SHAP_BG_SAMPLES", "100"))

    # Top N features to show in global summary plots
    top_n_features: int = int(os.getenv("SHAP_TOP_N", "10"))

    # Number of patients to show in local (waterfall) plots
    local_sample_size: int = int(os.getenv("SHAP_LOCAL_N", "5"))


class Settings:
    """Single entry point for all configuration sections."""

    paths:   Paths         = Paths
    whas500: WHAS500Config = WHAS500Config
    uci:     UCIHeartConfig = UCIHeartConfig
    model:   ModelConfig   = ModelConfig
    shap:    SHAPConfig    = SHAPConfig


settings = Settings()
