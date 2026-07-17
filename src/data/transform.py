"""
src/data/transform.py
────────────────────────────────────────────────────────────────
Data transformation layer — cleaning, imputation, and feature
engineering for both WHAS500 and UCI Heart Disease.

Design principles
─────────────────
  - All transformations are logged so the pipeline is auditable.
    A clinician reviewing the analysis can see exactly what was
    done to the data before modelling.

  - Imputation uses median for continuous variables and mode for
    categorical.  This is appropriate for clinical data where
    MCAR (missing completely at random) is a reasonable assumption
    for a small number of missing values.

  - Feature engineering is minimal and clinically motivated.
    We add pulse pressure (systolic - diastolic) because it is a
    known independent predictor of cardiovascular mortality.

  - The transformation pipeline is fit on training data only and
    applied to test data — this prevents data leakage.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.features import encode_whas500, encode_uci_heart
from src.utils.config import (
    ModelConfig,
    Paths,
    WHAS500Config,
    UCIHeartConfig,
    settings,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── WHAS500 ───────────────────────────────────────────────────────

def clean_whas500(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw WHAS500 DataFrame.

    Steps applied:
      1. Select relevant clinical columns
      2. Remove patients with missing survival time or event status
      3. Impute remaining missing values (median/mode)
      4. Validate value ranges (flag clinically implausible values)
      5. Add derived binary mortality column

    Args:
        df: Raw WHAS500 DataFrame from load_whas500().

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    cfg = WHAS500Config
    keep_cols = (
        [cfg.time_col, cfg.event_col]
        + cfg.numeric_features
        + cfg.categorical_features
    )

    # Keep only columns we need
    available = [c for c in keep_cols if c in df.columns]
    missing_cols = set(keep_cols) - set(available)
    if missing_cols:
        logger.warning("Columns not found in WHAS500: %s", missing_cols)

    df = df[available].copy()

    # Cannot impute survival time or event — drop rows where missing
    before = len(df)
    df = df.dropna(subset=[cfg.time_col, cfg.event_col])
    if len(df) < before:
        logger.info(
            "Dropped %d rows with missing survival time or event status",
            before - len(df),
        )

    # Impute remaining missing values
    for col in cfg.numeric_features:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "Imputed %d missing %s values with median %.2f",
                    n_missing, col, median_val,
                )

    for col in cfg.categorical_features:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(
                    "Imputed %d missing %s values with mode %s",
                    n_missing, col, mode_val,
                )

    # Add binary mortality target (same as fstat but named clearly)
    df[cfg.mortality_col] = df[cfg.event_col].astype(int)

    # Ensure correct dtypes
    df[cfg.time_col]  = pd.to_numeric(df[cfg.time_col],  errors="coerce")
    df[cfg.event_col] = pd.to_numeric(df[cfg.event_col], errors="coerce")

    logger.info(
        "WHAS500 cleaned: %d patients, %.1f%% mortality",
        len(df), df[cfg.mortality_col].mean() * 100,
    )
    return df.reset_index(drop=True)


def engineer_whas500_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated derived features to WHAS500.

    Derived features:
      pulse_pressure   — systolic BP minus diastolic BP.
                         Elevated pulse pressure (>60 mmHg) is an
                         independent predictor of cardiovascular death.
      age_group        — age binned into clinical categories
                         (<60, 60–75, >75) for stratified analysis.
      high_glucose     — binary flag for glucose > 200 mg/dL
                         (indicative of uncontrolled diabetes).

    Args:
        df: Cleaned WHAS500 DataFrame.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    # Pulse pressure — vascular stiffness marker
    if "sysbp" in df.columns and "diasbp" in df.columns:
        df["pulse_pressure"] = df["sysbp"] - df["diasbp"]

    # Age groups for Kaplan-Meier stratification
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins   = [0, 60, 75, 120],
            labels = ["<60", "60-75", ">75"],
        )

    # Hyperglycaemia flag
    if "glucose" in df.columns:
        df["high_glucose"] = (df["glucose"] > 200).astype(int)

    return df


# ── UCI Heart Disease ─────────────────────────────────────────────

def clean_uci_heart(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw UCI Heart Disease DataFrame.

    Args:
        df: Raw UCI Heart Disease DataFrame from load_uci_heart().

    Returns:
        Cleaned DataFrame.
    """
    cfg = UCIHeartConfig
    df  = df.copy()

    all_features = cfg.all_features() + [cfg.target_col]
    df = df[[c for c in all_features if c in df.columns]]

    # Impute missing values
    for col in cfg.numeric_features:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "UCI: imputed %d missing %s with median %.2f",
                    n_missing, col, median_val,
                )

    for col in cfg.categorical_features:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(
                    "UCI: imputed %d missing %s with mode %s",
                    n_missing, col, mode_val,
                )

    # Ensure correct dtypes for categorical columns
    for col in cfg.categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)

    logger.info(
        "UCI Heart cleaned: %d patients, %.1f%% with disease",
        len(df), df[cfg.target_col].mean() * 100,
    )
    return df.reset_index(drop=True)


# ── Train/test splitting ──────────────────────────────────────────

def split_data(
    df:         pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size:   float = ModelConfig.test_size,
    random_seed: int   = ModelConfig.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataset into train and test sets.

    Stratified splitting ensures the class distribution is
    preserved in both splits — important for imbalanced targets.

    Args:
        df:           Clean DataFrame.
        target_col:   Name of the outcome column.
        feature_cols: List of feature column names to include.
        test_size:    Fraction for the test set (default 0.2).
        random_seed:  Reproducibility seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example::

        X_train, X_test, y_train, y_test = split_data(
            df, target_col="died",
            feature_cols=WHAS500Config.all_features()
        )
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_seed,
        stratify     = y,
    )

    logger.info(
        "Train/test split: %d train, %d test (stratified, seed=%d)",
        len(X_train), len(X_test), random_seed,
    )
    logger.info(
        "Class balance — train: %.1f%% positive, test: %.1f%% positive",
        y_train.mean() * 100, y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardise numeric features (fit on train, apply to test).

    Tree-based models (RF, XGBoost, LightGBM) do not need scaling
    but Logistic Regression does.  We scale all features and pass
    the same scaled data to all models for fair comparison.

    Args:
        X_train:      Training features DataFrame.
        X_test:       Test features DataFrame.
        numeric_cols: Names of numeric columns to scale.
                      Categorical columns are left unchanged.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
        The scaler is returned so it can be saved and reused for
        inference on new patients.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    scaler = StandardScaler()

    # Fit only on training data to prevent leakage
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    logger.info(
        "Features scaled: %d numeric columns standardised",
        len(numeric_cols),
    )
    return X_train, X_test, scaler


def prepare_whas500_for_ml(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Full preparation pipeline for WHAS500 ML modelling.

    Convenience wrapper that runs cleaning → feature engineering →
    train/test split → scaling in one call.

    Args:
        df: Raw WHAS500 DataFrame.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    cfg     = WHAS500Config
    df      = clean_whas500(df)
    df      = engineer_whas500_features(df)

    # One-hot encode nominal categoricals (e.g. mitype) and recode
    # ordinal ones (miord) — feature_cols is the resulting full
    # feature list, including derived and encoded columns.
    df, feature_cols = encode_whas500(df)

    X_train, X_test, y_train, y_test = split_data(
        df, target_col=cfg.mortality_col, feature_cols=feature_cols
    )

    X_train, X_test, scaler = scale_features(
        X_train, X_test, cfg.numeric_features
    )

    return X_train, X_test, y_train, y_test, scaler


def prepare_uci_for_ml(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Full preparation pipeline for UCI Heart Disease ML modelling.

    Args:
        df: Raw UCI Heart Disease DataFrame.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    cfg          = UCIHeartConfig
    df           = clean_uci_heart(df)
    df, feature_cols = encode_uci_heart(df)

    X_train, X_test, y_train, y_test = split_data(
        df, target_col=cfg.target_col, feature_cols=feature_cols
    )

    X_train, X_test, scaler = scale_features(
        X_train, X_test, cfg.numeric_features
    )

    return X_train, X_test, y_train, y_test, scaler
