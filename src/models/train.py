"""
src/models/train.py
────────────────────────────────────────────────────────────────
ML training pipeline — trains four classifiers for patient
mortality prediction and saves the results.

Models trained
──────────────
  Logistic Regression  — interpretable linear baseline.
                         Coefficients have direct clinical meaning.
  Random Forest        — non-linear ensemble, naturally handles
                         feature interactions common in clinical data.
  XGBoost              — gradient boosted trees, strong out-of-the-box
                         performance, built-in handling of class imbalance.
  LightGBM             — fast gradient boosting, effective on
                         tabular clinical data with few samples.

Training strategy
─────────────────
  All models are evaluated with stratified k-fold cross-validation
  (k=5 by default) to get stable performance estimates on the small
  WHAS500 dataset.  Final models are trained on the full training set
  and evaluated on the held-out test set.

  Class imbalance is handled explicitly:
    - class_weight="balanced" for Logistic Regression and Random Forest
    - scale_pos_weight for XGBoost
    - is_unbalance=True for LightGBM

  This is important because WHAS500 has ~42% mortality — not severe
  imbalance, but enough to affect precision/recall tradeoffs.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.utils.config import ModelConfig, Paths, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Model registry ────────────────────────────────────────────────

def build_models(
    pos_weight: float = 1.0,
) -> dict[str, Any]:
    """Instantiate all four classifiers with configured hyperparameters.

    Args:
        pos_weight: Ratio of negative to positive samples.
                    Used to set class weights for imbalanced datasets.
                    Compute as: (n_negative / n_positive).

    Returns:
        Dict mapping model name → unfitted sklearn estimator.
    """
    import xgboost as xgb
    import lightgbm as lgb

    cfg = ModelConfig

    models = {
        "logistic_regression": LogisticRegression(
            C           = cfg.lr_C,
            max_iter    = cfg.lr_max_iter,
            solver      = cfg.lr_solver,
            class_weight = "balanced",   # handles class imbalance
            random_state = cfg.random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators     = cfg.rf_n_estimators,
            max_depth        = cfg.rf_max_depth,
            min_samples_leaf = cfg.rf_min_samples_leaf,
            class_weight     = "balanced",
            random_state     = cfg.random_seed,
            n_jobs           = -1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators     = cfg.xgb_n_estimators,
            max_depth        = cfg.xgb_max_depth,
            learning_rate    = cfg.xgb_learning_rate,
            subsample        = cfg.xgb_subsample,
            scale_pos_weight = pos_weight,   # balances positive class
            use_label_encoder = False,
            eval_metric      = "logloss",
            random_state     = cfg.random_seed,
            verbosity        = 0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators  = cfg.lgbm_n_estimators,
            max_depth     = cfg.lgbm_max_depth,
            learning_rate = cfg.lgbm_learning_rate,
            num_leaves    = cfg.lgbm_num_leaves,
            is_unbalance  = True,
            random_state  = cfg.random_seed,
            verbose       = -1,
        ),
    }

    logger.info(
        "Built %d models: %s",
        len(models), ", ".join(models.keys()),
    )
    return models


def cross_validate_model(
    model,
    X:       pd.DataFrame,
    y:       pd.Series,
    cv_folds: int = ModelConfig.cv_folds,
    seed:    int  = ModelConfig.random_seed,
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and return mean metrics.

    Stratified k-fold is used to ensure each fold has approximately
    the same class distribution as the full dataset.

    Args:
        model:    Unfitted sklearn estimator.
        X:        Feature DataFrame (training data).
        y:        Target Series.
        cv_folds: Number of CV folds.
        seed:     Random seed for fold assignment.

    Returns:
        Dict with mean and std of: roc_auc, average_precision,
        f1, precision, recall.

    Note:
        Cross-validation uses the training set only.
        Never pass test data here — that would be data leakage.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scoring = {
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",
        "f1":                "f1",
        "precision":         "precision",
        "recall":            "recall",
    }

    cv_results = cross_validate(
        model, X, y,
        cv       = cv,
        scoring  = scoring,
        n_jobs   = -1,
        return_train_score = False,
    )

    summary = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = round(float(scores.mean()), 4)
        summary[f"cv_{metric}_std"]  = round(float(scores.std()),  4)

    return summary


def train_all_models(
    X_train:  pd.DataFrame,
    X_test:   pd.DataFrame,
    y_train:  pd.Series,
    y_test:   pd.Series,
    dataset:  str = "whas500",
    save_models: bool = True,
) -> dict[str, dict]:
    """Train all four models, run cross-validation, and save results.

    Args:
        X_train:     Training features (scaled).
        X_test:      Test features (scaled).
        y_train:     Training labels.
        y_test:      Test labels.
        dataset:     Dataset name for logging and output files.
        save_models: Whether to pickle the fitted models to disk.

    Returns:
        Dict mapping model name → results dict containing:
          cv metrics (mean ± std across folds)
          test metrics (single held-out set)
          feature_importances (where available)
          training_time_s
    """
    from src.models.evaluate import evaluate_model

    # Compute positive class weight for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(
        "Class balance: %d negative, %d positive (pos_weight=%.2f)",
        n_neg, n_pos, pos_weight,
    )

    models   = build_models(pos_weight=pos_weight)
    results  = {}
    Paths.models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info("─" * 50)
        logger.info("Training: %s", name)
        t_start = time.perf_counter()

        # Cross-validation on training set
        logger.info("  Running %d-fold cross-validation...", ModelConfig.cv_folds)
        cv_metrics = cross_validate_model(model, X_train, y_train)
        logger.info(
            "  CV AUC: %.4f ± %.4f",
            cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"],
        )

        # Final fit on full training set
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t_start

        # Evaluate on held-out test set
        test_metrics = evaluate_model(model, X_test, y_test)
        logger.info(
            "  Test AUC: %.4f  AP: %.4f  F1: %.4f",
            test_metrics["roc_auc"],
            test_metrics["average_precision"],
            test_metrics["f1"],
        )

        # Feature importance (available for tree-based models)
        feature_importances = _extract_feature_importance(
            model, list(X_train.columns)
        )

        results[name] = {
            "model_name":          name,
            "dataset":             dataset,
            "cv_metrics":          cv_metrics,
            "test_metrics":        test_metrics,
            "feature_importances": feature_importances,
            "training_time_s":     round(train_time, 2),
            "n_train":             len(X_train),
            "n_test":              len(X_test),
        }

        # Save fitted model
        if save_models:
            model_path = Paths.models_dir / f"{dataset}_{name}.pkl"
            with open(model_path, "wb") as fh:
                pickle.dump(model, fh)
            logger.info("  Model saved to %s", model_path.name)

    # Save all results to JSON
    _save_results(results, dataset)
    _log_comparison_table(results)

    return results


def _extract_feature_importance(
    model,
    feature_names: list[str],
) -> list[dict] | None:
    """Extract feature importances where the model provides them.

    Args:
        model:         Fitted sklearn estimator.
        feature_names: Ordered list of feature column names.

    Returns:
        List of {'feature': str, 'importance': float} dicts sorted
        by importance descending, or None for models that don't
        provide importances (e.g. Logistic Regression uses
        coefficients instead, handled separately by SHAP).
    """
    importance_attr = None

    if hasattr(model, "feature_importances_"):
        # Random Forest, XGBoost, LightGBM
        importance_attr = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression — use absolute coefficient values
        importance_attr = np.abs(model.coef_[0])

    if importance_attr is None:
        return None

    pairs = sorted(
        zip(feature_names, importance_attr),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [
        {"feature": f, "importance": round(float(v), 6)}
        for f, v in pairs
    ]


def _save_results(results: dict, dataset: str) -> None:
    """Save training results to JSON."""
    Paths.results.mkdir(parents=True, exist_ok=True)
    out_path = Paths.results / f"ml_results_{dataset}.json"

    # Convert to JSON-serialisable form
    serialisable = {}
    for name, res in results.items():
        serialisable[name] = {
            k: v for k, v in res.items()
            if isinstance(v, (str, int, float, dict, list, type(None)))
        }

    out_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("ML results saved to %s", out_path)


def _log_comparison_table(results: dict) -> None:
    """Log a formatted comparison table of model performance."""
    logger.info("=" * 60)
    logger.info("  MODEL COMPARISON — TEST SET")
    logger.info("=" * 60)
    logger.info("  %-25s %-8s %-8s %-6s", "Model", "AUC", "AP", "F1")
    logger.info("  " + "─" * 50)
    for name, res in sorted(
        results.items(),
        key=lambda x: x[1]["test_metrics"]["roc_auc"],
        reverse=True,
    ):
        tm = res["test_metrics"]
        logger.info(
            "  %-25s %-8.4f %-8.4f %-6.4f",
            name, tm["roc_auc"], tm["average_precision"], tm["f1"],
        )
    logger.info("=" * 60)


def load_model(model_name: str, dataset: str = "whas500"):
    """Load a saved model from disk.

    Args:
        model_name: One of logistic_regression, random_forest,
                    xgboost, lightgbm.
        dataset:    Dataset the model was trained on.

    Returns:
        Fitted sklearn estimator.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Paths.models_dir / f"{dataset}_{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run the training pipeline first:\n"
            "  python -m src.models.train"
        )
    with open(model_path, "rb") as fh:
        return pickle.load(fh)
