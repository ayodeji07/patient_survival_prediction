"""
src/models/evaluate.py
────────────────────────────────────────────────────────────────
Model evaluation — computes all performance metrics and
generates comparison outputs.

Metrics computed
────────────────
  AUC-ROC             — overall discrimination ability.
                        The probability that the model ranks a
                        random positive above a random negative.
                        Threshold-independent.

  Average Precision   — area under the precision-recall curve.
                        More informative than AUC-ROC when the
                        positive class is rare.

  F1 score            — harmonic mean of precision and recall
                        at the default 0.5 threshold.

  Brier score         — mean squared error of probability forecasts.
                        Lower is better (0 = perfect, 1 = worst).
                        Measures calibration as well as discrimination.

  Calibration         — are the model's predicted probabilities
                        accurate?  A patient predicted at 70% risk
                        should die ~70% of the time.
                        Measured by plotting observed vs predicted
                        in decile bins.

Clinical context
────────────────
  For a mortality prediction model used in clinical triage,
  we care especially about:
    - High sensitivity (recall): we want to catch most deaths,
      even at the cost of some false alarms.
    - Calibration: a model that says 80% risk should mean
      ~80% of those patients die, not 50%.
    - AUC-ROC: overall discriminative performance across all
      possible clinical thresholds.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics for one model on the test set.

    Args:
        model:     Fitted sklearn estimator with predict_proba.
        X_test:    Test feature DataFrame.
        y_test:    True binary labels.
        threshold: Classification threshold (default 0.5).

    Returns:
        Dict with:
          roc_auc, average_precision, f1, precision, recall,
          accuracy, brier_score, specificity, balanced_accuracy,
          n_test, n_positive.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)
    y_true  = y_test.values

    # True negative rate (specificity = 1 - FPR)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "roc_auc":           round(float(roc_auc_score(y_true, y_prob)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "brier_score":       round(float(brier_score_loss(y_true, y_prob)), 4),
        "specificity":       round(float(specificity), 4),
        "threshold":         threshold,
        "n_test":            int(len(y_true)),
        "n_positive":        int(y_true.sum()),
    }

    return metrics


def find_optimal_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
) -> float:
    """Find the classification threshold that maximises a metric.

    The default 0.5 threshold is not always optimal — for
    mortality prediction we may want higher sensitivity
    (recall) at the cost of lower precision.

    Args:
        model:  Fitted sklearn estimator.
        X_val:  Validation feature DataFrame.
        y_val:  Validation labels.
        metric: Metric to optimise: "f1", "recall", "precision".

    Returns:
        Optimal threshold as a float in [0, 1].

    Example::

        threshold = find_optimal_threshold(model, X_val, y_val, metric="recall")
        predictions = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_prob     = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0.0
    best_thresh = 0.5

    metric_fn = {
        "f1":        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "recall":    lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    }.get(metric)

    if metric_fn is None:
        logger.warning("Unknown metric %r — using 0.5 threshold", metric)
        return 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score  = metric_fn(y_val.values, y_pred)
        if score > best_score:
            best_score  = score
            best_thresh = t

    logger.info(
        "Optimal threshold for %s = %.2f (score=%.4f)",
        metric, best_thresh, best_score,
    )
    return float(best_thresh)


def compare_models(
    models:   dict[str, Any],
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame.

    Args:
        models:  Dict of model_name → fitted sklearn estimator.
        X_test:  Test feature DataFrame.
        y_test:  Test labels.

    Returns:
        DataFrame with one row per model and columns for each metric,
        sorted by AUC-ROC descending.

    Example::

        comparison = compare_models(fitted_models, X_test, y_test)
        print(comparison[["model", "roc_auc", "average_precision", "f1"]])
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        rows.append(metrics)

    df = (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    # Reorder columns for readability
    first_cols = ["model", "roc_auc", "average_precision", "f1",
                  "recall", "precision", "specificity", "brier_score"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    return df


def compute_roc_curve_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, list[float]]:
    """Compute ROC curve points for plotting.

    Args:
        model:  Fitted estimator.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Dict with keys fpr, tpr, thresholds, auc.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    y_prob          = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr   = roc_curve(y_test.values, y_prob)
    auc             = roc_auc_score(y_test.values, y_prob)

    return {
        "fpr":        fpr.tolist(),
        "tpr":        tpr.tolist(),
        "thresholds": thr.tolist(),
        "auc":        round(float(auc), 4),
    }


def compute_calibration_data(
    model,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    n_bins:  int = 10,
) -> dict[str, list[float]]:
    """Compute calibration curve data (reliability diagram).

    A well-calibrated model's calibration curve lies on the
    diagonal (predicted probability ≈ observed frequency).

    Args:
        model:  Fitted estimator.
        X_test: Test features.
        y_test: True labels.
        n_bins: Number of bins for the calibration curve.

    Returns:
        Dict with mean_predicted_prob and fraction_of_positives lists.
    """
    from sklearn.calibration import calibration_curve

    y_prob = model.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(
        y_test.values, y_prob,
        n_bins=n_bins, strategy="quantile",
    )

    return {
        "mean_predicted_prob":  mean_pred.tolist(),
        "fraction_of_positives": fraction_pos.tolist(),
    }
