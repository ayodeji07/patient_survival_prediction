"""
src/visualisation/plots.py
────────────────────────────────────────────────────────────────
Python-side visualisation for ML results and SHAP explanations.

These functions produce the charts shown in the Quarto report
and the Jupyter notebooks.  R handles the survival analysis
plots (KM curves, forest plot) — see r/plots.R.

Chart types
───────────
  ROC curves          — compare all four models on one axis
  Precision-Recall    — compare AP curves
  Calibration         — reliability diagram per model
  SHAP beeswarm       — global feature importance (dot plot)
  SHAP bar chart      — mean |SHAP| per feature
  SHAP waterfall      — single patient explanation
  SHAP dependence     — feature value vs SHAP (interaction plot)
  Model comparison    — side-by-side AUC/F1 bar chart
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.constants import MODEL_COLOURS, MODEL_DISPLAY_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── ROC curves ────────────────────────────────────────────────────

def plot_roc_curves(
    models:   dict[str, Any],
    X_test:   pd.DataFrame,
    y_test:   pd.Series,
    title:    str = "ROC Curves — Model Comparison",
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """Plot ROC curves for all models on one axis.

    Args:
        models:    Dict of model_name → fitted estimator.
        X_test:    Test features.
        y_test:    True labels.
        title:     Plot title.
        save_path: Save PNG to this path if provided.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        y_prob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test.values, y_prob)
        auc          = roc_auc_score(y_test.values, y_prob)
        colour       = MODEL_COLOURS.get(name, "#888")
        display_name = MODEL_DISPLAY_NAMES.get(name, name)

        ax.plot(fpr, tpr, colour, lw=2,
                label=f"{display_name} (AUC = {auc:.3f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)

    return fig


# ── Precision-Recall curves ───────────────────────────────────────

def plot_pr_curves(
    models:    dict[str, Any],
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    title:     str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """Plot Precision-Recall curves for all models.

    Args:
        models:    Dict of model_name → fitted estimator.
        X_test:    Test features.
        y_test:    True labels.
        title:     Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(8, 7))
    baseline = y_test.mean()

    for name, model in models.items():
        y_prob            = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test.values, y_prob)
        ap                = average_precision_score(y_test.values, y_prob)
        colour            = MODEL_COLOURS.get(name, "#888")
        display_name      = MODEL_DISPLAY_NAMES.get(name, name)

        ax.plot(recall, precision, colour, lw=2,
                label=f"{display_name} (AP = {ap:.3f})")

    ax.axhline(baseline, color="black", linestyle="--", lw=1, alpha=0.5,
               label=f"Baseline (prevalence = {baseline:.2f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curves saved to %s", save_path)

    return fig


# ── Model comparison bar chart ────────────────────────────────────

def plot_model_comparison(
    results:   dict[str, dict],
    metrics:   list[str] = None,
    title:     str = "Model Comparison",
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """Side-by-side bar chart comparing model performance metrics.

    Args:
        results:   Dict of model_name → results dict from train_all_models().
        metrics:   Metrics to compare (default: AUC, AP, F1, Recall).
        title:     Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["roc_auc", "average_precision", "f1", "recall"]

    metric_labels = {
        "roc_auc":           "AUC-ROC",
        "average_precision": "Avg Precision",
        "f1":                "F1",
        "recall":            "Recall",
        "precision":         "Precision",
        "brier_score":       "Brier Score",
    }

    model_names   = list(results.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_names]

    n_metrics = len(metrics)
    x = np.arange(len(model_names))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [
            results[m]["test_metrics"].get(metric, 0)
            for m in model_names
        ]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, values, width,
            label  = metric_labels.get(metric, metric),
            alpha  = 0.85,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison plot saved to %s", save_path)

    return fig


# ── SHAP visualisations ───────────────────────────────────────────

def plot_shap_summary(
    shap_values:   np.ndarray,
    X:             pd.DataFrame,
    feature_names: list[str],
    title:         str = "SHAP Summary — Global Feature Importance",
    save_path:     Optional[Path] = None,
    plot_type:     str = "dot",
) -> None:
    """SHAP summary plot — beeswarm (dot) or bar chart.

    Args:
        shap_values:   SHAP values (n_samples, n_features).
        X:             Feature DataFrame.
        feature_names: Feature column names.
        title:         Plot title.
        save_path:     Optional save path.
        plot_type:     "dot" (the beeswarm plot) or "bar" (mean |SHAP|).
                       These are shap.summary_plot()'s actual accepted
                       values — "beeswarm" is NOT one of them and silently
                       produces an empty plot instead of raising an error.
    """
    import shap
    import matplotlib.pyplot as plt
    from src.utils.config import SHAPConfig

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.title(title, fontsize=13, fontweight="bold", pad=15)

    shap.summary_plot(
        shap_values,
        X.values,
        feature_names = feature_names,
        plot_type     = plot_type,
        max_display   = SHAPConfig.top_n_features,
        show          = False,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)
        plt.close()


def plot_shap_waterfall(
    explainer,
    X_patient:     pd.DataFrame,
    feature_names: list[str],
    patient_idx:   int = 0,
    title:         str = "SHAP Waterfall — Patient Explanation",
    save_path:     Optional[Path] = None,
) -> None:
    """SHAP waterfall plot for a single patient.

    Shows how each feature pushed the prediction above or below
    the baseline (expected model output).

    Args:
        explainer:     SHAP explainer.
        X_patient:     Single-row or multi-row feature DataFrame.
        feature_names: Feature names.
        patient_idx:   Which patient to explain (row index).
        title:         Plot title.
        save_path:     Optional save path.
    """
    import shap
    import matplotlib.pyplot as plt

    single = X_patient.iloc[[patient_idx]]
    shap_vals = explainer.shap_values(single)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    explanation = shap.Explanation(
        values    = shap_vals[0],
        base_values = float(base_value),
        data      = single.values[0],
        feature_names = feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP waterfall saved to %s", save_path)
        plt.close()


def plot_shap_dependence(
    shap_values:     np.ndarray,
    X:               pd.DataFrame,
    feature:         str,
    interaction_feature: Optional[str] = None,
    title:           Optional[str]  = None,
    save_path:       Optional[Path] = None,
) -> None:
    """SHAP dependence plot showing feature value vs SHAP value.

    Args:
        shap_values:         SHAP values array.
        X:                   Feature DataFrame.
        feature:             Feature to plot on x-axis.
        interaction_feature: Feature to colour-code by (optional).
        title:               Plot title.
        save_path:           Optional save path.
    """
    import shap
    import matplotlib.pyplot as plt

    feat_idx = list(X.columns).index(feature)
    inter_idx = (
        list(X.columns).index(interaction_feature)
        if interaction_feature and interaction_feature in X.columns
        else "auto"
    )

    plt.figure(figsize=(9, 6))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X.values,
        feature_names         = list(X.columns),
        interaction_index     = inter_idx,
        show                  = False,
    )

    if title:
        plt.title(title, fontsize=12, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP dependence plot saved to %s", save_path)
        plt.close()
