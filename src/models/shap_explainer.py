"""
src/models/shap_explainer.py
────────────────────────────────────────────────────────────────
SHAP (SHapley Additive exPlanations) for model explainability.

Three levels of explanation
────────────────────────────
  Global — which features matter most across the entire dataset?
    - Beeswarm plot: each dot is one patient, x-axis is SHAP value,
      colour shows feature value (red=high, blue=low).
    - Bar chart: mean absolute SHAP values per feature.
    Shows the average impact of each feature on model predictions.

  Local — why did the model predict this specific patient's outcome?
    - Waterfall plot: starts at E[f(x)] (base value = mean prediction)
      and shows how each feature pushes the prediction up or down.
    - Force plot: compact HTML showing the same information.
    Allows a clinician to say "the model predicted high risk because
    this patient had high heart rate (+0.12) and a history of CHF (+0.09)".

  Interaction — how do two features interact?
    - Dependence plot: x=feature value, y=SHAP value, colour=interaction.
    Shows whether the effect of feature A depends on the value of feature B.
    E.g. "high glucose increases risk more strongly in older patients".

Explainer selection
───────────────────
  TreeExplainer   — exact, fast, for Random Forest / XGBoost / LightGBM.
                    Uses the model's tree structure directly.
  LinearExplainer — exact, for Logistic Regression.
                    Uses the linear model's coefficients.
  KernelExplainer — model-agnostic, slow.  Fallback when no
                    specialised explainer is available.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import Paths, SHAPConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_explainer(
    model,
    X_train:   pd.DataFrame,
    model_name: str = "",
):
    """Build the appropriate SHAP explainer for a model type.

    Chooses the fastest exact explainer for each model type:
      - Tree models → TreeExplainer (fast, exact)
      - Logistic Regression → LinearExplainer (exact)
      - Others → KernelExplainer (slow but universal)

    Args:
        model:      Fitted sklearn estimator.
        X_train:    Training data (used for background distribution
                    in KernelExplainer).
        model_name: Model name string for logging.

    Returns:
        SHAP explainer object.
    """
    import shap

    model_type = type(model).__name__.lower()

    if any(t in model_type for t in ["forest", "xgb", "lgbm", "boost", "tree"]):
        logger.info("Using TreeExplainer for %s", model_name or model_type)
        explainer = shap.TreeExplainer(model)

    elif "logistic" in model_type or "linear" in model_type:
        logger.info("Using LinearExplainer for %s", model_name or model_type)
        # LinearExplainer needs background data for the expected value
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer  = shap.LinearExplainer(model, background)

    else:
        logger.info(
            "Using KernelExplainer for %s (this may be slow)",
            model_name or model_type,
        )
        background = shap.sample(
            X_train, SHAPConfig.background_samples
        )
        explainer  = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            background,
        )

    return explainer


def compute_shap_values(
    explainer,
    X:         pd.DataFrame,
    model_name: str = "",
) -> np.ndarray:
    """Compute SHAP values for a feature matrix.

    Args:
        explainer:  SHAP explainer from build_explainer().
        X:          Feature DataFrame to explain.
        model_name: For logging.

    Returns:
        SHAP values array of shape (n_samples, n_features).
        Positive values push the prediction toward class 1 (death).
        Negative values push toward class 0 (survival).
    """
    import shap

    logger.info(
        "Computing SHAP values for %s (%d samples, %d features)...",
        model_name, len(X), X.shape[1],
    )

    shap_values = explainer.shap_values(X)

    # Older shap versions return a list [class_0_vals, class_1_vals] for
    # binary classifiers; newer versions (0.45+) return a single ndarray
    # of shape (n_samples, n_features, n_classes) instead.
    # Either way, we want the positive class (death = 1).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    logger.info(
        "SHAP values computed: shape=%s, mean_abs=%.4f",
        shap_values.shape,
        np.abs(shap_values).mean(),
    )
    return shap_values


def global_feature_importance(
    shap_values:   np.ndarray,
    feature_names: list[str],
    top_n:         int = SHAPConfig.top_n_features,
) -> list[dict[str, Any]]:
    """Compute mean absolute SHAP values per feature (global importance).

    Mean |SHAP value| is the standard measure of global feature
    importance in SHAP analysis.  It answers: "on average, how much
    does this feature affect the model's prediction?"

    Args:
        shap_values:   SHAP value array (n_samples, n_features).
        feature_names: Ordered list of feature names.
        top_n:         Number of top features to return.

    Returns:
        List of dicts with 'feature' and 'mean_abs_shap' keys,
        sorted by importance descending.

    Example::

        importance = global_feature_importance(shap_vals, feature_names)
        # → [{"feature": "age", "mean_abs_shap": 0.142}, ...]
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs    = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1], reverse=True,
    )
    return [
        {
            "feature":        feat,
            "mean_abs_shap":  round(float(val), 6),
            "rank":           i + 1,
        }
        for i, (feat, val) in enumerate(pairs[:top_n])
    ]


def local_explanation(
    explainer,
    X_sample:     pd.DataFrame,
    feature_names: list[str],
    patient_idx:  int = 0,
) -> dict[str, Any]:
    """Explain the prediction for one specific patient.

    Returns a waterfall-style explanation showing how each feature
    pushed the model's prediction above or below the baseline.

    Args:
        explainer:     SHAP explainer.
        X_sample:      Feature DataFrame containing the patient.
        feature_names: Ordered feature names.
        patient_idx:   Row index of the patient to explain.

    Returns:
        Dict with:
          base_value — model's average prediction (expected value)
          prediction — this patient's predicted probability
          contributions — list of {feature, value, shap_value, direction}
          top_risk_factors — features that increase predicted risk
          top_protective   — features that decrease predicted risk
    """
    import shap

    single_patient = X_sample.iloc[[patient_idx]]
    shap_vals      = compute_shap_values(explainer, single_patient)

    # Expected value (baseline prediction)
    base_value = float(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    # Patient's SHAP values
    patient_shap = shap_vals[0]
    prediction   = base_value + patient_shap.sum()

    contributions = []
    for feat, val, shap_val in zip(
        feature_names,
        single_patient.values[0],
        patient_shap,
    ):
        contributions.append({
            "feature":    feat,
            "value":      round(float(val), 4),
            "shap_value": round(float(shap_val), 6),
            "direction":  "increases_risk" if shap_val > 0 else "protective",
        })

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "patient_idx":      patient_idx,
        "base_value":       round(base_value, 4),
        "prediction":       round(float(prediction), 4),
        "contributions":    contributions[:10],   # top 10
        "top_risk_factors": [
            c for c in contributions if c["shap_value"] > 0
        ][:3],
        "top_protective":   [
            c for c in contributions if c["shap_value"] < 0
        ][:3],
    }


def interaction_analysis(
    shap_values:   np.ndarray,
    X:             pd.DataFrame,
    feature_names: list[str],
    top_n:         int = 5,
) -> list[dict[str, Any]]:
    """Identify the strongest feature interaction pairs.

    Uses the correlation between SHAP values and feature values
    to identify which features have the strongest interaction effects.

    Args:
        shap_values:   SHAP value array (n_samples, n_features).
        X:             Feature DataFrame.
        feature_names: Feature names.
        top_n:         Number of top interactions to return.

    Returns:
        List of dicts describing the top interaction pairs.
    """
    n_features = len(feature_names)
    interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Measure how much feature j's value correlates with
            # the SHAP values of feature i — this signals an interaction
            corr = float(np.corrcoef(
                shap_values[:, i],
                X.iloc[:, j].values,
            )[0, 1])
            interactions.append({
                "feature_a":   feature_names[i],
                "feature_b":   feature_names[j],
                "interaction": round(abs(corr), 4),
                "direction":   "positive" if corr > 0 else "negative",
            })

    return sorted(interactions, key=lambda x: -x["interaction"])[:top_n]


def run_full_shap_analysis(
    model,
    X_train:       pd.DataFrame,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    feature_names: list[str],
    model_name:    str,
    dataset:       str = "whas500",
) -> dict[str, Any]:
    """Run the complete SHAP analysis pipeline for one model.

    Computes global importance, local explanations for a sample of
    patients, and interaction analysis.  Saves results to JSON.

    Args:
        model:         Fitted sklearn estimator.
        X_train:       Training features (for explainer background).
        X_test:        Test features (for SHAP values).
        y_test:        True test labels.
        feature_names: Feature column names.
        model_name:    Model identifier string.
        dataset:       Dataset name for output file naming.

    Returns:
        Dict with global, local, and interaction results.
    """
    logger.info("─" * 50)
    logger.info("SHAP analysis: %s on %s", model_name, dataset)

    # Build explainer
    explainer   = build_explainer(model, X_train, model_name)

    # Compute SHAP values on test set
    shap_values = compute_shap_values(explainer, X_test, model_name)

    # Global importance
    global_imp  = global_feature_importance(shap_values, feature_names)
    logger.info(
        "Top 3 features: %s",
        ", ".join(f['feature'] for f in global_imp[:3])
    )

    # Local explanations for a sample of patients
    # Include both high-risk and low-risk patients for contrast
    y_prob     = model.predict_proba(X_test)[:, 1]
    high_risk  = np.argsort(y_prob)[-SHAPConfig.local_sample_size:]
    local_expl = [
        local_explanation(explainer, X_test, feature_names, int(idx))
        for idx in high_risk
    ]

    # Interaction analysis
    interactions = interaction_analysis(shap_values, X_test, feature_names)

    results = {
        "model_name":         model_name,
        "dataset":            dataset,
        "n_test_samples":     len(X_test),
        "feature_names":      feature_names,
        "global_importance":  global_imp,
        "local_explanations": local_expl,
        "interactions":       interactions,
        "shap_values_mean":   [
            round(float(v), 6)
            for v in shap_values.mean(axis=0)
        ],
        "shap_values_std":    [
            round(float(v), 6)
            for v in shap_values.std(axis=0)
        ],
    }

    # Save results
    out_path = Paths.results / f"shap_{dataset}_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("SHAP results saved to %s", out_path.name)

    # Save raw SHAP values as numpy array for plotting
    np.save(
        Paths.results / f"shap_values_{dataset}_{model_name}.npy",
        shap_values,
    )

    return results
