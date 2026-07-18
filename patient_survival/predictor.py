"""
patient_survival/predictor.py
────────────────────────────────────────────────────────────────
SurvivalPredictor — the public-facing prediction API.

This is the interface that external code (CLI, notebooks, other
projects) should use.  It hides all internal pipeline details
behind a clean, stable API.

Design
──────
SurvivalPredictor is stateful — it loads models once at
construction and reuses them for all subsequent predictions.
This means the first call may be slow (model loading) but
all subsequent calls are fast.

The predict() method accepts a DataFrame of patient features
and returns a DataFrame of predictions with confidence scores.
The explain() method adds SHAP-based explanations per patient.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import Paths, WHAS500Config, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SurvivalPredictor:
    """Patient mortality prediction with SHAP explainability.

    Loads a trained model and provides predict() and explain()
    methods for use in the CLI, notebooks, or other code.

    Args:
        model_name: Which model to use for predictions.
                    One of: logistic_regression, random_forest,
                    xgboost (default), lightgbm.
        dataset:    Dataset the model was trained on.
                    "whas500" (default) or "uci".

    Example::

        # Basic usage
        predictor = SurvivalPredictor(model_name="xgboost")
        results   = predictor.predict(patients_df)
        print(results[["patient_id", "mortality_risk", "risk_level"]])

        # With SHAP explanations
        explained = predictor.explain(patients_df.head(5))
        print(explained["explanations"][0])
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        dataset:    str = "whas500",
    ) -> None:
        self.model_name  = model_name
        self.dataset     = dataset
        self._model      = None
        self._scaler     = None
        self._explainer  = None
        self._feature_names: list[str] = []

        logger.info(
            "SurvivalPredictor initialised: model=%s, dataset=%s",
            model_name, dataset,
        )

    def _ensure_loaded(self) -> None:
        """Load model and scaler from disk on first use (lazy loading)."""
        if self._model is not None:
            return

        logger.info("Loading model: %s / %s", self.model_name, self.dataset)

        from src.models.train import load_model
        self._model = load_model(self.model_name, self.dataset)

        # Load the scaler used during training
        scaler_path = Paths.models_dir / f"{self.dataset}_scaler.pkl"
        if scaler_path.exists():
            import pickle
            with open(scaler_path, "rb") as fh:
                self._scaler = pickle.load(fh)
        else:
            logger.warning(
                "Scaler not found at %s — features will not be scaled. "
                "Results may be degraded for Logistic Regression.",
                scaler_path,
            )

        # Load feature names from training results
        results_path = Paths.results / f"ml_results_{self.dataset}.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            if self.model_name in results:
                pass   # feature names from training data columns

        logger.info("Model loaded")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same preprocessing used during training.

        Mirrors src.data.transform.prepare_whas500_for_ml /
        prepare_uci_for_ml exactly (feature engineering → one-hot
        encoding → scaling) so the columns the model sees at
        inference time match training precisely — same names,
        same derived features, same order.

        Args:
            df: Raw patient feature DataFrame.

        Returns:
            Preprocessed DataFrame ready for the model.
        """
        from src.utils.config import UCIHeartConfig
        from src.data.transform import engineer_whas500_features
        from src.data.features import encode_whas500, encode_uci_heart

        cfg = WHAS500Config if self.dataset == "whas500" else UCIHeartConfig

        # Ensure all raw clinical columns are present before deriving
        # engineered/encoded features from them.
        expected_raw = cfg.all_features()
        missing      = [c for c in expected_raw if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing features: {missing}\n"
                f"Expected: {expected_raw}"
            )

        if self.dataset == "whas500":
            df = engineer_whas500_features(df)
            df, feature_cols = encode_whas500(df)
        else:
            df, feature_cols = encode_uci_heart(df)

        X = df[feature_cols].copy()

        # Apply scaling if scaler is available
        if self._scaler is not None:
            numeric_in_X = [c for c in cfg.numeric_features if c in X.columns]
            X[numeric_in_X] = self._scaler.transform(X[numeric_in_X])

        return X

    def predict(
        self,
        df:        pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Predict mortality risk for a DataFrame of patients.

        Args:
            df:        DataFrame with clinical feature columns.
                       Column names must match WHAS500Config.all_features().
            threshold: Classification threshold (default 0.5).

        Returns:
            DataFrame with columns:
              mortality_risk  — predicted probability of death (0.0–1.0)
              predicted_class — 1 = high risk, 0 = low risk
              risk_level      — human-readable: Low / Moderate / High
              model           — model name used
            Row order matches the input DataFrame.

        Example::

            patients = pd.read_csv("new_patients.csv")
            results  = predictor.predict(patients)
            results.to_csv("predictions.csv", index=False)
        """
        self._ensure_loaded()
        X = self._preprocess(df)

        probs    = self._model.predict_proba(X)[:, 1]
        classes  = (probs >= threshold).astype(int)

        risk_levels = pd.cut(
            probs,
            bins   = [0, 0.3, 0.6, 1.0],
            labels = ["Low", "Moderate", "High"],
            include_lowest = True,
        )

        results = pd.DataFrame({
            "mortality_risk":  probs.round(4),
            "predicted_class": classes,
            "risk_level":      risk_levels,
            "model":           self.model_name,
        })

        logger.info(
            "Predicted %d patients: %d high risk (%.1f%%)",
            len(results),
            (results["risk_level"] == "High").sum(),
            (results["risk_level"] == "High").mean() * 100,
        )
        return results

    def explain(
        self,
        df:       pd.DataFrame,
        top_n:    int = 5,
    ) -> dict:
        """Generate SHAP explanations for each patient in df.

        Args:
            df:    DataFrame of patients to explain.
            top_n: Number of top features to include per patient.

        Returns:
            Dict with:
              predictions   — same as predict() output
              explanations  — list of per-patient explanation dicts
              global_importance — top features across all patients
        """
        import shap

        self._ensure_loaded()
        X = self._preprocess(df)

        if self._explainer is None:
            from src.models.shap_explainer import build_explainer
            self._explainer = build_explainer(self._model, X, self.model_name)

        from src.models.shap_explainer import (
            compute_shap_values,
            global_feature_importance,
            local_explanation,
        )

        shap_values = compute_shap_values(self._explainer, X, self.model_name)
        global_imp  = global_feature_importance(shap_values, list(X.columns))

        explanations = []
        for i in range(len(df)):
            expl = local_explanation(self._explainer, X, list(X.columns), i)
            explanations.append(expl)

        predictions = self.predict(df)

        return {
            "predictions":       predictions.to_dict("records"),
            "explanations":      explanations,
            "global_importance": global_imp[:top_n],
        }

    def batch_predict_csv(
        self,
        input_path:  Path,
        output_path: Path,
        explain:     bool = False,
    ) -> pd.DataFrame:
        """Read a CSV of patients, predict, and write results to CSV.

        Args:
            input_path:  Path to input CSV with clinical features.
            output_path: Path to write prediction results.
            explain:     Whether to include SHAP explanations.

        Returns:
            Results DataFrame (also written to output_path).
        """
        logger.info("Batch prediction: %s → %s", input_path, output_path)
        df = pd.read_csv(input_path)
        logger.info("Loaded %d patients from %s", len(df), input_path.name)

        if explain:
            result   = self.explain(df)
            out_df   = pd.DataFrame(result["predictions"])
        else:
            out_df   = self.predict(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.info(
            "Predictions written to %s (%d rows)", output_path, len(out_df)
        )
        return out_df
