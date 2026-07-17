"""tests/test_shap.py — SHAP explainability tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _make_fitted_model(n=200, n_features=8, seed=42):
    from src.models.train import build_models
    rng = np.random.RandomState(seed)
    X   = pd.DataFrame(rng.randn(n, n_features),
                       columns=[f'feat_{i}' for i in range(n_features)])
    y   = pd.Series((rng.randn(n) > 0.3).astype(int))
    model = build_models()['random_forest']
    model.fit(X[:160], y[:160])
    return model, X[:160], X[160:]


class TestBuildExplainer:
    def test_tree_explainer_for_rf(self):
        import shap
        from src.models.shap_explainer import build_explainer
        model, X_tr, _ = _make_fitted_model()
        explainer = build_explainer(model, X_tr, 'random_forest')
        assert explainer is not None

    def test_linear_explainer_for_lr(self):
        import shap
        from src.models.train import build_models
        from src.models.shap_explainer import build_explainer
        rng = np.random.RandomState(0)
        X   = pd.DataFrame(rng.randn(100, 5),
                            columns=[f'f{i}' for i in range(5)])
        y   = pd.Series((rng.randn(100) > 0.2).astype(int))
        model = build_models()['logistic_regression']
        model.fit(X, y)
        explainer = build_explainer(model, X, 'logistic_regression')
        assert explainer is not None


class TestComputeSHAPValues:
    def test_output_shape(self):
        from src.models.shap_explainer import build_explainer, compute_shap_values
        model, X_tr, X_te = _make_fitted_model()
        explainer = build_explainer(model, X_tr)
        shap_vals = compute_shap_values(explainer, X_te)
        assert shap_vals.shape == X_te.shape


class TestGlobalImportance:
    def test_returns_list_of_dicts(self):
        from src.models.shap_explainer import (
            build_explainer, compute_shap_values, global_feature_importance
        )
        model, X_tr, X_te = _make_fitted_model()
        explainer = build_explainer(model, X_tr)
        shap_vals = compute_shap_values(explainer, X_te)
        importance = global_feature_importance(shap_vals, list(X_te.columns), top_n=5)
        assert len(importance) == 5
        assert all('feature' in d and 'mean_abs_shap' in d for d in importance)

    def test_sorted_descending(self):
        from src.models.shap_explainer import (
            build_explainer, compute_shap_values, global_feature_importance
        )
        model, X_tr, X_te = _make_fitted_model()
        explainer = build_explainer(model, X_tr)
        shap_vals = compute_shap_values(explainer, X_te)
        importance = global_feature_importance(shap_vals, list(X_te.columns))
        values = [d['mean_abs_shap'] for d in importance]
        assert values == sorted(values, reverse=True)


class TestLocalExplanation:
    def test_returns_expected_keys(self):
        from src.models.shap_explainer import (
            build_explainer, local_explanation
        )
        model, X_tr, X_te = _make_fitted_model()
        explainer = build_explainer(model, X_tr)
        expl = local_explanation(explainer, X_te, list(X_te.columns), patient_idx=0)
        assert 'base_value' in expl
        assert 'prediction' in expl
        assert 'contributions' in expl
        assert 'top_risk_factors' in expl
