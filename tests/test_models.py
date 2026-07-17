"""tests/test_models.py — ML model tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _make_data(n=200, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    X   = pd.DataFrame(rng.randn(n, n_features),
                       columns=[f'f{i}' for i in range(n_features)])
    y   = pd.Series((rng.randn(n) > 0.3).astype(int))
    return X, y


class TestBuildModels:
    def test_returns_four_models(self):
        from src.models.train import build_models
        models = build_models()
        assert len(models) == 4
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert 'lightgbm' in models

    def test_models_have_fit_predict(self):
        from src.models.train import build_models
        for name, model in build_models().items():
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict_proba')

    def test_models_fit_and_predict(self):
        from src.models.train import build_models
        X, y = _make_data()
        for name, model in build_models().items():
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            assert probs.shape == (len(X),)
            assert (probs >= 0).all() and (probs <= 1).all()


class TestEvaluateModel:
    def test_returns_expected_keys(self):
        from src.models.train import build_models
        from src.models.evaluate import evaluate_model
        X, y = _make_data()
        model = build_models()['random_forest']
        model.fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        assert 'roc_auc' in metrics
        assert 'f1' in metrics
        assert 'recall' in metrics
        assert 'brier_score' in metrics

    def test_roc_auc_in_valid_range(self):
        from src.models.train import build_models
        from src.models.evaluate import evaluate_model
        X, y = _make_data()
        model = build_models()['logistic_regression']
        model.fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        assert 0.0 <= metrics['roc_auc'] <= 1.0

    def test_compare_models_returns_dataframe(self):
        from src.models.train import build_models
        from src.models.evaluate import compare_models
        X, y = _make_data()
        fitted = {}
        for name, model in build_models().items():
            model.fit(X[:160], y[:160])
            fitted[name] = model
        comparison = compare_models(fitted, X[160:], y[160:])
        assert isinstance(comparison, pd.DataFrame)
        assert 'roc_auc' in comparison.columns
        assert len(comparison) == 4


class TestCrossValidation:
    def test_cv_returns_expected_keys(self):
        from src.models.train import build_models, cross_validate_model
        X, y = _make_data()
        model = build_models()['random_forest']
        cv_metrics = cross_validate_model(model, X, y, cv_folds=3)
        assert 'cv_roc_auc_mean' in cv_metrics
        assert 'cv_roc_auc_std'  in cv_metrics
        assert 'cv_f1_mean'      in cv_metrics

    def test_cv_auc_in_valid_range(self):
        from src.models.train import build_models, cross_validate_model
        X, y = _make_data()
        model = build_models()['logistic_regression']
        cv    = cross_validate_model(model, X, y, cv_folds=3)
        assert 0.0 <= cv['cv_roc_auc_mean'] <= 1.0
