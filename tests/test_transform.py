"""tests/test_transform.py — Data pipeline tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.utils.config import WHAS500Config, UCIHeartConfig


def _make_whas500(n=100):
    import random; random.seed(42); np.random.seed(42)
    cfg = WHAS500Config
    df = pd.DataFrame({
        'lenfol': np.random.randint(1, 3000, n).astype(float),
        'fstat':  np.random.randint(0, 2, n),
        'age':    np.random.uniform(30, 90, n),
        'bmi':    np.random.uniform(18, 40, n),
        'sysbp':  np.random.uniform(80, 200, n),
        'diasbp': np.random.uniform(50, 120, n),
        'hr':     np.random.uniform(50, 130, n),
        'glucose':np.random.uniform(60, 300, n),
        'los':    np.random.uniform(1, 30, n),
        'sex':    np.random.randint(0, 2, n),
        'chf':    np.random.randint(0, 2, n),
        'miord':  np.random.randint(1, 3, n),
        'mitype': np.random.randint(1, 4, n),
        'cvd':    np.random.randint(0, 2, n),
        'afb':    np.random.randint(0, 2, n),
        'sho':    np.random.randint(0, 2, n),
        'av3':    np.random.randint(0, 2, n),
    })
    return df


class TestCleanWHAS500:
    def test_returns_dataframe(self):
        from src.data.transform import clean_whas500
        df = clean_whas500(_make_whas500())
        assert isinstance(df, pd.DataFrame)

    def test_no_missing_after_clean(self):
        from src.data.transform import clean_whas500
        df = _make_whas500()
        df.loc[:5, 'bmi'] = np.nan
        cleaned = clean_whas500(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_mortality_col_added(self):
        from src.data.transform import clean_whas500
        cleaned = clean_whas500(_make_whas500())
        assert 'died' in cleaned.columns

    def test_mortality_col_is_binary(self):
        from src.data.transform import clean_whas500
        cleaned = clean_whas500(_make_whas500())
        assert set(cleaned['died'].unique()).issubset({0, 1})

    def test_rows_dropped_when_target_missing(self):
        from src.data.transform import clean_whas500
        df = _make_whas500(50)
        df.loc[:5, 'fstat'] = np.nan
        cleaned = clean_whas500(df)
        assert len(cleaned) < 50


class TestEngineerFeatures:
    def test_pulse_pressure_added(self):
        from src.data.transform import engineer_whas500_features, clean_whas500
        cleaned = clean_whas500(_make_whas500())
        df = engineer_whas500_features(cleaned)
        assert 'pulse_pressure' in df.columns

    def test_pulse_pressure_correct(self):
        from src.data.transform import engineer_whas500_features, clean_whas500
        cleaned = clean_whas500(_make_whas500())
        df = engineer_whas500_features(cleaned)
        expected = df['sysbp'] - df['diasbp']
        pd.testing.assert_series_equal(df['pulse_pressure'], expected, check_names=False)

    def test_age_group_added(self):
        from src.data.transform import engineer_whas500_features, clean_whas500
        cleaned = clean_whas500(_make_whas500())
        df = engineer_whas500_features(cleaned)
        assert 'age_group' in df.columns


class TestSplitData:
    def test_split_sizes(self):
        from src.data.transform import split_data, clean_whas500
        cleaned = clean_whas500(_make_whas500(100))
        X_tr, X_te, y_tr, y_te = split_data(
            cleaned, 'died', WHAS500Config.numeric_features
        )
        assert len(X_tr) + len(X_te) == len(cleaned)
        assert len(X_tr) > len(X_te)

    def test_stratification_preserves_balance(self):
        from src.data.transform import split_data, clean_whas500
        cleaned = clean_whas500(_make_whas500(200))
        _, _, y_tr, y_te = split_data(cleaned, 'died', WHAS500Config.numeric_features)
        assert abs(y_tr.mean() - y_te.mean()) < 0.15

    def test_no_overlap(self):
        from src.data.transform import split_data, clean_whas500
        cleaned = clean_whas500(_make_whas500(100))
        X_tr, X_te, _, _ = split_data(
            cleaned.reset_index(), 'died', ['index'] + WHAS500Config.numeric_features
        )
        assert len(set(X_tr['index']) & set(X_te['index'])) == 0
