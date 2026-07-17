"""tests/test_cli.py — CLI tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typer.testing import CliRunner
from patient_survival.cli import app


runner = CliRunner()


class TestCLIInfo:
    def test_info_whas500_exits_ok(self):
        result = runner.invoke(app, ["info", "--dataset", "whas500"])
        assert result.exit_code == 0

    def test_info_uci_exits_ok(self):
        result = runner.invoke(app, ["info", "--dataset", "uci"])
        assert result.exit_code == 0

    def test_info_shows_feature_names(self):
        result = runner.invoke(app, ["info", "--dataset", "whas500"])
        assert "age" in result.output
        assert "bmi" in result.output


class TestCLIRun:
    def test_missing_input_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, [
            "run",
            "--input", str(tmp_path / "nonexistent.csv"),
            "--output", str(tmp_path / "out.csv"),
        ])
        assert result.exit_code != 0

    def test_run_with_valid_csv(self, tmp_path):
        """Test predict run end-to-end with a minimal CSV.
        
        This test is skipped if no trained model is present,
        since training requires the full dataset pipeline.
        """
        from src.utils.config import Paths
        model_path = Paths.models_dir / "whas500_xgboost.pkl"
        if not model_path.exists():
            pytest.skip("No trained model found — run predict train first")

        from src.utils.config import WHAS500Config
        rng = np.random.RandomState(0)
        n   = 10
        data = {c: rng.randn(n) for c in WHAS500Config.numeric_features}
        data.update({c: rng.randint(0, 2, n) for c in WHAS500Config.categorical_features})
        df = pd.DataFrame(data)

        input_path  = tmp_path / "patients.csv"
        output_path = tmp_path / "predictions.csv"
        df.to_csv(input_path, index=False)

        result = runner.invoke(app, [
            "run",
            "--input",  str(input_path),
            "--output", str(output_path),
        ])
        assert result.exit_code == 0
        assert output_path.exists()
        out = pd.read_csv(output_path)
        assert "mortality_risk" in out.columns
        assert "risk_level" in out.columns
