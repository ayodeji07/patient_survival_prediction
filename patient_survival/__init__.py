"""
patient_survival
────────────────────────────────────────────────────────────────
Patient Survival & Outcome Prediction — Python package.

Install:
    pip install -e .

Usage as a library:
    from patient_survival import SurvivalPredictor

Usage as a CLI:
    predict --input patients.csv --output predictions.csv
    predict --input patients.csv --explain --model xgboost
────────────────────────────────────────────────────────────────
"""

from patient_survival.predictor import SurvivalPredictor

__version__ = "1.0.0"
__all__      = ["SurvivalPredictor"]
