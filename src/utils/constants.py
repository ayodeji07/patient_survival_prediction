"""
src/utils/constants.py
────────────────────────────────────────────────────────────────
Clinical feature metadata, display labels, and reference ranges.

Separating display names and clinical context from the pipeline
code means the models stay data-agnostic while the reports and
visualisations remain clinically interpretable.

Every feature has:
  - A human-readable display name for plots and reports
  - A brief clinical description
  - Normal range (where applicable) for context in SHAP plots

These are used by the visualisation module and the Quarto report.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations


# ── WHAS500 feature metadata ──────────────────────────────────────

WHAS500_FEATURE_META: dict[str, dict] = {
    "age": {
        "display":     "Age (years)",
        "description": "Patient age at hospital admission",
        "unit":        "years",
        "normal":      "18–100",
    },
    "bmi": {
        "display":     "BMI (kg/m²)",
        "description": "Body mass index at admission",
        "unit":        "kg/m²",
        "normal":      "18.5–24.9",
    },
    "sysbp": {
        "display":     "Systolic BP (mmHg)",
        "description": "Initial systolic blood pressure",
        "unit":        "mmHg",
        "normal":      "90–120",
    },
    "diasbp": {
        "display":     "Diastolic BP (mmHg)",
        "description": "Initial diastolic blood pressure",
        "unit":        "mmHg",
        "normal":      "60–80",
    },
    "hr": {
        "display":     "Heart Rate (bpm)",
        "description": "Initial heart rate at admission",
        "unit":        "bpm",
        "normal":      "60–100",
    },
    "glucose": {
        "display":     "Serum Glucose (mg/dL)",
        "description": "Initial serum glucose level",
        "unit":        "mg/dL",
        "normal":      "70–100",
    },
    "los": {
        "display":     "Length of Stay (days)",
        "description": "Duration of the index hospitalisation",
        "unit":        "days",
        "normal":      "1–14",
    },
    "sex": {
        "display":     "Sex",
        "description": "Patient sex (0=Male, 1=Female)",
        "unit":        "binary",
        "normal":      None,
    },
    "chf": {
        "display":     "Heart Failure",
        "description": "Congestive heart failure complication (0=No, 1=Yes)",
        "unit":        "binary",
        "normal":      None,
    },
    "miord": {
        "display":     "MI Order",
        "description": "Myocardial infarction order (1=First, 2=Recurrent)",
        "unit":        "categorical",
        "normal":      None,
    },
    "mitype": {
        "display":     "MI Type",
        "description": "MI type (1=Q-wave, 2=Non-Q-wave, 3=Indeterminate)",
        "unit":        "categorical",
        "normal":      None,
    },
    "cvd": {
        "display":     "CVD History",
        "description": "History of cardiovascular disease (0=No, 1=Yes)",
        "unit":        "binary",
        "normal":      None,
    },
    "afb": {
        "display":     "Atrial Fibrillation",
        "description": "Atrial fibrillation complication (0=No, 1=Yes)",
        "unit":        "binary",
        "normal":      None,
    },
    "sho": {
        "display":     "Cardiogenic Shock",
        "description": "Cardiogenic shock complication (0=No, 1=Yes)",
        "unit":        "binary",
        "normal":      None,
    },
    "av3": {
        "display":     "AV Block (3rd degree)",
        "description": "Third-degree atrioventricular block (0=No, 1=Yes)",
        "unit":        "binary",
        "normal":      None,
    },
}

# ── UCI Heart Disease feature metadata ───────────────────────────

UCI_FEATURE_META: dict[str, dict] = {
    "age": {
        "display":     "Age (years)",
        "description": "Patient age",
        "unit":        "years",
        "normal":      "18–100",
    },
    "sex": {
        "display":     "Sex",
        "description": "1=Male, 0=Female",
        "unit":        "binary",
        "normal":      None,
    },
    "cp": {
        "display":     "Chest Pain Type",
        "description": "1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic",
        "unit":        "categorical",
        "normal":      None,
    },
    "trestbps": {
        "display":     "Resting BP (mmHg)",
        "description": "Resting blood pressure at admission",
        "unit":        "mmHg",
        "normal":      "90–120",
    },
    "chol": {
        "display":     "Cholesterol (mg/dL)",
        "description": "Serum cholesterol",
        "unit":        "mg/dL",
        "normal":      "<200",
    },
    "fbs": {
        "display":     "Fasting Blood Sugar >120",
        "description": "Fasting blood sugar > 120 mg/dL (1=True)",
        "unit":        "binary",
        "normal":      None,
    },
    "restecg": {
        "display":     "Resting ECG",
        "description": "0=Normal, 1=ST-T abnormality, 2=LV hypertrophy",
        "unit":        "categorical",
        "normal":      None,
    },
    "thalach": {
        "display":     "Max Heart Rate",
        "description": "Maximum heart rate achieved",
        "unit":        "bpm",
        "normal":      "60–202",
    },
    "exang": {
        "display":     "Exercise Angina",
        "description": "Exercise-induced angina (1=Yes, 0=No)",
        "unit":        "binary",
        "normal":      None,
    },
    "oldpeak": {
        "display":     "ST Depression",
        "description": "ST depression induced by exercise relative to rest",
        "unit":        "mm",
        "normal":      "0–6",
    },
    "slope": {
        "display":     "ST Slope",
        "description": "Slope of peak exercise ST segment (1=Up, 2=Flat, 3=Down)",
        "unit":        "categorical",
        "normal":      None,
    },
    "ca": {
        "display":     "Vessels Coloured",
        "description": "Number of major vessels coloured by fluoroscopy (0–3)",
        "unit":        "count",
        "normal":      None,
    },
    "thal": {
        "display":     "Thalassemia",
        "description": "3=Normal, 6=Fixed defect, 7=Reversible defect",
        "unit":        "categorical",
        "normal":      None,
    },
}

# ── Model display names ───────────────────────────────────────────

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
}

# Colours for consistent model identification across all plots
MODEL_COLOURS: dict[str, str] = {
    "logistic_regression": "#3498db",   # blue
    "random_forest":       "#2ecc71",   # green
    "xgboost":             "#e74c3c",   # red
    "lightgbm":            "#f39c12",   # orange
}

# ── Severity levels for SHAP waterfall annotations ───────────────

RISK_LEVELS: dict[str, str] = {
    "low":      "Low Risk",
    "moderate": "Moderate Risk",
    "high":     "High Risk",
}

RISK_COLOURS: dict[str, str] = {
    "low":      "#2ecc71",
    "moderate": "#f39c12",
    "high":     "#e74c3c",
}
