# LinkedIn drafts

Two versions: a short caption for the **Featured** section (link preview + 1-2 lines),
and a longer version for an actual **feed post** (gets far more visibility than a
silent Featured-section add, since it shows up in people's feeds).

Suggested image for either: `assets/cox_forest.png` or `assets/km_by_age.png`.

---

## Featured section (short caption)

**Title:** Patient Survival Prediction — R + Python Clinical Data Science Pipeline

**Description:**
End-to-end survival analysis and ML mortality prediction on real acute heart attack
outcomes (WHAS500, n=500). Kaplan-Meier/Cox PH in R, 4-model comparison + SHAP
explainability in Python, AUC 0.88. Full report and code linked below.

---

## Feed post (long version)

I built an end-to-end clinical survival analysis pipeline predicting mortality after
acute heart attacks — and along the way found a result that's a good reminder of why
you check your assumptions twice.

**The setup:** 500 real patients from the Worcester Heart Attack Study, tracked for
actual survival outcomes. Two independent analyses cross-checking each other:
Kaplan-Meier curves + a Cox proportional hazards model in R, and four ML classifiers
(Logistic Regression, Random Forest, XGBoost, LightGBM) compared head-to-head in
Python, with SHAP explainability so every prediction is traceable, not a black box.

**The result that stood out:** women in this cohort had significantly worse raw
survival than men (p = 0.005). Looked like a real effect — until I controlled for age
in the Cox model, at which point sex dropped out as non-significant entirely. The
women in the dataset simply skewed older, and age was doing the real work. Easy to
miss if you only look at one type of analysis.

**The model:** Logistic Regression came out on top (AUC 0.88) — not XGBoost or
LightGBM. On a dataset this size, with the right feature engineering, the simplest
interpretable model can beat the more complex ones. Worth remembering before reaching
for the fanciest tool by default.

Heart failure and cardiogenic shock roughly double mortality risk independent of
everything else measured; age remains the single strongest predictor overall.

Tech: R (survival, survminer) · Python (scikit-learn, XGBoost, LightGBM, SHAP) ·
Quarto for a reproducible combined report · full test suite · Typer CLI.

Full report (viewable directly, no setup needed): [link to report.pdf]
Code: [link to GitHub repo]

#DataScience #MachineLearning #SurvivalAnalysis #Healthcare #Python #RStats

---

## Notes on posting

- Attach one image (the Cox forest plot or KM-by-age chart) directly to the post —
  posts with an image get meaningfully more reach than link-only posts.
- Don't rely on the auto-generated link preview for `report.pdf` — GitHub's preview
  card for PDFs is plain; the attached chart image is what stops the scroll.
- If you want engagement, end with a question or invite discussion (e.g. "Curious
  how others have handled confounding in observational clinical data — anyone hit
  a similar surprise?") rather than just a link dump.
