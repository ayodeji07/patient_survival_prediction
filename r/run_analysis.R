# r/run_analysis.R
# ─────────────────────────────────────────────────────────────────
# Main entry point for the R survival analysis pipeline.
#
# Run from the project root:
#   Rscript r/run_analysis.R
#
# What this script does:
#   1. Loads required R packages (installs if missing)
#   2. Reads the WHAS500 data exported by Python
#   3. Runs Kaplan-Meier analysis (overall + stratified)
#   4. Runs log-rank tests
#   5. Fits the Cox proportional hazards model
#   6. Checks the PH assumption (Schoenfeld residuals)
#   7. Saves all results as JSON and CSV
#   8. Generates and saves all plots as PNGs
#
# Outputs written to data/results/:
#   survival_results.json   ← primary results consumed by Python
#   cox_coefficients.csv    ← Cox model coefficients table
#   km_overall.png
#   km_by_age.png
#   km_by_sex.png
#   cox_forest.png
#   ph_assumption.png
# ─────────────────────────────────────────────────────────────────

# Determine the script directory for relative path resolution
SCRIPT_DIR <- dirname(sys.frame(1)$ofile)
if (is.null(SCRIPT_DIR) || SCRIPT_DIR == "") {
  SCRIPT_DIR <- getwd()
}

# Load helper functions and analysis functions
source(file.path(SCRIPT_DIR, "utils.R"))
source(file.path(SCRIPT_DIR, "survival.R"))
source(file.path(SCRIPT_DIR, "plots.R"))


# ── Setup ─────────────────────────────────────────────────────────

log_info("=" , script = "run_analysis.R")
log_info("  PATIENT SURVIVAL ANALYSIS — R PIPELINE", script = "run_analysis.R")
log_info("  Dataset: WHAS500 (Worcester Heart Attack Study)",
         script = "run_analysis.R")
log_info("=" , script = "run_analysis.R")

load_required_packages()


# ── Load data ─────────────────────────────────────────────────────

data_path <- project_path("data", "processed", "whas500_for_r.csv")

if (!file.exists(data_path)) {
  log_error(
    "WHAS500 data not found at %s.\n  Run the Python pipeline first:\n  python -m src.data.pipeline",
    data_path
  )
  stop("Data file missing — run Python pipeline first")
}

log_info("Loading WHAS500 from %s", data_path)
df <- read.csv(data_path, stringsAsFactors = FALSE)
log_info("Loaded: %d patients, %d columns", nrow(df), ncol(df))


# ── Data preparation ──────────────────────────────────────────────

# Add age group for stratified analysis
df$age_group <- cut(
  df$age,
  breaks = c(0, 60, 75, Inf),
  labels = c("<60", "60-75", ">75"),
  right  = FALSE
)

# Recode sex for display (0=Male, 1=Female in WHAS500)
df$sex_label <- ifelse(df$sex == 0, "Male", "Female")

log_info(
  "Age group distribution: %s",
  paste(table(df$age_group), collapse = ", ")
)
log_info(
  "Sex distribution: %d male (%.1f%%), %d female (%.1f%%)",
  sum(df$sex == 0), mean(df$sex == 0) * 100,
  sum(df$sex == 1), mean(df$sex == 1) * 100
)


# ── Kaplan-Meier analysis ─────────────────────────────────────────

log_info("─── Kaplan-Meier Analysis ─────────────────────────")

km_overall_result  <- fit_km_overall(df)
km_age_result      <- fit_km_stratified(df, "age_group")
km_sex_result      <- fit_km_stratified(df, "sex_label")
km_chf_result      <- fit_km_stratified(df, "chf")

log_info(
  "Overall: median survival = %d days",
  km_overall_result$median_survival_days
)


# ── Cox proportional hazards model ───────────────────────────────

log_info("─── Cox Proportional Hazards Model ────────────────")

# Feature selection: clinically relevant predictors
# Excludes length-of-stay (los) from the main model because it
# is partially a consequence of severity, not a pure predictor
cox_features <- c(
  "age", "bmi", "sysbp", "hr", "glucose",
  "sex", "chf", "cvd", "afb", "sho", "av3", "miord"
)

# Keep only features that exist in this dataset
cox_features <- cox_features[cox_features %in% names(df)]

cox_result <- fit_cox_model(df, cox_features)

# Save Cox coefficients as CSV for Python
cox_coef_path <- project_path("data", "results", "cox_coefficients.csv")
write.csv(
  cox_result$tidy_results,
  cox_coef_path,
  row.names = FALSE
)
log_info("Cox coefficients saved to %s", cox_coef_path)


# ── Package and export results ────────────────────────────────────

log_info("─── Exporting Results ──────────────────────────────")

results <- package_results(
  km_overall = km_overall_result,
  km_age     = km_age_result,
  km_sex     = km_sex_result,
  cox_result = cox_result
)

export_json(results, "survival_results.json")


# ── Generate plots ────────────────────────────────────────────────

log_info("─── Generating Plots ───────────────────────────────")

generate_all_plots(
  km_overall_fit = km_overall_result$km_fit,
  km_age_fit     = km_age_result$km_fit,
  km_sex_fit     = km_sex_result$km_fit,
  cox_result     = cox_result,
  df             = df
)


# ── Summary ───────────────────────────────────────────────────────

log_info("=" , script = "run_analysis.R")
log_info("  R ANALYSIS COMPLETE", script = "run_analysis.R")
log_info(
  "  Concordance (C-index): %.3f (SE = %.3f)",
  cox_result$concordance,
  cox_result$concordance_se
)
log_info(
  "  Median survival: %d days",
  km_overall_result$median_survival_days
)
log_info(
  "  Results written to: data/results/",
  script = "run_analysis.R"
)
log_info("=" , script = "run_analysis.R")
