# r/survival.R
# ─────────────────────────────────────────────────────────────────
# Core survival analysis functions for the WHAS500 dataset.
#
# Three analyses in this file:
#
# 1. Kaplan-Meier survival estimation
#    Non-parametric estimator of the survival function S(t).
#    No model assumptions — shows the empirical probability of
#    surviving past time t.  Stratified by age group and sex.
#
# 2. Log-rank test
#    Tests whether survival curves differ significantly between
#    groups.  Null hypothesis: no difference in survival between
#    the groups being compared.
#
# 3. Cox proportional hazards model
#    Semi-parametric model that estimates the effect of each
#    clinical feature on the hazard (instantaneous risk) of death.
#    The hazard ratio (HR) is the key output:
#      HR > 1 → feature increases risk of death
#      HR < 1 → feature is protective
#      HR = 1 → no effect
#
# Results are exported as JSON and CSV for Python to read.
# Plots are saved as PNGs for the Quarto report.
# ─────────────────────────────────────────────────────────────────

source(file.path(dirname(sys.frame(1)$ofile), "utils.R"))


# ── Kaplan-Meier analysis ─────────────────────────────────────────

#' Fit Kaplan-Meier survival curves for the overall cohort.
#'
#' Returns the survival object and summary statistics including
#' median survival time with 95% confidence interval.
#'
#' @param df Data frame with lenfol (time) and fstat (event) columns.
#' @return List with km_fit (survfit object) and summary statistics.
#'
#' @examples
#' result <- fit_km_overall(whas_df)
#' print(result$median_survival)
fit_km_overall <- function(df) {
  log_info("Fitting Kaplan-Meier: overall cohort (n=%d)", nrow(df))

  km_fit <- survfit(
    Surv(lenfol, fstat) ~ 1,
    data = df
  )

  # Extract median survival and 95% CI
  km_summary <- summary(km_fit)$table
  median_survival <- km_fit$time[which.min(abs(km_fit$surv - 0.5))]

  result <- list(
    n                 = nrow(df),
    n_events          = sum(df$fstat),
    event_rate_pct    = round(mean(df$fstat) * 100, 2),
    median_survival_days = as.numeric(km_fit$time[km_fit$surv <= 0.5][1]),
    # 1-year and 5-year survival probabilities
    survival_1yr      = round(summary(km_fit, times = 365)$surv,  4),
    survival_5yr      = round(summary(km_fit, times = 1825)$surv, 4),
    km_fit            = km_fit
  )

  log_info(
    "KM overall: median survival = %d days, 1yr = %.1f%%, 5yr = %.1f%%",
    result$median_survival_days,
    result$survival_1yr * 100,
    result$survival_5yr * 100
  )

  return(result)
}


#' Fit stratified Kaplan-Meier curves and run the log-rank test.
#'
#' Stratification variable can be age_group, sex, chf, etc.
#' The log-rank test p-value tells us whether survival differs
#' significantly between strata.
#'
#' @param df         Data frame.
#' @param strata_var Character: column name to stratify by.
#' @return List with km_fit, log_rank_test, and strata_summary.
#'
#' @examples
#' result <- fit_km_stratified(df, "age_group")
#' result <- fit_km_stratified(df, "sex")
fit_km_stratified <- function(df, strata_var) {
  log_info(
    "Fitting stratified KM: strata = %s (%d levels)",
    strata_var,
    length(unique(df[[strata_var]]))
  )

  # Build the formula dynamically
  formula <- as.formula(
    sprintf("Surv(lenfol, fstat) ~ %s", strata_var)
  )

  km_fit <- survfit(formula, data = df)

  # Log-rank test for difference between strata
  log_rank <- survdiff(formula, data = df)

  # p-value from chi-squared distribution
  p_value <- 1 - pchisq(log_rank$chisq, df = length(log_rank$n) - 1)

  result <- list(
    strata_var     = strata_var,
    km_fit         = km_fit,
    log_rank_chisq = round(log_rank$chisq, 4),
    log_rank_df    = length(log_rank$n) - 1,
    log_rank_pval  = round(p_value, 6),
    significant    = p_value < 0.05
  )

  log_info(
    "Log-rank test (%s): chi2=%.3f, p=%.4f — %s",
    strata_var,
    result$log_rank_chisq,
    result$log_rank_pval,
    ifelse(result$significant, "SIGNIFICANT", "not significant")
  )

  return(result)
}


# ── Cox proportional hazards model ───────────────────────────────

#' Fit a Cox proportional hazards model on WHAS500 features.
#'
#' The Cox model estimates the effect of each clinical feature
#' on the hazard of death, adjusting for all other features.
#'
#' Proportional hazards assumption: each feature's effect on
#' hazard is constant over time.  We test this with Schoenfeld
#' residuals after fitting.
#'
#' @param df           Data frame with all clinical features.
#' @param feature_cols Character vector of feature column names.
#' @return List with cox_fit, tidy_results, and PH test results.
#'
#' @examples
#' features <- c("age", "bmi", "sysbp", "hr", "glucose",
#'               "sex", "chf", "cvd", "afb")
#' result <- fit_cox_model(df, features)
fit_cox_model <- function(df, feature_cols) {
  log_info(
    "Fitting Cox PH model: %d patients, %d features",
    nrow(df), length(feature_cols)
  )

  # Build formula from feature list
  rhs     <- paste(feature_cols, collapse = " + ")
  formula <- as.formula(
    sprintf("Surv(lenfol, fstat) ~ %s", rhs)
  )

  cox_fit <- coxph(formula, data = df, ties = "breslow")

  # Tidy the output using broom
  # tidy() returns: term, estimate (log HR), std.error, statistic, p.value
  tidy_results <- broom::tidy(cox_fit, exponentiate = TRUE, conf.int = TRUE)
  tidy_results <- tidy_results %>%
    dplyr::rename(
      feature   = term,
      hazard_ratio = estimate,
      hr_lower  = conf.low,
      hr_upper  = conf.high
    ) %>%
    dplyr::mutate(
      significant = p.value < 0.05,
      direction   = dplyr::case_when(
        hazard_ratio > 1 & p.value < 0.05 ~ "increases_risk",
        hazard_ratio < 1 & p.value < 0.05 ~ "protective",
        TRUE                               ~ "not_significant"
      )
    )

  # Model fit statistics (concordance index = discrimination)
  cox_glance   <- broom::glance(cox_fit)
  concordance  <- summary(cox_fit)$concordance

  # ── Proportional hazards assumption test ──────────────────────
  # Tests whether the Schoenfeld residuals are correlated with time.
  # A significant p-value (< 0.05) for a feature means the PH
  # assumption is violated for that feature.
  ph_test <- tryCatch(
    cox.zph(cox_fit),
    error = function(e) {
      log_warning("PH test failed: %s", e$message)
      NULL
    }
  )

  ph_results <- NULL
  if (!is.null(ph_test)) {
    ph_results <- as.data.frame(ph_test$table) %>%
      tibble::rownames_to_column("feature") %>%
      dplyr::mutate(ph_violated = p < 0.05)
  }

  result <- list(
    cox_fit          = cox_fit,
    tidy_results     = tidy_results,
    concordance      = round(concordance[1], 4),
    concordance_se   = round(concordance[2], 4),
    aic              = round(cox_glance$AIC, 2),
    log_likelihood   = round(cox_glance$logLik, 4),
    n_events         = sum(df$fstat),
    n_total          = nrow(df),
    ph_test          = ph_results
  )

  log_info(
    "Cox model: concordance = %.3f (SE=%.3f), AIC = %.1f",
    result$concordance,
    result$concordance_se,
    result$aic
  )

  # Log significant findings
  sig_features <- tidy_results %>%
    dplyr::filter(significant) %>%
    dplyr::select(feature, hazard_ratio, p.value)

  if (nrow(sig_features) > 0) {
    log_info("Significant predictors (p < 0.05):")
    for (i in seq_len(nrow(sig_features))) {
      log_info(
        "  %-20s HR=%.3f  p=%.4f",
        sig_features$feature[i],
        sig_features$hazard_ratio[i],
        sig_features$p.value[i]
      )
    }
  }

  return(result)
}


# ── Results export ────────────────────────────────────────────────

#' Package all survival analysis results into a JSON-serialisable list.
#'
#' Called by run_analysis.R after all analyses are complete.
#' Python reads this JSON to incorporate R results into the report.
#'
#' @param km_overall   Result from fit_km_overall().
#' @param km_age       Result from fit_km_stratified() by age_group.
#' @param km_sex       Result from fit_km_stratified() by sex.
#' @param cox_result   Result from fit_cox_model().
#' @return Named list suitable for jsonlite::write_json().
package_results <- function(km_overall, km_age, km_sex, cox_result) {

  # Convert tidy Cox results to a plain data frame (no S3 objects)
  cox_df <- as.data.frame(cox_result$tidy_results)
  cox_df <- cox_df[, c("feature", "hazard_ratio", "hr_lower", "hr_upper",
                        "p.value", "significant", "direction")]

  list(
    dataset = "WHAS500",
    n_patients        = km_overall$n,
    n_events          = km_overall$n_events,
    event_rate_pct    = km_overall$event_rate_pct,

    kaplan_meier = list(
      overall = list(
        median_survival_days = km_overall$median_survival_days,
        survival_1yr_pct     = round(km_overall$survival_1yr  * 100, 1),
        survival_5yr_pct     = round(km_overall$survival_5yr  * 100, 1)
      ),
      log_rank_age = list(
        strata        = "age_group",
        chisq         = km_age$log_rank_chisq,
        p_value       = km_age$log_rank_pval,
        significant   = km_age$significant
      ),
      log_rank_sex = list(
        strata        = "sex",
        chisq         = km_sex$log_rank_chisq,
        p_value       = km_sex$log_rank_pval,
        significant   = km_sex$significant
      )
    ),

    cox_model = list(
      concordance    = cox_result$concordance,
      concordance_se = cox_result$concordance_se,
      aic            = cox_result$aic,
      coefficients   = cox_df
    )
  )
}
