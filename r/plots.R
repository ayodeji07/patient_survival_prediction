# r/plots.R
# ─────────────────────────────────────────────────────────────────
# Publication-quality survival analysis plots using ggplot2
# and survminer.
#
# All plots follow a consistent visual style:
#   - Clean white background (theme_bw)
#   - Colour palette accessible to colour-blind readers
#   - 95% confidence intervals shown as shaded ribbons
#   - At-risk tables below KM curves (clinical convention)
#
# Outputs are saved to data/results/ as PNGs and imported by
# the Quarto report.
# ─────────────────────────────────────────────────────────────────

# utils.R is already sourced by run_analysis.R before this file, which
# provides project_path()/save_plot()/log_info() etc. used below.


# ── Colour palette ────────────────────────────────────────────────
# Accessible to deuteranopia (most common colour blindness type).
# Consistent with the Python Plotly palette in plots.py.
PALETTE <- c(
  "#2196F3",   # blue
  "#F44336",   # red
  "#4CAF50",   # green
  "#FF9800",   # orange
  "#9C27B0"    # purple
)


# ── Kaplan-Meier plots ────────────────────────────────────────────

#' Plot the overall Kaplan-Meier survival curve.
#'
#' Shows the empirical survival function for the full WHAS500
#' cohort with 95% confidence interval ribbon.
#'
#' @param km_fit    survfit object from fit_km_overall().
#' @param title     Plot title string.
#' @return ggplot2 object.
plot_km_overall <- function(km_fit, title = "Overall Survival — WHAS500 Cohort") {

  p <- survminer::ggsurvplot(
    km_fit,
    data             = NULL,
    risk.table       = TRUE,           # at-risk table below curve
    risk.table.height = 0.25,
    conf.int         = TRUE,
    conf.int.fill    = PALETTE[1],
    conf.int.alpha   = 0.15,
    palette          = PALETTE[1],
    xlab             = "Follow-up Time (days)",
    ylab             = "Survival Probability",
    title            = title,
    surv.median.line = "hv",           # horizontal + vertical median line
    ggtheme          = theme_bw(base_size = 13),
    risk.table.col   = "strata",
    fontsize         = 4
  )

  return(p)
}


#' Plot stratified Kaplan-Meier curves with log-rank p-value.
#'
#' @param km_fit     survfit object from fit_km_stratified().
#' @param df         Data frame (needed for at-risk table).
#' @param strata_var Column name used for stratification.
#' @param strata_labels Named vector mapping strata codes to labels.
#'   e.g. c("0" = "Male", "1" = "Female")
#' @param title      Plot title.
#' @param pval       Log-rank p-value to annotate on the plot.
#' @return ggplot2 / survminer object.
plot_km_stratified <- function(
  km_fit,
  df,
  strata_var,
  strata_labels = NULL,
  title         = NULL,
  pval          = TRUE
) {

  if (is.null(title)) {
    title <- sprintf("Survival by %s — WHAS500", strata_var)
  }

  n_strata <- length(unique(df[[strata_var]]))
  colours  <- PALETTE[seq_len(n_strata)]

  p <- survminer::ggsurvplot(
    km_fit,
    data              = df,
    risk.table        = TRUE,
    risk.table.height = 0.28,
    conf.int          = TRUE,
    conf.int.alpha    = 0.12,
    palette           = colours,
    pval              = pval,          # annotate log-rank p-value
    pval.size         = 4.5,
    xlab              = "Follow-up Time (days)",
    ylab              = "Survival Probability",
    title             = title,
    legend.title      = strata_var,
    legend.labs       = strata_labels,
    ggtheme           = theme_bw(base_size = 13),
    risk.table.col    = "strata",
    fontsize          = 4
  )

  return(p)
}


# ── Cox model forest plot ─────────────────────────────────────────

#' Create a forest plot of Cox model hazard ratios.
#'
#' A forest plot shows each feature's HR with its 95% CI as a
#' horizontal line + point.  The vertical dashed line at HR=1
#' is the null (no effect) reference.  Features with CI entirely
#' to the right of HR=1 are risk factors; entirely to the left
#' are protective.
#'
#' @param tidy_results Data frame from fit_cox_model()$tidy_results.
#' @param title        Plot title.
#' @return ggplot2 object.
plot_cox_forest <- function(
  tidy_results,
  title = "Cox Model — Hazard Ratios with 95% CI"
) {

  # Sort by hazard ratio for visual clarity
  df_plot <- tidy_results %>%
    dplyr::arrange(hazard_ratio) %>%
    dplyr::mutate(
      feature  = factor(feature, levels = feature),
      colour   = dplyr::case_when(
        significant & hazard_ratio > 1 ~ "Risk factor",
        significant & hazard_ratio < 1 ~ "Protective",
        TRUE                           ~ "Not significant"
      )
    )

  colour_map <- c(
    "Risk factor"     = "#F44336",
    "Protective"      = "#4CAF50",
    "Not significant" = "#9E9E9E"
  )

  p <- ggplot2::ggplot(
    df_plot,
    ggplot2::aes(
      x     = hazard_ratio,
      y     = feature,
      xmin  = hr_lower,
      xmax  = hr_upper,
      colour = colour
    )
  ) +
    ggplot2::geom_vline(
      xintercept = 1, linetype = "dashed",
      colour = "black", linewidth = 0.6, alpha = 0.7
    ) +
    ggplot2::geom_errorbarh(
      height    = 0.3,
      linewidth = 0.8,
      alpha     = 0.8
    ) +
    ggplot2::geom_point(size = 3) +
    ggplot2::scale_colour_manual(
      values = colour_map,
      name   = "Effect"
    ) +
    ggplot2::scale_x_log10(
      breaks = scales::log_breaks(n = 8),
      labels = scales::number_format(accuracy = 0.1)
    ) +
    ggplot2::labs(
      title   = title,
      x       = "Hazard Ratio (log scale)",
      y       = NULL,
      caption = "Bars show 95% confidence intervals. Dashed line at HR = 1 (null effect)."
    ) +
    ggplot2::theme_bw(base_size = 13) +
    ggplot2::theme(
      legend.position = "bottom",
      plot.caption    = ggplot2::element_text(colour = "grey50", size = 9)
    )

  return(p)
}


#' Plot Schoenfeld residuals for the proportional hazards assumption test.
#'
#' One panel per feature.  A flat, horizontal smoothing line
#' suggests the PH assumption holds.  A significant trend means
#' the feature's effect changes over time — the PH assumption
#' is violated.
#'
#' @param ph_test   cox.zph object from cox.zph(cox_fit).
#' @return ggplot2 object (arranged with gridExtra).
plot_ph_assumption <- function(ph_test) {
  p <- survminer::ggcoxzph(
    ph_test,
    ggtheme = theme_bw(base_size = 11),
    font.main = 11
  )
  return(p)
}


# ── Combined output ───────────────────────────────────────────────

#' Generate and save all plots to data/results/.
#'
#' Called by run_analysis.R after all model fitting is complete.
#'
#' @param km_overall_fit   survfit from fit_km_overall().
#' @param km_age_fit       survfit from fit_km_stratified(age_group).
#' @param km_sex_fit       survfit from fit_km_stratified(sex).
#' @param cox_result       Full result from fit_cox_model().
#' @param df               Clean WHAS500 data frame.
generate_all_plots <- function(
  km_overall_fit,
  km_age_fit,
  km_sex_fit,
  cox_result,
  df
) {
  log_info("Generating survival analysis plots...")

  # 1. Overall KM curve
  p_overall <- plot_km_overall(km_overall_fit)
  save_plot(p_overall$plot, "km_overall.png", width = 9, height = 7)

  # 2. KM by age group
  p_age <- plot_km_stratified(
    km_age_fit, df,
    strata_var    = "age_group",
    strata_labels = c("<60", "60-75", ">75"),
    title         = "Survival by Age Group — WHAS500",
    pval          = TRUE
  )
  save_plot(p_age$plot, "km_by_age.png", width = 10, height = 8)

  # 3. KM by sex
  p_sex <- plot_km_stratified(
    km_sex_fit, df,
    strata_var    = "sex",
    strata_labels = c("Male", "Female"),
    title         = "Survival by Sex — WHAS500",
    pval          = TRUE
  )
  save_plot(p_sex$plot, "km_by_sex.png", width = 10, height = 8)

  # 4. Cox forest plot
  p_forest <- plot_cox_forest(cox_result$tidy_results)
  save_plot(p_forest, "cox_forest.png", width = 10, height = 8)

  # 5. PH assumption check
  if (!is.null(cox_result$cox_fit)) {
    ph_test <- tryCatch(cox.zph(cox_result$cox_fit), error = function(e) NULL)
    if (!is.null(ph_test)) {
      p_ph <- plot_ph_assumption(ph_test)
      save_plot(
        gridExtra::arrangeGrob(grobs = p_ph),
        "ph_assumption.png", width = 14, height = 10
      )
    }
  }

  log_info("All plots saved to data/results/")
}
