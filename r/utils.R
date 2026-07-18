# r/utils.R
# ─────────────────────────────────────────────────────────────────
# Utility functions shared across the R survival analysis scripts.
#
# Covers:
#   - Package installation and loading
#   - Path helpers (resolves paths relative to project root)
#   - JSON result export (R results → Python-readable JSON)
#   - Logging helpers consistent with the Python pipeline
# ─────────────────────────────────────────────────────────────────


# ── Package management ────────────────────────────────────────────

#' Install a package if it is not already installed, then load it.
#'
#' @param pkg Character string: the CRAN package name.
#' @return Invisible NULL. Side effect: package is loaded.
#'
#' @examples
#' require_package("survival")
#' require_package("ggplot2")
require_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("[INSTALL] Installing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  invisible(NULL)
}


#' Load all packages required for the survival analysis.
#'
#' Called once at the top of run_analysis.R.  If any package
#' is missing it will be installed automatically.
load_required_packages <- function() {
  pkgs <- c(
    "survival",    # Kaplan-Meier, Cox PH — the core survival analysis package
    "survminer",   # ggplot2-based survival curve visualisation
    "ggplot2",     # grammar of graphics plotting
    "dplyr",       # data manipulation
    "broom",       # tidy model outputs (tidy, glance, augment)
    "jsonlite",    # export results to JSON for Python
    "scales",      # axis formatting in ggplot2
    "gridExtra",   # arrange multiple ggplot2 panels
    "ggfortify"    # autoplot for survival objects
  )
  lapply(pkgs, require_package)
  invisible(NULL)
}


# ── Path helpers ──────────────────────────────────────────────────

#' Resolve a path relative to the project root.
#'
#' Assumes run_analysis.R is run from the project root directory.
#' Works correctly whether running interactively in RStudio or
#' from the command line via Rscript.
#'
#' @param ... Path components (passed to file.path).
#' @return Absolute path string.
#'
#' @examples
#' project_path("data", "raw", "whas500_for_r.csv")
project_path <- function(...) {
  # commandArgs() reflects the originally-invoked script (r/run_analysis.R)
  # regardless of call depth, so this resolves correctly whether called
  # from run_analysis.R directly or from a function inside survival.R/
  # plots.R. Falls back to the working directory if not run via Rscript
  # (e.g. sourced interactively in RStudio, per the docs above).
  args      <- commandArgs(trailingOnly = FALSE)
  file_flag <- grep("^--file=", args, value = TRUE)
  script_dir <- if (length(file_flag) == 1) {
    dirname(normalizePath(sub("^--file=", "", file_flag)))
  } else {
    getwd()
  }
  # The R scripts live in r/, so the project root is one level up
  root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
  file.path(root, ...)
}


# ── Logging ───────────────────────────────────────────────────────

#' Log a message with a timestamp and level indicator.
#'
#' Mirrors the Python logger format:
#' "2024-01-15 09:32:11 | INFO     | r/survival.R | Message"
#'
#' @param level   Character: "INFO", "WARNING", or "ERROR".
#' @param message Character: the message to log.
#' @param script  Character: the script name for the log line.
log_msg <- function(level, message, script = "r/analysis") {
  ts  <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("%s | %-8s | %-25s | %s\n", ts, level, script, message))
}

# `script` must be pulled out as its own named argument (not left in `...`)
# -- otherwise a call like log_info("=", script = "run_analysis.R") passes
# script= straight into sprintf(msg, ...) instead of on to log_msg(), and
# sprintf warns "one argument not used by format" on every such call.
log_info    <- function(msg, ..., script = "r/analysis") log_msg("INFO",    sprintf(msg, ...), script = script)
log_warning <- function(msg, ..., script = "r/analysis") log_msg("WARNING", sprintf(msg, ...), script = script)
log_error   <- function(msg, ..., script = "r/analysis") log_msg("ERROR",   sprintf(msg, ...), script = script)


# ── JSON export ───────────────────────────────────────────────────

#' Export a named list as a JSON file to data/results/.
#'
#' Python reads this file to incorporate R results into the
#' Quarto report and the combined analysis.
#'
#' @param results Named list of results to serialise.
#' @param filename Output filename (without path).
#' @param pretty   Logical: use pretty-printed JSON (default TRUE).
#' @return Invisible: the output file path.
#'
#' @examples
#' export_json(list(n=500, events=210), "survival_results.json")
export_json <- function(results, filename, pretty = TRUE) {
  out_dir  <- project_path("data", "results")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, filename)

  jsonlite::write_json(results, out_path, pretty = pretty, auto_unbox = TRUE)
  log_info("Results exported to %s", out_path)
  invisible(out_path)
}


#' Save a ggplot to the results directory as a PNG.
#'
#' @param plot     ggplot2 object.
#' @param filename Output filename (without path).
#' @param width    Image width in inches (default 10).
#' @param height   Image height in inches (default 7).
#' @param dpi      Resolution (default 150).
save_plot <- function(plot, filename, width = 10, height = 7, dpi = 150) {
  out_dir  <- project_path("data", "results")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, filename)

  ggplot2::ggsave(
    filename = out_path,
    plot     = plot,
    width    = width,
    height   = height,
    dpi      = dpi,
    bg       = "white"
  )
  log_info("Plot saved to %s", out_path)
  invisible(out_path)
}
