"""
src/data/features.py
────────────────────────────────────────────────────────────────
Feature definitions, one-hot encoding, and survival data
preparation helpers.

This module is the boundary between the raw/cleaned data and
what the models actually receive.  It handles:

  - One-hot encoding of nominal categorical features
    (e.g. MI type has 3 categories → 2 dummy columns)
  - Ordinal encoding of ordered categorical features
    (e.g. MI order: first=0, recurrent=1)
  - Preparation of survival (time, event) arrays for R and lifelines
  - Feature name tracking so SHAP plots show interpretable labels

The encoding choices are clinically motivated:
  - Binary features (sex, chf, cvd, ...) need no encoding — 0/1 is
    already interpretable.
  - Nominal multi-category features (mitype, cp, thal) are one-hot
    encoded because there is no natural ordering.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import WHAS500Config, UCIHeartConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Columns that need one-hot encoding ───────────────────────────
# Binary columns (already 0/1) and truly ordinal columns are left as-is.
# Only nominal categoricals with 3+ levels are one-hot encoded.
#
# Categories are fixed explicitly (not inferred from whatever values
# happen to appear in a given batch) so that encoding a single new
# patient at inference time produces the exact same dummy columns,
# in the same order, as encoding the full training set.

WHAS500_OHE_CATEGORIES: dict[str, list[int]] = {
    # WHAS500's own categoricals (mitype, miord, sex, chf, ...) are all
    # already binary (0/1) in the real data — nothing to one-hot encode.
    # Kept as an empty dict (rather than removing the mechanism) so a
    # future nominal 3+-level feature can be added the same way UCI's are.
}
UCI_OHE_CATEGORIES: dict[str, list[int]] = {
    "cp":      [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope":   [1, 2, 3],
    "thal":    [3, 6, 7],
}


def encode_whas500(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply one-hot encoding to WHAS500 nominal categorical features.

    All of WHAS500's categoricals (sex, chf, miord, mitype, cvd, afb,
    sho, av3) are already binary (0/1) in the real data, so this is
    currently a passthrough — kept as a function (rather than inlined)
    so it mirrors encode_uci_heart()'s interface and can absorb a
    genuinely nominal feature later without changing call sites.

    Args:
        df: Cleaned WHAS500 DataFrame.

    Returns:
        Tuple of (encoded_df, feature_names).
        feature_names is the ordered list of all feature columns
        in the returned DataFrame — used for SHAP plots.
    """
    df = df.copy()

    for col, categories in WHAS500_OHE_CATEGORIES.items():
        if col not in df.columns:
            continue
        cat_series = pd.Series(
            pd.Categorical(df[col], categories=categories), index=df.index
        )
        dummies = pd.get_dummies(
            cat_series, prefix=col, drop_first=True, dtype=int
        )
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        logger.debug(
            "One-hot encoded '%s': %d new columns", col, len(dummies.columns)
        )

    # Collect all feature columns (exclude target and survival cols)
    exclude = {
        WHAS500Config.time_col,
        WHAS500Config.event_col,
        WHAS500Config.mortality_col,
        "age_group",   # string column for analysis only
        # diasbp is exactly determined by sysbp and pulse_pressure
        # (pulse_pressure = sysbp - diasbp), so keeping all three would
        # give infinite VIF. sysbp + pulse_pressure together already
        # carry both "BP level" and "pulse pressure" signal, so diasbp
        # is dropped here rather than pulse_pressure (which has known
        # independent prognostic value — see engineer_whas500_features).
        "diasbp",
    }
    feature_names = [c for c in df.columns if c not in exclude]

    logger.info(
        "WHAS500 encoded: %d features", len(feature_names)
    )
    return df, feature_names


def encode_uci_heart(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply one-hot encoding to UCI Heart Disease nominal features.

    cp (chest pain type): 4 unordered categories → one-hot
    restecg: 3 categories → one-hot
    slope: 3 categories → one-hot
    thal: 3 categories → one-hot

    Args:
        df: Cleaned UCI Heart Disease DataFrame.

    Returns:
        Tuple of (encoded_df, feature_names).
    """
    df = df.copy()

    for col, categories in UCI_OHE_CATEGORIES.items():
        if col not in df.columns:
            continue
        cat_series = pd.Series(
            pd.Categorical(df[col], categories=categories), index=df.index
        )
        dummies = pd.get_dummies(
            cat_series, prefix=col, drop_first=True, dtype=int
        )
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    exclude = {UCIHeartConfig.target_col}
    feature_names = [c for c in df.columns if c not in exclude]

    logger.info("UCI Heart encoded: %d features", len(feature_names))
    return df, feature_names


def prepare_survival_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract survival time and event arrays for lifelines / R.

    Args:
        df: Cleaned WHAS500 DataFrame with lenfol and fstat columns.

    Returns:
        Tuple of:
          durations — 1D float array of follow-up times in days
          events    — 1D boolean array (True = event/death occurred)

    Example::

        durations, events = prepare_survival_arrays(whas_df)
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(durations, events)
    """
    cfg       = WHAS500Config
    durations = df[cfg.time_col].values.astype(float)
    events    = df[cfg.event_col].values.astype(bool)

    logger.debug(
        "Survival arrays: n=%d, events=%d (%.1f%%)",
        len(durations),
        events.sum(),
        events.mean() * 100,
    )
    return durations, events


def get_feature_display_names(
    feature_names: list[str],
    dataset:       str = "whas500",
) -> dict[str, str]:
    """Map internal column names to human-readable display labels.

    Used by SHAP plots, model cards, and the Quarto report.

    Args:
        feature_names: List of column names from the encoded DataFrame.
        dataset:       "whas500" or "uci".

    Returns:
        Dict mapping column name → display label.
    """
    from src.utils.constants import WHAS500_FEATURE_META, UCI_FEATURE_META

    meta = WHAS500_FEATURE_META if dataset == "whas500" else UCI_FEATURE_META
    result = {}

    for name in feature_names:
        # Check direct match first
        if name in meta:
            result[name] = meta[name]["display"]
        else:
            # Handle one-hot encoded columns: "mitype_2" → "MI Type = 2"
            base = name.rsplit("_", 1)[0]
            suffix = name.rsplit("_", 1)[-1] if "_" in name else ""
            if base in meta:
                result[name] = f"{meta[base]['display']} = {suffix}"
            else:
                # Fallback: prettify the column name
                result[name] = name.replace("_", " ").title()

    return result


def export_for_r(
    df:          pd.DataFrame,
    output_path: "Path | None" = None,
) -> "Path":
    """Export the cleaned WHAS500 DataFrame to CSV for R analysis.

    R reads this file directly — no need for rpy2 or any R-Python
    bridge.  The survival analysis runs independently in R and
    writes its results back to data/results/ as JSON/CSV.

    Args:
        df:          Cleaned WHAS500 DataFrame.
        output_path: Override the default output path.

    Returns:
        Path to the written CSV file.
    """
    from src.utils.config import Paths

    dest = output_path or (Paths.processed / "whas500_for_r.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Include only the columns R needs for survival analysis
    cfg     = WHAS500Config
    r_cols  = (
        [cfg.time_col, cfg.event_col]
        + cfg.numeric_features
        + cfg.categorical_features
    )
    r_cols  = [c for c in r_cols if c in df.columns]
    df[r_cols].to_csv(dest, index=False)

    logger.info(
        "WHAS500 exported for R: %d rows, %d columns → %s",
        len(df), len(r_cols), dest,
    )
    return dest
