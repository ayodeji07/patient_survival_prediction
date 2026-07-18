"""
src/data/extract.py
────────────────────────────────────────────────────────────────
Data extraction layer — downloads WHAS500 and UCI Heart Disease
datasets from their public URLs and saves them to data/raw/.

Both datasets are freely available and download in seconds.
No credentials or manual steps required.

WHAS500 (Worcester Heart Attack Study)
───────────────────────────────────────
500 acute MI patients from Worcester, Massachusetts (1975–2001).
Real time-to-event data — follow-up in days, vital status at
last contact.  Gold standard for demonstrating survival analysis.

UCI Heart Disease (Cleveland subset)
──────────────────────────────────────
303 patients, 13 clinical features, binary disease target.
Binarised from 4-class (0=no disease, 1–4=disease presence).
Widely used ML benchmark, well-understood feature set.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils.config import Paths, WHAS500Config, UCIHeartConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _download(url: str, dest: Path, name: str) -> Path:
    """Download a file from URL to dest, skipping if already present.

    Args:
        url:  Public download URL.
        dest: Destination file path.
        name: Human-readable dataset name for log messages.

    Returns:
        Path to the downloaded file.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    if dest.exists():
        logger.info("%s already downloaded: %s", name, dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s...", name, url)

    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        logger.info("%s downloaded: %s (%d KB)", name, dest.name, size_kb)
    except Exception as exc:
        logger.error("Failed to download %s: %s", name, exc)
        raise

    return dest


def load_whas500(force_download: bool = False) -> pd.DataFrame:
    """Load the WHAS500 dataset.

    Loaded from scikit-survival's bundled copy (sksurv.datasets.
    load_whas500) — the standard Python distribution of this dataset.
    No network download is involved; this is the real, correctly
    labelled WHAS500 data (500 acute MI patients). Key columns:
      lenfol — follow-up time in days (time to death or censoring)
      fstat  — vital status (1=died, 0=censored/alive at last contact)
      age, bmi, sysbp, hr, ... — clinical features

    A CSV snapshot is cached to data/raw/whas500.csv for inspection,
    but the authoritative load always goes through scikit-survival.

    Args:
        force_download: Re-write the cached CSV snapshot even if one
                        already exists. Loading itself is always from
                        the bundled package data (nothing to "re-download").

    Returns:
        Raw DataFrame with all original WHAS500 columns.

    Example::

        df = load_whas500()
        print(df.shape)              # (500, 16)
        print(df["fstat"].mean())    # ~0.43 (43% mortality)
    """
    from sksurv.datasets import load_whas500 as _load_whas500_bundled

    X, y = _load_whas500_bundled()
    df = X.rename(columns={"gender": "sex"}).copy()

    # sksurv stores binary categoricals as a "category" dtype of '0'/'1'
    # strings — convert to int so the rest of the pipeline (imputation,
    # scaling, models) sees plain numeric columns.
    for col in df.columns:
        if str(df[col].dtype) == "category":
            df[col] = df[col].astype(int)

    df[WHAS500Config.time_col]  = y["lenfol"].astype(float)
    df[WHAS500Config.event_col] = y["fstat"].astype(int)

    # Cache a CSV snapshot for inspection / consistency with data/raw/
    dest = Paths.whas500_csv
    if force_download or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        logger.info("WHAS500 cached to %s", dest)

    logger.info(
        "WHAS500 loaded (bundled via scikit-survival): %d patients, "
        "%d columns, %.1f%% died during follow-up",
        len(df),
        len(df.columns),
        df[WHAS500Config.event_col].mean() * 100,
    )
    return df


def load_uci_heart(force_download: bool = False) -> pd.DataFrame:
    """Load the UCI Heart Disease (Cleveland) dataset.

    The file has no header — column names are assigned from
    UCIHeartConfig.columns.  Missing values are coded as '?' in
    the original file and are converted to NaN.

    Args:
        force_download: Re-download even if the file exists locally.

    Returns:
        Raw DataFrame with named columns and NaN missing values.

    Example::

        df = load_uci_heart()
        print(df.shape)              # (303, 14)
        print(df["target"].mean())   # ~0.54 (54% have disease)
    """
    dest = Paths.uci_csv

    if force_download and dest.exists():
        dest.unlink()

    _download(UCIHeartConfig.download_url, dest, "UCI Heart Disease")

    logger.info("Loading UCI Heart Disease from %s", dest)
    df = pd.read_csv(
        dest,
        header  = None,
        names   = UCIHeartConfig.columns,
        na_values = ["?"],   # UCI uses '?' for missing values
    )

    # Binarise target: 0 = no disease, 1 = disease (original 1-4 → 1)
    df["target"] = (df["target"] > 0).astype(int)

    logger.info(
        "UCI Heart Disease loaded: %d patients, %d features, "
        "%.1f%% with heart disease",
        len(df),
        len(df.columns) - 1,
        df["target"].mean() * 100,
    )
    return df


def load_both_datasets(
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets in one call.

    Convenience wrapper for the pipeline and notebooks.

    Args:
        force_download: Re-download both datasets.

    Returns:
        Tuple of (whas500_df, uci_heart_df).

    Example::

        whas, uci = load_both_datasets()
    """
    Paths.ensure_all()
    whas = load_whas500(force_download=force_download)
    uci  = load_uci_heart(force_download=force_download)
    return whas, uci
