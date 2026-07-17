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
    """Load the WHAS500 dataset, downloading it if necessary.

    WHAS500 contains 500 acute MI patients with real survival
    time data.  Key columns:
      lenfol — follow-up time in days (time to death or censoring)
      fstat  — vital status (1=died, 0=censored/alive at last contact)
      age, bmi, sysbp, hr, glucose, ... — clinical features

    Args:
        force_download: Re-download even if the file exists locally.

    Returns:
        Raw DataFrame with all original WHAS500 columns.

    Raises:
        FileNotFoundError: If download fails and no local copy exists.

    Example::

        df = load_whas500()
        print(df.shape)              # (500, 22)
        print(df["fstat"].mean())    # ~0.42 (42% mortality)
    """
    dest = Paths.whas500_csv

    if force_download and dest.exists():
        dest.unlink()

    _download(WHAS500Config.download_url, dest, "WHAS500")

    logger.info("Loading WHAS500 from %s", dest)
    df = pd.read_csv(dest)

    # Standardise column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    logger.info(
        "WHAS500 loaded: %d patients, %d columns, "
        "%.1f%% died during follow-up",
        len(df),
        len(df.columns),
        df["fstat"].mean() * 100 if "fstat" in df.columns else 0,
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
