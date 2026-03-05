"""
Shared data loading and feature engineering.

Data is downloaded on first use via kagglehub and cached automatically at
~/.cache/kagglehub/datasets/sriharshaeedala/airline-delay/
"""

import glob
import os

import numpy as np
import pandas as pd
import kagglehub

KAGGLE_DATASET = "sriharshaeedala/airline-delay"

# All modeling excludes data on or after this date to avoid COVID distortion.
# Monthly data: rows where (year, month) >= (2020, 3) are dropped.
COVID_CUTOFF = "2020-03-01"

DELAY_COLS = [
    "arr_del15", "arr_cancelled", "arr_diverted",
    "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
    "arr_delay", "carrier_delay", "weather_delay",
    "nas_delay", "security_delay", "late_aircraft_delay",
]


def load_data() -> pd.DataFrame:
    """Download (or load from cache) the airline delay dataset and return raw DataFrame."""
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {dataset_path}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {csv_files[0]}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-flight rows, fill NaNs, add delay_rate target and flight_date column."""
    df = df[df["arr_flights"] > 0].copy()
    df[DELAY_COLS] = df[DELAY_COLS].fillna(0)
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"]
    # Construct a proper date from year + month (day=1 sentinel for monthly data).
    df["flight_date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df


def filter_pre_covid(df: pd.DataFrame, date_col: str = "flight_date") -> pd.DataFrame:
    """
    Keep only rows where date_col < COVID_CUTOFF ('2020-03-01').

    Logs row counts and date ranges before and after filtering, and asserts
    that no post-cutoff rows remain.
    """
    cutoff = pd.Timestamp(COVID_CUTOFF)

    # Robustly parse; drop any rows with unparseable dates.
    dates = pd.to_datetime(df[date_col], errors="coerce")
    n_invalid = int(dates.isna().sum())
    if n_invalid:
        print(f"  [filter_pre_covid] Dropping {n_invalid:,} rows with invalid/missing dates.")
        df = df[dates.notna()].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    print(
        f"  [filter_pre_covid] Before: {len(df):,} rows  |  "
        f"date range {df[date_col].min().date()} → {df[date_col].max().date()}"
    )
    print("  Row counts by year (pre-filter):")
    print(df.groupby("year").size().rename("rows").to_string())

    df_filtered = df[df[date_col] < cutoff].copy()

    print(
        f"  [filter_pre_covid] After : {len(df_filtered):,} rows  |  "
        f"date range {df_filtered[date_col].min().date()} → {df_filtered[date_col].max().date()}"
    )
    print("  Row counts by year (post-filter):")
    print(df_filtered.groupby("year").size().rename("rows").to_string())

    assert df_filtered[date_col].max() < cutoff, (
        f"Sanity check failed: max post-filter date {df_filtered[date_col].max().date()} "
        f">= COVID_CUTOFF {COVID_CUTOFF}"
    )

    return df_filtered


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ML-ready features from the preprocessed DataFrame."""
    out = df.copy()
    flights     = out["arr_flights"]
    total_delay = out["arr_delay"].replace(0, np.nan)

    out["cancel_rate"]         = out["arr_cancelled"]       / flights
    out["divert_rate"]         = out["arr_diverted"]         / flights
    out["carrier_share"]       = out["carrier_delay"]        / total_delay
    out["weather_share"]       = out["weather_delay"]        / total_delay
    out["nas_share"]           = out["nas_delay"]            / total_delay
    out["late_aircraft_share"] = out["late_aircraft_delay"]  / total_delay
    out["month_sin"]           = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"]           = np.cos(2 * np.pi * out["month"] / 12)
    out["is_peak_season"]      = out["month"].isin([6, 7, 8, 12]).astype(int)

    return out


def get_df() -> pd.DataFrame:
    """Convenience: load → preprocess → filter pre-COVID → feature engineer."""
    return build_features(filter_pre_covid(preprocess(load_data())))
