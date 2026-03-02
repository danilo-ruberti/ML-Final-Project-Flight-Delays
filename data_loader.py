"""
Shared data loading and feature engineering used by both eda.ipynb and main.py.

Data is downloaded on first use via kagglehub and cached automatically at
~/.cache/kagglehub/datasets/sriharshaeedala/airline-delay/
"""

import glob
import os

import numpy as np
import pandas as pd
import kagglehub

KAGGLE_DATASET = "sriharshaeedala/airline-delay"

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
    """Drop zero-flight rows, fill NaNs, add delay_rate target."""
    df = df[df["arr_flights"] > 0].copy()
    df[DELAY_COLS] = df[DELAY_COLS].fillna(0)
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"]
    return df


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
    """Convenience: load + preprocess + feature engineer in one call."""
    return build_features(preprocess(load_data()))