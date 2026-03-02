"""
ML Final Project — Airline Flight Delay Prediction
====================================================
Dataset : https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
Target  : delay_rate = arr_del15 / arr_flights  (regression)
Split   : time-based  —  train 2013-2020 / test 2021-2023
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join("data", "Airline_Delay_Cause.csv")
TRAIN_YEARS = list(range(2013, 2021))   # 2013–2020
TEST_YEARS  = list(range(2021, 2024))   # 2021–2023
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with no flights
    df = df[df["arr_flights"] > 0].copy()

    # Fill numeric NaNs in delay columns with 0
    delay_cols = [
        "arr_del15", "arr_cancelled", "arr_diverted",
        "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
        "arr_delay", "carrier_delay", "weather_delay",
        "nas_delay", "security_delay", "late_aircraft_delay",
    ]
    df[delay_cols] = df[delay_cols].fillna(0)

    # Target
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"]

    return df


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flights = out["arr_flights"]
    total_delay = out["arr_delay"].replace(0, np.nan)

    # Normalised counts
    out["cancel_rate"]        = out["arr_cancelled"] / flights
    out["divert_rate"]        = out["arr_diverted"]  / flights

    # Delay cause shares
    out["carrier_share"]       = out["carrier_delay"]       / total_delay
    out["weather_share"]       = out["weather_delay"]        / total_delay
    out["nas_share"]           = out["nas_delay"]            / total_delay
    out["late_aircraft_share"] = out["late_aircraft_delay"]  / total_delay

    # Cyclical month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Peak season flag
    out["is_peak_season"] = out["month"].isin([6, 7, 8, 12]).astype(int)

    # Encode carrier / airport as integer labels
    for col in ("carrier", "airport"):
        le = LabelEncoder()
        out[f"{col}_enc"] = le.fit_transform(out[col].astype(str))

    return out


# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    train = df[df["year"].isin(TRAIN_YEARS)]
    test  = df[df["year"].isin(TEST_YEARS)]
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# 5. Model training & evaluation
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "year", "month_sin", "month_cos", "is_peak_season",
    "arr_flights", "cancel_rate", "divert_rate",
    "carrier_share", "weather_share", "nas_share", "late_aircraft_share",
    "carrier_enc", "airport_enc",
]

TARGET_COL = "delay_rate"


def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 1)

    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"  {name:<30}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "fitted": model}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load & process
    df = load_data(DATA_PATH)
    df = preprocess(df)
    df = build_features(df)

    # Split
    train, test = split(df)
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test[TARGET_COL]

    # Models
    models = {
        "Baseline (mean)":       DummyRegressor(strategy="mean"),
        "Ridge Regression":      Ridge(alpha=1.0),
        "Random Forest":         RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    print("\n--- Model Results (test set 2021-2023) ---")
    results = []
    for name, model in models.items():
        res = evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    # Summary table
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "fitted"} for r in results])
    print("\n" + summary.to_string(index=False))

    # Feature importances for the best tree model
    best = next(r for r in results if r["model"] == "Random Forest")["fitted"]
    imp = pd.Series(best.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n--- Random Forest Feature Importances ---")
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()