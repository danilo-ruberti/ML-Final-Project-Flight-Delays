"""
ML Final Project — Airline Flight Delay Prediction
====================================================
Dataset : https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
Target  : delay_rate = arr_del15 / arr_flights  (regression)
Split   : time-based  —  train 2013-2020 / test 2021-2023
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_loader import get_df

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_YEARS  = list(range(2013, 2021))   # 2013–2020
TEST_YEARS   = list(range(2021, 2024))   # 2021–2023
RANDOM_STATE = 42

FEATURE_COLS = [
    "year", "month_sin", "month_cos", "is_peak_season",
    "arr_flights", "cancel_rate", "divert_rate",
    "carrier_share", "weather_share", "nas_share", "late_aircraft_share",
    "carrier_enc", "airport_enc",
]
TARGET_COL = "delay_rate"


# ---------------------------------------------------------------------------
# Encode categoricals (carrier / airport) after the shared feature build
# ---------------------------------------------------------------------------
def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("carrier", "airport"):
        le = LabelEncoder()
        out[f"{col}_enc"] = le.fit_transform(out[col].astype(str))
    return out


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    train = df[df["year"].isin(TRAIN_YEARS)]
    test  = df[df["year"].isin(TEST_YEARS)]
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, 1)

    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"  {name:<30}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "fitted": model}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = encode(get_df())

    train, test = split(df)
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test[TARGET_COL]

    models = {
        "Baseline (mean)":   DummyRegressor(strategy="mean"),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    print("\n--- Model Results (test set 2021-2023) ---")
    results = [evaluate(name, m, X_train, y_train, X_test, y_test) for name, m in models.items()]

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "fitted"} for r in results])
    print("\n" + summary.to_string(index=False))

    best = next(r for r in results if r["model"] == "Random Forest")["fitted"]
    imp = pd.Series(best.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n--- Random Forest Feature Importances ---")
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()