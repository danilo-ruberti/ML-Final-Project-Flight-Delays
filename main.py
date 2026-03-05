"""
ML Final Project — Airline Flight Delay Prediction
====================================================
Dataset : https://www.kaggle.com/datasets/sriharshaeedala/airline-delay
Target  : delay_rate = arr_del15 / arr_flights  (regression)
Split   : time-based, pre-COVID only (data < 2020-03-01)
          train = all full years except the last two
          test  = last two full years before cutoff (typically 2018–2019)

Run:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data.data_loader import get_df, COVID_CUTOFF

# Individual model builders — each exposes NAME (str) and build() -> estimator
from models.baseline.model       import NAME as N_BASE,  build as build_base
from models.ridge.model          import NAME as N_RIDGE, build as build_ridge
from models.random_forest.model  import NAME as N_RF,    build as build_rf
from models.gradient_boosting.model import NAME as N_GB, build as build_gb
from models.xgboost.model        import NAME as N_XGB,   build as build_xgb
from models.lightgbm.model       import NAME as N_LGB,   build as build_lgb
from models.catboost.model       import NAME as N_CAT,   build as build_cat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "year", "month_sin", "month_cos", "is_peak_season",
    "arr_flights", "cancel_rate", "divert_rate",
    "carrier_share", "weather_share", "nas_share", "late_aircraft_share",
    "carrier_enc", "airport_enc",
]
TARGET_COL = "delay_rate"

# Registry: (display_name, factory_fn)  — controls run order
MODEL_REGISTRY = [
    (N_BASE,  build_base),
    (N_RIDGE, build_ridge),
    (N_RF,    build_rf),
    (N_GB,    build_gb),
    (N_XGB,   build_xgb),
    (N_LGB,   build_lgb),
    (N_CAT,   build_cat),
]


# ---------------------------------------------------------------------------
# Encode categoricals — fit on TRAIN only to prevent leakage
# ---------------------------------------------------------------------------
def encode(train: pd.DataFrame, test: pd.DataFrame):
    """
    Fit label encoders on train, apply to both splits.
    Unknown categories in test receive code -1.
    Returns (encoded_train, encoded_test).
    """
    train = train.copy()
    test  = test.copy()
    for col in ("carrier", "airport"):
        cats = train[col].astype("category").cat.categories
        train[f"{col}_enc"] = pd.Categorical(train[col], categories=cats).codes
        test[f"{col}_enc"]  = pd.Categorical(test[col],  categories=cats).codes
    return train, test


# ---------------------------------------------------------------------------
# Split — dynamic, based on full years in filtered data
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    """
    test  = last 2 full years (all 12 months present) before COVID cutoff
    train = all full years before the test window
    Partial years (e.g. Jan–Feb 2020) are excluded from both splits.
    """
    year_counts = df.groupby("year")["month"].nunique()
    full_years  = sorted(year_counts[year_counts == 12].index.tolist())

    if len(full_years) < 3:
        raise ValueError(f"Need at least 3 full years to split; found: {full_years}")

    test_years  = full_years[-2:]
    train_years = full_years[:-2]

    train = df[df["year"].isin(train_years)]
    test  = df[df["year"].isin(test_years)]

    print(f"  Train: {train_years[0]}–{train_years[-1]}  ({len(train):,} rows)")
    print(f"  Test : {test_years[0]}–{test_years[-1]}   ({len(test):,} rows)")
    return train, test, train_years, test_years


# ---------------------------------------------------------------------------
# Evaluate a single model
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
    print(f"COVID cutoff: {COVID_CUTOFF}  (only data before this date is used)\n")

    df = get_df()
    train_raw, test_raw, train_years, test_years = split(df)
    train, test = encode(train_raw, test_raw)

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test[TARGET_COL]

    test_label = f"{test_years[0]}–{test_years[-1]}"
    print(f"\n--- Model Results (test set {test_label}, pre-COVID) ---")
    results = [
        evaluate(name, build(), X_train, y_train, X_test, y_test)
        for name, build in MODEL_REGISTRY
    ]

    # Summary table
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "fitted"} for r in results])
    summary = summary.sort_values("RMSE").reset_index(drop=True)
    print("\n--- Summary (sorted by RMSE) ---")
    print(summary.to_string(index=False))

    # Feature importances for tree-based models
    importance_models = [r for r in results if hasattr(r["fitted"], "feature_importances_")]
    for r in importance_models:
        imp = (
            pd.Series(r["fitted"].feature_importances_, index=FEATURE_COLS)
            .sort_values(ascending=False)
        )
        print(f"\n--- {r['model']} Feature Importances ---")
        print(imp.round(4).to_string())

    return results, X_test, y_test, FEATURE_COLS


if __name__ == "__main__":
    main()
