"""LightGBM regressor."""
from lightgbm import LGBMRegressor

NAME = "LightGBM"
RANDOM_STATE = 42


def build() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
