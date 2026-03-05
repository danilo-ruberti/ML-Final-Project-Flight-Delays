"""CatBoost regressor."""
from catboost import CatBoostRegressor

NAME = "CatBoost"
RANDOM_STATE = 42


def build() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
