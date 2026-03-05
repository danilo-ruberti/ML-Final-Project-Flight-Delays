"""Scikit-learn Gradient Boosting regressor."""
from sklearn.ensemble import GradientBoostingRegressor

NAME = "Gradient Boosting"
RANDOM_STATE = 42


def build() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )
