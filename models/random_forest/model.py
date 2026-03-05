"""Random Forest regressor."""
from sklearn.ensemble import RandomForestRegressor

NAME = "Random Forest"
RANDOM_STATE = 42


def build() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
