"""Ridge regression."""
from sklearn.linear_model import Ridge

NAME = "Ridge Regression"


def build() -> Ridge:
    return Ridge(alpha=1.0)
