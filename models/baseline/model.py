"""Baseline: always predict the training-set mean."""
from sklearn.dummy import DummyRegressor

NAME = "Baseline (mean)"


def build() -> DummyRegressor:
    return DummyRegressor(strategy="mean")
