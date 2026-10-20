from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Forecaster(ABC):
    """Common interface: fit(y) returns self, predict(horizon) returns an array."""

    name = "base"

    @abstractmethod
    def fit(self, y):
        ...

    @abstractmethod
    def predict(self, horizon):
        ...


class NaiveForecaster(Forecaster):
    """Tomorrow looks like today."""

    name = "naive"

    def fit(self, y):
        self.last_ = float(y.iloc[-1])
        return self

    def predict(self, horizon):
        return np.full(horizon, self.last_)


class SeasonalNaiveForecaster(Forecaster):
    """Each day looks like the same weekday last week."""

    name = "seasonal_naive"

    def __init__(self, season=7):
        self.season = season

    def fit(self, y):
        self.last_season_ = y.to_numpy(dtype=float)[-self.season:]
        return self

    def predict(self, horizon):
        return np.resize(self.last_season_, horizon)


class SeasonalAverageForecaster(Forecaster):
    """Each day looks like the average of that weekday over recent weeks."""

    name = "seasonal_avg"

    def __init__(self, season=7, n_seasons=8):
        self.season = season
        self.n_seasons = n_seasons

    def fit(self, y):
        values = y.to_numpy(dtype=float)
        k = min(self.n_seasons, len(values) // self.season)
        tail = values[-k * self.season:]
        self.profile_ = tail.reshape(k, self.season).mean(axis=0)
        return self

    def predict(self, horizon):
        return np.resize(self.profile_, horizon)


class RegressionForecaster(Forecaster):
    """Ridge regression on calendar features (log scale)."""

    name = "ridge_calendar"

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def _features(index, t):
        dow = np.eye(7)[index.dayofweek.to_numpy()]
        angle = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
        fourier = np.column_stack(
            [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)]
        )
        return np.column_stack([dow, fourier, t / 365.0])

    def fit(self, y):
        t = np.arange(len(y))
        X = self._features(y.index, t)
        target = np.log1p(y.clip(lower=0).to_numpy(dtype=float))
        self.model_ = Ridge(alpha=self.alpha).fit(X, target)
        self.last_date_ = y.index[-1]
        self.n_ = len(y)
        return self

    def predict(self, horizon):
        future = pd.date_range(
            self.last_date_ + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        t = self.n_ + np.arange(horizon)
        pred = np.expm1(self.model_.predict(self._features(future, t)))
        return np.clip(pred, 0, None)


def default_models():
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        SeasonalAverageForecaster(),
        RegressionForecaster(),
    ]
