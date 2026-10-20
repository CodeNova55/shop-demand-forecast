import numpy as np
import pytest

from demandforecast.data import make_demand_data
from demandforecast.models import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    default_models,
)


def test_naive_repeats_last_value():
    y = make_demand_data(n_days=60)
    pred = NaiveForecaster().fit(y).predict(5)
    assert np.all(pred == y.iloc[-1])


def test_seasonal_naive_repeats_last_week():
    y = make_demand_data(n_days=60)
    pred = SeasonalNaiveForecaster().fit(y).predict(14)
    last_week = y.to_numpy()[-7:]
    assert np.array_equal(pred, np.concatenate([last_week, last_week]))


@pytest.mark.parametrize("model", default_models(), ids=lambda m: m.name)
def test_predict_shape_and_finite(model):
    y = make_demand_data(n_days=400)
    pred = model.fit(y).predict(14)
    assert pred.shape == (14,)
    assert np.isfinite(pred).all()
