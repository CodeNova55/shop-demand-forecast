import numpy as np
import pandas as pd

from demandforecast.data import make_demand_data


def test_length_and_index():
    y = make_demand_data(n_days=100)
    assert len(y) == 100
    assert isinstance(y.index, pd.DatetimeIndex)


def test_non_negative():
    assert (make_demand_data() >= 0).all()


def test_reproducible():
    a = make_demand_data(seed=1)
    b = make_demand_data(seed=1)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
