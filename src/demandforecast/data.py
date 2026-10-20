import numpy as np
import pandas as pd


def make_demand_data(n_days=730, start="2023-01-01", seed=42):
    """Generate synthetic daily unit sales for a shop.

    Demand = linear trend * weekly pattern * yearly seasonality + noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 100 + 0.05 * t
    weekly = np.array([0.90, 0.85, 0.90, 0.95, 1.10, 1.35, 1.20])  # Mon..Sun
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    noise = rng.normal(0, 6, n_days)

    demand = trend * weekly[dow] * yearly + noise
    demand = np.clip(np.round(demand), 0, None)
    return pd.Series(demand, index=dates, name="units_sold")
