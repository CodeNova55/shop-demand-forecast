import copy

import numpy as np
import pandas as pd

from demandforecast.data import make_demand_data
from demandforecast.models import default_models


def compute_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / np.sum(np.abs(actual))),
    }


def rolling_backtest(y, models, initial=365, horizon=14, step=14):
    """Rolling-origin backtest: train on the past, forecast the next `horizon` days."""
    rows = []
    for fold, start in enumerate(range(initial, len(y) - horizon + 1, step)):
        train = y.iloc[:start]
        test = y.iloc[start:start + horizon]
        for model in models:
            fitted = copy.deepcopy(model).fit(train)
            pred = fitted.predict(horizon)
            rows.append({"fold": fold, "model": fitted.name,
                         **compute_metrics(test.to_numpy(), pred)})
    return pd.DataFrame(rows)


def summarize(results):
    """Average each metric across folds, best model (lowest MAE) first."""
    return (results.groupby("model")[["mae", "rmse", "wape"]]
            .mean()
            .sort_values("mae"))


if __name__ == "__main__":
    y = make_demand_data()
    results = rolling_backtest(y, default_models())
    print(f"Folds: {results['fold'].nunique()}")
    print(summarize(results).round(3))
