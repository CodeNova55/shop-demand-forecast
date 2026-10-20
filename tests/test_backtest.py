from demandforecast.backtest import compute_metrics, rolling_backtest, summarize
from demandforecast.data import make_demand_data
from demandforecast.models import default_models


def test_metrics_perfect_forecast():
    m = compute_metrics([1, 2, 3], [1, 2, 3])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["wape"] == 0


def test_rolling_backtest_covers_all_models():
    y = make_demand_data(n_days=500)
    results = rolling_backtest(y, default_models(), initial=400, horizon=14, step=14)
    assert set(results["model"]) == {m.name for m in default_models()}
    assert results.groupby("model").size().nunique() == 1


def test_regression_beats_naive():
    y = make_demand_data()
    summary = summarize(rolling_backtest(y, default_models()))
    assert summary.loc["ridge_calendar", "mae"] < summary.loc["naive", "mae"]
