import pandas as pd
import pytest

from src.predict import explain_history_row, predict_history
from tests.conftest import make_firm_rows


def test_multi_year_history_computes_nonzero_two_year_slope():
    """predict_one can carry at most one prior year, so ebitda_slope2_ta is
    always masked to 0 through that path. A 3+ year history run through the
    same features.build_trends groupby that training uses should populate it.
    """
    rows = make_firm_rows(
        "growing_firm", [2010, 2011, 2012], ebitda=[50.0, 80.0, 140.0]
    )
    history = pd.DataFrame(rows)

    results = predict_history(history, latest_only=True)
    assert len(results) == 1
    assert 0.0 <= results.iloc[0]["probability"] <= 1.0
    assert results.iloc[0]["company_name"] == "growing_firm"
    assert results.iloc[0]["year"] == 2012

    explanation = explain_history_row(history)
    assert explanation["feature_values"]["ebitda_slope2_ta"] != 0.0


def test_predict_history_batches_multiple_companies():
    rows = make_firm_rows("firm_a", [2010, 2011])
    rows += make_firm_rows("firm_b", [2010, 2011])
    history = pd.DataFrame(rows)

    latest = predict_history(history, latest_only=True)
    assert sorted(latest["company_name"]) == ["firm_a", "firm_b"]
    assert (latest["probability"].between(0.0, 1.0)).all()

    full = predict_history(history, latest_only=False)
    assert len(full) == 4


def test_predict_history_accepts_x_column_names():
    rows = make_firm_rows("firm_a", [2010, 2011])
    history = pd.DataFrame(rows).rename(columns={
        "current_assets": "X1", "cogs": "X2", "dep_amort": "X3", "ebitda": "X4",
        "inventory": "X5", "net_income": "X6", "receivables": "X7", "market_value": "X8",
        "net_sales": "X9", "total_assets": "X10", "lt_debt": "X11", "ebit": "X12",
        "gross_profit": "X13", "curr_liabilities": "X14", "retained_earnings": "X15",
        "total_revenue": "X16", "total_liabilities": "X17", "total_opex": "X18",
    })

    results = predict_history(history, latest_only=True)
    assert len(results) == 1


def test_predict_history_rejects_duplicate_company_year_rows():
    rows = make_firm_rows("firm_a", [2010, 2010])
    history = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="one row per year"):
        predict_history(history)


def test_predict_history_rejects_missing_columns():
    history = pd.DataFrame([{"company_name": "firm_a", "year": 2010, "total_assets": 100.0}])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_history(history)
