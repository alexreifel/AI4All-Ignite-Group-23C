"""Golden predictions: five fixed inputs with probabilities frozen from the
trained artifacts in models/. If a retrain legitimately changes these
numbers, regenerate them deliberately, do not just widen the tolerance.
"""

import pytest

from src.predict import predict_one

GOLDEN_INPUTS = [
    {
        "name": "healthy_large_firm",
        "financials": {
            "current_assets": 500.0, "cogs": 700.0, "dep_amort": 20.0, "ebitda": 150.0,
            "inventory": 200.0, "net_income": 60.0, "receivables": 120.0, "market_value": 800.0,
            "net_sales": 1000.0, "total_assets": 900.0, "lt_debt": 150.0, "ebit": 130.0,
            "gross_profit": 300.0, "curr_liabilities": 180.0, "retained_earnings": 300.0,
            "total_revenue": 1000.0, "total_liabilities": 400.0, "total_opex": 850.0,
        },
        "prev_year": None,
        "expected_probability": 0.0011706747789437442,
    },
    {
        "name": "distressed_small_firm",
        "financials": {
            "current_assets": 50.0, "cogs": 300.0, "dep_amort": 15.0, "ebitda": -40.0,
            "inventory": 80.0, "net_income": -90.0, "receivables": 30.0, "market_value": 10.0,
            "net_sales": 250.0, "total_assets": 200.0, "lt_debt": 150.0, "ebit": -60.0,
            "gross_profit": -20.0, "curr_liabilities": 180.0, "retained_earnings": -150.0,
            "total_revenue": 250.0, "total_liabilities": 350.0, "total_opex": 300.0,
        },
        "prev_year": None,
        "expected_probability": 0.002659888549236743,
    },
    {
        "name": "zero_inventory_service_firm",
        "financials": {
            "current_assets": 200.0, "cogs": 100.0, "dep_amort": 5.0, "ebitda": 60.0,
            "inventory": 0.0, "net_income": 30.0, "receivables": 80.0, "market_value": 400.0,
            "net_sales": 300.0, "total_assets": 350.0, "lt_debt": 20.0, "ebit": 55.0,
            "gross_profit": 200.0, "curr_liabilities": 60.0, "retained_earnings": 100.0,
            "total_revenue": 300.0, "total_liabilities": 100.0, "total_opex": 250.0,
        },
        "prev_year": None,
        "expected_probability": 0.0010684217301788217,
    },
    {
        "name": "deteriorating_firm_with_trend",
        "financials": {
            "current_assets": 150.0, "cogs": 400.0, "dep_amort": 20.0, "ebitda": 10.0,
            "inventory": 150.0, "net_income": -30.0, "receivables": 60.0, "market_value": 100.0,
            "net_sales": 450.0, "total_assets": 500.0, "lt_debt": 200.0, "ebit": -5.0,
            "gross_profit": 80.0, "curr_liabilities": 220.0, "retained_earnings": 20.0,
            "total_revenue": 450.0, "total_liabilities": 380.0, "total_opex": 440.0,
        },
        "prev_year": {
            "current_assets": 200.0, "cogs": 380.0, "dep_amort": 18.0, "ebitda": 60.0,
            "inventory": 130.0, "net_income": 15.0, "receivables": 70.0, "market_value": 180.0,
            "net_sales": 480.0, "total_assets": 480.0, "lt_debt": 190.0, "ebit": 40.0,
            "gross_profit": 100.0, "curr_liabilities": 200.0, "retained_earnings": 50.0,
            "total_revenue": 480.0, "total_liabilities": 340.0, "total_opex": 420.0,
        },
        "expected_probability": 0.00963165373645673,
    },
    {
        "name": "improving_firm_with_trend",
        "financials": {
            "current_assets": 300.0, "cogs": 500.0, "dep_amort": 25.0, "ebitda": 90.0,
            "inventory": 180.0, "net_income": 40.0, "receivables": 90.0, "market_value": 350.0,
            "net_sales": 650.0, "total_assets": 600.0, "lt_debt": 100.0, "ebit": 65.0,
            "gross_profit": 220.0, "curr_liabilities": 150.0, "retained_earnings": 120.0,
            "total_revenue": 650.0, "total_liabilities": 250.0, "total_opex": 560.0,
        },
        "prev_year": {
            "current_assets": 260.0, "cogs": 480.0, "dep_amort": 22.0, "ebitda": 60.0,
            "inventory": 160.0, "net_income": 15.0, "receivables": 80.0, "market_value": 300.0,
            "net_sales": 600.0, "total_assets": 550.0, "lt_debt": 110.0, "ebit": 40.0,
            "gross_profit": 190.0, "curr_liabilities": 160.0, "retained_earnings": 100.0,
            "total_revenue": 600.0, "total_liabilities": 260.0, "total_opex": 540.0,
        },
        "expected_probability": 0.001446250126003422,
    },
]


@pytest.mark.parametrize("case", GOLDEN_INPUTS, ids=[c["name"] for c in GOLDEN_INPUTS])
def test_golden_prediction_matches_frozen_probability(case):
    if case["expected_probability"] is None:
        pytest.skip(f"{case['name']}: expected_probability not yet frozen from trained artifacts")
    result = predict_one(case["financials"], case["prev_year"])
    assert result["probability"] == pytest.approx(case["expected_probability"], abs=1e-6)
