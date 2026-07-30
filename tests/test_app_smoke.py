from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.predict import predict_one

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_runs_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception


def test_app_defaults_to_csv_batch_upload():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.radio(key="screen_input_mode").value == "CSV Batch Upload"


def test_manual_entry_mode_runs_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["screen_input_mode"] = "Manual Entry"
    at.run()
    assert not at.exception


def test_predict_one_returns_probability_in_unit_interval():
    financials = {
        "current_assets": 500.0, "cogs": 700.0, "dep_amort": 20.0, "ebitda": 150.0,
        "inventory": 200.0, "net_income": 60.0, "receivables": 120.0, "market_value": 800.0,
        "net_sales": 1000.0, "total_assets": 900.0, "lt_debt": 150.0, "ebit": 130.0,
        "gross_profit": 300.0, "curr_liabilities": 180.0, "retained_earnings": 300.0,
        "total_revenue": 1000.0, "total_liabilities": 400.0, "total_opex": 850.0,
    }
    result = predict_one(financials, None)
    assert 0.0 <= result["probability"] <= 1.0
