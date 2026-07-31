from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.constants import FEATURE_LABELS, FEATURES_ALL
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


def test_feature_labels_cover_all_features():
    assert set(FEATURE_LABELS) == set(FEATURES_ALL)


def test_csv_batch_explain_flow_runs_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    csv_bytes = (
        "company_name,year,current_assets,cogs,dep_amort,ebitda,inventory,net_income,"
        "receivables,market_value,net_sales,total_assets,lt_debt,ebit,gross_profit,"
        "curr_liabilities,retained_earnings,total_revenue,total_liabilities,total_opex\n"
        "Acme Co,2016,500,700,20,150,200,60,120,800,1000,900,150,130,300,180,300,1000,400,850\n"
        "Acme Co,2017,520,710,20,160,210,65,125,820,1050,920,150,135,310,185,305,1050,410,860\n"
    ).encode("utf-8")
    at.file_uploader(key="csv_upload").set_value(("batch.csv", csv_bytes, "text/csv"))
    at.run()
    at.button(key="csv_submit").click().run()
    assert not at.exception

    at.selectbox(key="csv_explain_company").set_value("Acme Co").run()
    at.button(key="csv_explain_submit").click().run()
    assert not at.exception
    assert "csv_explanation" in at.session_state
