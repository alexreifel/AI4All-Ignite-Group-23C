import pandas as pd
import pytest

from src import validate
from tests.conftest import make_firm_rows


def test_check_schema_passes_on_clean_data(tiny_df):
    validate.check_schema(tiny_df)


def test_check_schema_fails_on_missing_column(tiny_df):
    df = tiny_df.drop(columns=["total_assets"])
    with pytest.raises(ValueError):
        validate.check_schema(df)


def test_check_schema_fails_on_duplicate_firm_year(tiny_df):
    df = pd.concat([tiny_df, tiny_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate.check_schema(df)


def test_check_schema_fails_on_nulls(tiny_df):
    df = tiny_df.copy()
    df.loc[0, "total_assets"] = None
    with pytest.raises(ValueError):
        validate.check_schema(df)


def test_check_label_structure_passes_when_constant_per_firm(tiny_df):
    validate.check_label_structure(tiny_df)


def test_check_label_structure_fails_on_inconsistent_labels():
    rows = make_firm_rows("flaky_firm", [2000, 2001, 2002], status_label="alive")
    rows[1]["status_label"] = "failed"
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        validate.check_label_structure(df)


def test_validate_for_training_raises_on_bad_data():
    rows = make_firm_rows("flaky_firm", [2000, 2001], status_label="alive")
    rows[1]["status_label"] = "failed"
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        validate.validate_for_training(df)
