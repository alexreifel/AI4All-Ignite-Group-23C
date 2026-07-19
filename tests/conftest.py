import pandas as pd
import pytest

from src.constants import RAW_DOLLAR_COLS


def make_firm_rows(company_name, years, status_label="alive", **overrides):
    """Build minimal raw-dollar-column rows for a firm across the given years.

    Defaults give sane non-zero denominators everywhere so safe_div never
    produces a NaN unless a test deliberately overrides a column to test that.
    """
    rows = []
    for year in years:
        row = {c: 100.0 for c in RAW_DOLLAR_COLS}
        row["company_name"] = company_name
        row["status_label"] = status_label
        row["year"] = year
        for key, values in overrides.items():
            row[key] = values[years.index(year)]
        rows.append(row)
    return rows


@pytest.fixture
def tiny_df():
    rows = []
    rows += make_firm_rows("firm_alive", [2000, 2001, 2002], status_label="alive")
    rows += make_firm_rows("firm_failed", [2000, 2001, 2002], status_label="failed")
    return pd.DataFrame(rows)
