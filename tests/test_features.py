import pandas as pd
import pytest

from src.features import build_ratios, build_trends, compute_caps, compute_medians, relabel_target, safe_div
from tests.conftest import make_firm_rows


def test_safe_div_zero_denominator_is_nan():
    result = safe_div(pd.Series([10.0]), pd.Series([0.0]))
    assert result.isna().all()


def test_safe_div_near_zero_denominator_is_nan():
    result = safe_div(pd.Series([10.0]), pd.Series([1e-8]))
    assert result.isna().all()


def test_safe_div_negative_denominator_below_floor_is_nan():
    result = safe_div(pd.Series([10.0]), pd.Series([-1e-8]))
    assert result.isna().all()


def test_safe_div_negative_denominator_above_floor_is_valid():
    result = safe_div(pd.Series([10.0]), pd.Series([-2.0]))
    assert result.iloc[0] == -5.0


def test_safe_div_normal_division():
    result = safe_div(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
    assert list(result) == [5.0, 5.0]


def test_relabel_target_only_last_row_of_failed_firm_is_positive():
    rows = make_firm_rows("firm_failed", [2000, 2001, 2002], status_label="failed")
    result = relabel_target(pd.DataFrame(rows))
    assert result["target"].tolist() == [0, 0, 1]


def test_relabel_target_alive_firm_all_zero():
    rows = make_firm_rows("firm_alive", [2000, 2001], status_label="alive")
    result = relabel_target(pd.DataFrame(rows))
    assert result["target"].tolist() == [0, 0]


def test_relabel_target_sorts_by_company_and_year():
    rows = make_firm_rows("firm_failed", [2002, 2000, 2001], status_label="failed")
    result = relabel_target(pd.DataFrame(rows))
    assert result["year"].tolist() == [2000, 2001, 2002]
    assert result["target"].tolist() == [0, 0, 1]


def test_trend_masking_across_year_gap():
    rows = make_firm_rows(
        "gapped_firm", [2000, 2001, 2003],
        ebitda=[10.0, 20.0, 40.0],
        total_assets=[100.0, 100.0, 100.0],
    )
    df = build_trends(build_ratios(pd.DataFrame(rows)))

    consecutive = df.loc[df["year"] == 2001, "d_ebitda_ta"].iloc[0]
    assert consecutive == pytest.approx((20.0 - 10.0) / 100.0)

    gapped = df.loc[df["year"] == 2003, "d_ebitda_ta"].iloc[0]
    assert gapped == 0.0


def test_years_of_history_counts_rows_in_order():
    rows = make_firm_rows("firm_alive", [2000, 2001, 2002])
    df = build_trends(build_ratios(pd.DataFrame(rows)))
    assert df["years_of_history"].tolist() == [1, 2, 3]


def test_consec_neg_ni_resets_on_sign_change():
    rows = make_firm_rows(
        "firm_alive", [2000, 2001, 2002, 2003],
        net_income=[-5.0, -5.0, 5.0, -5.0],
    )
    df = build_trends(build_ratios(pd.DataFrame(rows)))
    assert df["consec_neg_ni"].tolist() == [1, 2, 0, 1]


def test_compute_caps_and_medians_use_train_mask_only(tiny_df):
    df = build_trends(build_ratios(tiny_df))
    train_mask = df["year"] <= 2001

    caps = compute_caps(df, train_mask, ["roa"])
    expected_caps = tuple(df.loc[train_mask, "roa"].quantile([0.01, 0.99]).tolist())
    assert caps["roa"] == pytest.approx(expected_caps)

    medians = compute_medians(df, train_mask, ["roa"])
    assert medians["roa"] == pytest.approx(float(df.loc[train_mask, "roa"].median()))
