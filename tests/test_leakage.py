import pandas as pd
import pytest

from src.constants import RATIO_COLS, TREND_COLS
from src.features import build_ratios, build_trends, compute_caps, compute_medians
from tests.conftest import make_firm_rows


def test_caps_and_medians_match_train_window_only_computation():
    """Proves compute_caps/compute_medians never let a non-train row influence
    the fitted statistics, by comparing against an independently computed
    train-only quantile/median.
    """
    train_rows = make_firm_rows("firm_a", [2000, 2001], net_income=[10.0, 12.0])
    non_train_rows = make_firm_rows("firm_b", [2015, 2016], net_income=[-500.0, -600.0])
    df = pd.DataFrame(train_rows + non_train_rows)
    df = build_trends(build_ratios(df))
    train_mask = df["year"] <= 2001

    cap_cols = RATIO_COLS + TREND_COLS
    caps = compute_caps(df, train_mask, cap_cols)
    medians = compute_medians(df, train_mask, RATIO_COLS)

    train_only = df.loc[train_mask]
    for col in cap_cols:
        expected = tuple(train_only[col].quantile([0.01, 0.99]).tolist())
        assert caps[col] == pytest.approx(expected), f"{col} cap leaked non-train rows"

    for col in RATIO_COLS:
        expected = float(train_only[col].median())
        assert medians[col] == pytest.approx(expected), f"{col} median leaked non-train rows"

    # The non-train rows are wildly different, so if leakage happened this
    # assertion would fail loudly rather than silently passing.
    full_median_roa = float(df["roa"].median())
    assert medians["roa"] != pytest.approx(full_median_roa)
