"""Feature engineering: relabeling, ratios, trends, winsorization, imputation,
and cluster assignment.

Every function here is a direct port of the notebook's feature engineering
cells. The compute_*/apply_* functions are split on purpose: compute_* is
only ever called on training data (fit), apply_* is called on any data
(fit or inference) using already-fitted parameters. predict.py never calls
a compute_* function, so it structurally cannot recompute a winsorization
cap, an imputation median, or a cluster fit from user input.
"""

import numpy as np

from src.constants import RATIO_COLS, TREND_COLS, SAFE_DIV_FLOOR


def safe_div(num, den, floor=SAFE_DIV_FLOOR):
    """Ratio with near-zero/negative denominators set to NaN."""
    out = num / den.where(den.abs() >= floor)
    return out.replace([np.inf, -np.inf], np.nan)


def relabel_target(df):
    """Sort by (company_name, year) and build the corrected target.

    Every historical row of a failed firm is originally marked "failed" in
    status_label. Only the firm's final observed year should count as a
    positive (bankrupt next year); all prior rows are relabeled alive.
    """
    df = df.sort_values(["company_name", "year"]).reset_index(drop=True)
    final_year = df.groupby("company_name")["year"].transform("max")
    is_final_row = df["year"] == final_year
    df["target"] = ((df["status_label"] == "failed") & is_final_row).astype(int)
    return df


def build_ratios(df):
    """Add inventory_is_zero, Altman Z-Score components, and financial ratios."""
    df = df.copy()

    df["inventory_is_zero"] = (df["inventory"] == 0).astype(int)

    df["altman_x1"] = safe_div(df["current_assets"] - df["curr_liabilities"], df["total_assets"])
    df["altman_x2"] = safe_div(df["retained_earnings"], df["total_assets"])
    df["altman_x3"] = safe_div(df["ebit"], df["total_assets"])
    df["altman_x4"] = safe_div(df["market_value"], df["total_liabilities"])
    df["altman_x5"] = safe_div(df["net_sales"], df["total_assets"])
    df["altman_z"] = (1.2 * df["altman_x1"] + 1.4 * df["altman_x2"] +
                      3.3 * df["altman_x3"] + 0.6 * df["altman_x4"] +
                      1.0 * df["altman_x5"])

    df["roa"] = safe_div(df["net_income"], df["total_assets"])
    df["ebitda_ta"] = safe_div(df["ebitda"], df["total_assets"])
    df["net_margin"] = safe_div(df["net_income"], df["total_revenue"].where(df["total_revenue"] > 0))
    df["gross_margin"] = safe_div(df["gross_profit"], df["net_sales"].where(df["net_sales"] > 0))

    df["liab_ta"] = safe_div(df["total_liabilities"], df["total_assets"])
    df["ltdebt_ta"] = safe_div(df["lt_debt"], df["total_assets"])

    df["current_ratio"] = safe_div(df["current_assets"], df["curr_liabilities"])
    df["inventory_turnover"] = safe_div(df["cogs"], df["inventory"])
    df["dso"] = safe_div(df["receivables"], df["net_sales"].where(df["net_sales"] > 0)) * 365

    return df


def build_trends(df):
    """Add year-over-year/2-year trend features, masked across year gaps,
    plus consec_neg_ni and years_of_history.

    Requires build_ratios to have already run, since d_inv_turnover uses the
    inventory_turnover column built there.
    """
    df = df.copy()
    g = df.groupby("company_name")
    gap1 = g["year"].diff().eq(1)
    gap2 = gap1 & gap1.groupby(df["company_name"]).shift(1).fillna(False)

    df["d_ebitda_ta"] = (g["ebitda"].diff() / df["total_assets"]).where(gap1)
    df["d_net_income_ta"] = (g["net_income"].diff() / df["total_assets"]).where(gap1)
    df["pct_chg_ca"] = (g["current_assets"].pct_change().where(gap1).replace([np.inf, -np.inf], np.nan))
    df["ebitda_slope2_ta"] = ((df["ebitda"] - g["ebitda"].shift(2)) / 2 / df["total_assets"]).where(gap2)
    df["d_inv_turnover"] = g["inventory_turnover"].diff().where(gap1)

    neg = (df["net_income"] < 0).astype(int)
    block = ((neg != neg.shift()) | (df["company_name"] != df["company_name"].shift())).cumsum()
    df["consec_neg_ni"] = neg.groupby(block).cumsum()

    df[TREND_COLS] = df[TREND_COLS].fillna(0)
    df["years_of_history"] = g.cumcount() + 1

    return df


def compute_caps(df, train_mask, cols):
    """1st/99th percentile caps for cols, fit on train_mask rows only."""
    q = df.loc[train_mask, cols].quantile([0.01, 0.99])
    return {c: (float(q.loc[0.01, c]), float(q.loc[0.99, c])) for c in cols}


def apply_caps(df, caps):
    """Clip cols to previously-fitted caps. Does not compute new caps."""
    df = df.copy()
    for col, (lo, hi) in caps.items():
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def compute_medians(df, train_mask, cols):
    """Medians for cols, fit on train_mask rows only."""
    med = df.loc[train_mask, cols].median()
    return {c: float(med[c]) for c in cols}


def apply_medians(df, medians):
    """Fill NaNs in cols with previously-fitted medians. Does not compute new medians."""
    df = df.copy()
    df[list(medians.keys())] = df[list(medians.keys())].fillna(medians)
    return df


def compute_ta_log_caps(df, train_mask):
    """1st/99th percentile caps for total_assets, fit on train_mask rows only."""
    q = df.loc[train_mask, "total_assets"].quantile([0.01, 0.99])
    return (float(q[0.01]), float(q[0.99]))


def apply_log_total_assets(df, ta_caps):
    """Add log_total_assets using previously-fitted total_assets caps."""
    df = df.copy()
    lo, hi = ta_caps
    df["log_total_assets"] = np.log1p(df["total_assets"].clip(lo, hi))
    return df


def assign_cluster(df, scaler, kmeans, cluster_cols):
    """Assign industry_cluster and industry_cluster_1 using a fitted scaler/KMeans.

    Pure apply: scaler and kmeans must already be fitted on training data.
    """
    df = df.copy()
    Xc = scaler.transform(df[cluster_cols])
    df["industry_cluster"] = kmeans.predict(Xc)
    df["industry_cluster_1"] = (df["industry_cluster"] == 1).astype(int)
    return df


def engineer_features(df, train_mask=None, fit=True, artifacts=None):
    """Orchestrate ratio/trend construction, winsorization, imputation, and
    log_total_assets.

    fit=True: train_mask is required. Computes caps/medians/ta_caps from
    train_mask rows and applies them to the whole frame. Returns
    (df, artifacts) where artifacts holds the fitted caps/medians/ta_caps.

    fit=False: artifacts is required (as returned by a prior fit=True call,
    or loaded from disk). Applies the given artifacts only; no quantile or
    median is computed from the input df.
    """
    df = build_ratios(df)
    df = build_trends(df)

    cap_cols = RATIO_COLS + TREND_COLS

    if fit:
        if train_mask is None:
            raise ValueError("train_mask is required when fit=True")
        caps = compute_caps(df, train_mask, cap_cols)
        df = apply_caps(df, caps)
        medians = compute_medians(df, train_mask, RATIO_COLS)
        df = apply_medians(df, medians)
        ta_caps = compute_ta_log_caps(df, train_mask)
        df = apply_log_total_assets(df, ta_caps)
        fitted = {"caps": caps, "medians": medians, "ta_caps": ta_caps}
        return df, fitted
    else:
        if artifacts is None:
            raise ValueError("artifacts is required when fit=False")
        df = apply_caps(df, artifacts["caps"])
        df = apply_medians(df, artifacts["medians"])
        df = apply_log_total_assets(df, artifacts["ta_caps"])
        return df
