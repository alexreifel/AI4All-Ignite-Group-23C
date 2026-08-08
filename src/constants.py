"""Shared constants for the bankruptcy prediction pipeline.

These lists mirror the notebooks/ pipeline exactly (see notebooks/04_feature_engineering.ipynb).
They live in one place so validate.py, features.py, train.py, and predict.py
cannot drift apart on column names or feature order.
"""

# Map dataset columns to make them readable, same mapping as the notebook.
COLMAP = {
    "X1": "current_assets",     "X2": "cogs",               "X3": "dep_amort",
    "X4": "ebitda",             "X5": "inventory",          "X6": "net_income",
    "X7": "receivables",        "X8": "market_value",       "X9": "net_sales",
    "X10": "total_assets",      "X11": "lt_debt",           "X12": "ebit",
    "X13": "gross_profit",      "X14": "curr_liabilities",  "X15": "retained_earnings",
    "X16": "total_revenue",     "X17": "total_liabilities", "X18": "total_opex",
}
RAW_DOLLAR_COLS = list(COLMAP.values())

# Ratios built in the feature engineering cell, in the notebook's order.
RATIO_COLS = [
    "altman_x1", "altman_x2", "altman_x3", "altman_x4", "altman_x5",
    "roa", "ebitda_ta", "net_margin", "gross_margin",
    "liab_ta", "ltdebt_ta", "current_ratio", "inventory_turnover",
    "dso", "altman_z",
]

# Year-over-year and 2-year slope trend features, masked across year gaps.
TREND_COLS = [
    "d_ebitda_ta", "d_net_income_ta", "pct_chg_ca",
    "ebitda_slope2_ta", "d_inv_turnover",
]

# Columns winsorized together at train-window 1st/99th percentile.
CAP_COLS = RATIO_COLS + TREND_COLS

# altman_z is excluded from clustering since it's a fixed linear combination
# of altman_x1-x5, which are already in RATIO_COLS.
CLUSTER_COLS = [c for c in RATIO_COLS if c != "altman_z"]

# Altman Z-Score fixed 1968 weights.
ALTMAN_COEFS = {
    "altman_x1": 1.2, "altman_x2": 1.4, "altman_x3": 3.3,
    "altman_x4": 0.6, "altman_x5": 1.0,
}
ALTMAN_LOW = 1.81
ALTMAN_HIGH = 2.99

# Feature set for tree/boosting models.
FEATURES_ALL = (
    RATIO_COLS + TREND_COLS +
    ["consec_neg_ni", "years_of_history", "inventory_is_zero", "log_total_assets"] +
    ["industry_cluster_1"]
)

# Feature set for Logistic Regression: drop industry cluster and altman_z
# to avoid multicollinearity.
FEATURES_LR = [f for f in FEATURES_ALL if f not in ["industry_cluster_1", "altman_z"]]

# Feature set for Decision Tree: drop altman_z to avoid overfitting on
# Altman Z-Score splits.
FEATURES_DT = [f for f in FEATURES_ALL if f != "altman_z"]

# Human-readable labels for FEATURES_ALL, used to render SHAP explanations
# understandably in the app. Must cover every key in FEATURES_ALL.
FEATURE_LABELS = {
    "altman_x1": "Working capital / Total assets",
    "altman_x2": "Retained earnings / Total assets",
    "altman_x3": "EBIT / Total assets",
    "altman_x4": "Market equity / Total liabilities",
    "altman_x5": "Sales / Total assets",
    "roa": "Return on assets",
    "ebitda_ta": "EBITDA / Total assets",
    "net_margin": "Net margin",
    "gross_margin": "Gross margin",
    "liab_ta": "Total liabilities / Total assets",
    "ltdebt_ta": "Long-term debt / Total assets",
    "current_ratio": "Current ratio",
    "inventory_turnover": "Inventory turnover",
    "dso": "Days sales outstanding",
    "altman_z": "Altman Z-Score",
    "d_ebitda_ta": "Change in EBITDA / TA (YoY)",
    "d_net_income_ta": "Change in net income / TA (YoY)",
    "pct_chg_ca": "% change in current assets (YoY)",
    "ebitda_slope2_ta": "2-year EBITDA / TA slope",
    "d_inv_turnover": "Change in inventory turnover (YoY)",
    "consec_neg_ni": "Consecutive years of negative net income",
    "years_of_history": "Years of financial history available",
    "inventory_is_zero": "Zero-inventory flag",
    "log_total_assets": "Log of total assets",
    "industry_cluster_1": "Industry cluster (unsupervised)",
}

# Time-based train/val/test split years, inclusive on both ends.
TRAIN_YEARS = (1999, 2011)
VAL_YEARS = (2012, 2014)
TEST_YEARS = (2015, 2018)

SAFE_DIV_FLOOR = 1e-6

# Expected silhouette-selected cluster count. FEATURES_ALL hard-codes
# industry_cluster_1, so training must land on k=2 for that column to exist.
EXPECTED_K_OPT = 2
