# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI4All Group 23C project predicting US corporate bankruptcy from annual financial statement data (`american_bankruptcy.csv`, 78,682 firm-years, 1999–2018). The entire ML pipeline lives in one Jupyter notebook: `AI4All_Group_23C_Project_Code.ipynb`.

## Running the Notebook

This project runs in Google Colab or a local Jupyter environment.

```bash
# Install dependencies (if running locally)
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap imbalanced-learn

# Launch Jupyter
jupyter notebook AI4All_Group_23C_Project_Code.ipynb
```

The CSV must be in the same directory as the notebook. All cells must be run in order — later cells depend on state built by earlier ones.

## Pipeline Architecture

The notebook is organized into five sequential sections:

1. **Loading Data** — Reads CSV, renames encoded columns (`X1`–`X18`) to human-readable names.

2. **Data Cleaning** — Fixes a label leakage issue: every historical row of a failed firm was originally marked `"failed"`. The notebook corrects this so only the firm's final observed year receives `target=1` (bankrupt next year); all prior rows are relabeled `alive`. This produces 609 true positives from 78,682 rows (~0.77% base rate).

3. **EDA** — Distribution plots, bankruptcy-rate-by-year chart, feature correlation heatmaps, and trajectory plots for example failed firms.

4. **Feature Engineering** — All engineered features are derived before the train/val/test split:
   - **Altman Z-Score** and its five sub-components (`altman_x1`–`altman_x5`)
   - **Financial ratios**: profitability (ROA, EBITDA/TA, margins), leverage (liab/TA, LT debt/TA), liquidity (current ratio, inventory turnover, DSO)
   - **Trend features**: year-over-year and 2-year slope changes in EBITDA, net income, current assets, inventory turnover — masked to 0 when a firm has year gaps
   - **Structural flags**: `inventory_is_zero`, `consec_neg_ni`, `years_of_history`
   - **K-Means cluster label** (`industry_cluster_1`): fit on training rows only, applied to all rows
   - **Data leakage prevention**: `TRAIN_MASK = df["year"] <= 2011`; winsorization caps, imputation medians, and the K-Means/StandardScaler are all fit exclusively on training rows, then applied to val/test.

5. **Model Building** — Five models, evaluated on a held-out 2015–2018 test set:
   - **Logistic Regression** (`FEATURES_LR`, excludes `altman_z` and cluster dummy to avoid multicollinearity)
   - **Altman Z-Score baseline** (fixed 1968 formula, included as a reference point)
   - **XGBoost (primary)** — hyperparameters tuned via `TimeSeriesSplit` CV with firm-level leakage prevention; threshold selected from val set; final model retrained on train+val
   - **XGBoost (SMOTE)** — same hyperparameters but trained on SMOTE-resampled data instead of `scale_pos_weight`
   - **Decision Tree** (depth=3, `FEATURES_DT`, for interpretability) and **Random Forest**

6. **Evaluation** — Confusion matrices, metrics table (ROC-AUC, PR-AUC, F1, Precision, Recall), PR curves, ROC curves, SHAP beeswarm/bar/waterfall plots, and Platt-scaled probability calibration.

## Key Design Decisions

- **Time-based splits** (not random): train 1999–2011, val 2012–2014, test 2015–2018. This prevents future data leakage.
- **CV folds use grouped exclusion**: any firm with a row in a fold's validation window is fully removed from that fold's training set to prevent firm-level leakage.
- **`safe_div()`** sets near-zero denominators to NaN, which are later imputed with training medians.
- **Calibration model** uses `FrozenEstimator` wrapping the tuning-stage XGBoost (never trained on val), with Platt scaling fit on val only.
- **Feature sets differ by model**: `FEATURES_LR`, `FEATURES_DT`, and `FEATURES_ALL` are defined separately to handle multicollinearity concerns per model type.
