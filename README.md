# Bankruptcy Risk Screener

[![CI](https://github.com/alexreifel/AI4All-Ignite-Group-23C/actions/workflows/ci.yml/badge.svg)](https://github.com/alexreifel/AI4All-Ignite-Group-23C/actions/workflows/ci.yml)

## Overview

A tool that reads a US public company's yearly financial statements and estimates the chance it files for bankruptcy within the next year. It runs on a calibrated XGBoost model trained on 20 years of NYSE and NASDAQ filings, and we benchmark it against the Altman Z-Score, the formula the industry has leaned on since 1968, to see whether modern machine learning actually does better.

Built by **AI4ALL Ignite Summer 2026, Group 23C**: Michelle Jiang, Alex Reifel, Palak Goindwani, Abdurrahman Oyediran, Rashid Mikidadi, and Eddy.

The full analysis lives in `notebooks/AI4All_Group_23C_Project_Code.ipynb`, which stays the source of truth for every transformation, hyperparameter, and threshold used in the app.

## How to Get Started

Run the project locally in a few steps.

1. **Clone the repository**

```
git clone https://github.com/alexreifel/AI4All-Ignite-Group-23C.git
cd AI4All-Ignite-Group-23C
```

2. **Install the packages** (a virtual environment is cleaner)

```
python -m venv .venv               # optional
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the tests** to confirm everything works

```
pytest -q
```

4. **Launch the app** (it uses the trained model already saved under `models/`)

```
streamlit run app/streamlit_app.py
```

To retrain the model from scratch, run `python -m src.train`. This reads `american_bankruptcy.csv` from the repo root, runs the whole pipeline again, and overwrites everything in `models/`.

## What We Learned

A few things stuck with us by the end of this project.

Accuracy is a trap when the thing you care about almost never happens. Only about 0.77% of the company-years in our data end in bankruptcy, so a model that predicts "everyone survives" scores over 99% and catches nothing. We leaned on PR-AUC and recall instead, which actually reward finding the rare failures.

Cleaning the labels mattered more than any fancy model. The raw data marked every year of a doomed company as "failed," even years when it was doing fine. Fixing that so only the final year counts as bankrupt changed the whole problem, and skipping it would have quietly taught the model nonsense.

How you split the data is a decision, not a detail. Because the same firm shows up across many years, a random split would let the model peek at a company's future. Splitting by time instead, with the 2008 crisis sitting only in the training years, forced the model to prove it still works in calmer periods.

Modern ML really does beat the old formula, but not by magic. Our XGBoost model outscored the 1968 Altman Z-Score on the test set, and SHAP showed it weighs the same financial signals quite differently than Altman did. Still, it flags a lot of false alarms, so we treat it as a screening tool that points humans toward companies worth a closer look, not a final verdict.

## Repo Structure

```
AI4All-Ignite-Group-23C/
  american_bankruptcy.csv   raw dataset
  src/
    constants.py            shared column lists and split boundaries
    validate.py             schema and label-structure checks
    features.py             relabeling, ratios, trends, winsorization, imputation, clustering
    train.py                reproduces the notebook pipeline end to end, saves artifacts
    predict.py              loads artifacts, scores new input
  app/
    streamlit_app.py        three-tab Streamlit app
  tests/                    pytest suite
  models/                   serialized model artifacts (committed, not regenerated in CI)
  notebooks/
    AI4All_Group_23C_Project_Code.ipynb   original notebook, untouched
  .github/workflows/ci.yml  ruff + pytest on push and PR
```

## Model Card

**Data.** 78,682 US public company firm-years (1999-2018) from the Kaggle American Bankruptcy dataset of NYSE and NASDAQ firms, with 18 raw financial statement fields per firm-year.

**Target definition.** One-year bankruptcy filing. The raw data marks every historical row of a firm that eventually failed as `"failed"`, so we corrected the label: only a firm's final observed year counts as a positive (bankrupt next year), and all prior years are relabeled `alive`. This leaves roughly 609 true positives out of 78,682 rows, about a 0.77% base rate.

**Features.** The Altman Z-Score and its five components, profitability ratios (ROA, EBITDA/TA, margins), leverage ratios (liabilities/TA, long-term debt/TA), liquidity ratios (current ratio, inventory turnover, days sales outstanding), year-over-year and 2-year slope trends masked across year gaps, consecutive years of negative net income, years of firm history, and an unsupervised industry cluster label from K-Means fit on training rows only.

**Model.** XGBoost, tuned by time-series cross-validation with firm-level leakage prevention, so any firm appearing in a fold's validation window is fully excluded from that fold's training rows. The tuning-stage model trains on 1999-2011 data only.

**Calibration.** Platt scaling (sigmoid) fit on the 2012-2014 validation set, wrapping the tuning-stage model rather than a train-plus-validation refit, so the probabilities are calibrated on data the model never trained on. This is the version deployed in the app. The decision threshold is the F1-optimal point on validation, in calibrated-probability units.

**Test metrics (2015-2018).** See the Model Performance tab in the app, or `models/metrics.json`, for the full comparison across Logistic Regression, the Altman Z baseline, XGBoost (primary), XGBoost with SMOTE, Decision Tree, Random Forest, and the deployed calibrated model. The test set has only 119 true positives, so these numbers carry real uncertainty and a handful of different outcomes would move them noticeably.

**Limitations.** Trained only on public NYSE and NASDAQ US firms. It predicts a filing within one year of the reported financials, not a multi-year risk estimate. The alive class carries survivorship bias, since firms that were acquired, delisted, or simply stopped reporting for non-bankruptcy reasons are not separated from firms that stayed healthy. This is a screening tool meant to prioritize further review, not an automated decision system.

## Drift Policy

New data has to pass `src/validate.py` (`validate_for_training`, plus `check_drift` against the saved training reference stats) before it's used for anything. Retraining is a manual `python -m src.train` run, with no automated pipeline. The app always scores against whatever is currently committed in `models/`.
