# Bankruptcy Risk Screener

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

Screens a US public company's annual financial statement data for one-year bankruptcy risk using
a calibrated XGBoost model trained on 20 years of NYSE/NASDAQ filings. Built on top of the analysis
in `notebooks/AI4All_Group_23C_Project_Code.ipynb`, which remains the source of truth for every
transformation, hyperparameter, and threshold used here.

## Quickstart

```bash
pip install -r requirements.txt

# run the test suite
pytest -q

# launch the app (uses the artifacts already committed under models/)
streamlit run app/streamlit_app.py
```

To retrain from scratch:

```bash
python -m src.train
```

This reads `american_bankruptcy.csv` from the repo root, reproduces the full pipeline, and
overwrites everything in `models/`.

## Model card

**Data.** 78,682 US public company firm-years (1999-2018), from the Kaggle American Bankruptcy
dataset of NYSE/NASDAQ-listed firms, 18 raw financial statement fields per firm-year.

**Target definition.** One-year bankruptcy filing. The raw data marks every historical row of a
firm that eventually failed as `"failed"`, so the label was corrected: only a firm's final
observed year is counted as a positive (bankrupt next year), and all prior years are relabeled
`alive`. This produces roughly 609 true positives out of 78,682 rows (about 0.77% base rate).

**Features.** Altman Z-Score and its five components, profitability ratios (ROA, EBITDA/TA,
margins), leverage ratios (liabilities/TA, long-term debt/TA), liquidity ratios (current ratio,
inventory turnover, days sales outstanding), year-over-year and 2-year slope trend features
masked across year gaps, consecutive years of negative net income, years of firm history, and an
unsupervised industry cluster label (K-Means, fit on training rows only).

**Model.** XGBoost, hyperparameters tuned by time-series cross-validation with firm-level
leakage prevention (any firm with a row in a fold's validation window is fully excluded from
that fold's training rows). The tuning-stage model is trained on 1999-2011 data only.

**Calibration.** Platt scaling (sigmoid), fit on the 2012-2014 validation set, wrapping the
tuning-stage model rather than a train+val refit, so its probabilities are calibrated on data it
never trained on. This is the model deployed in the app. The frozen decision threshold is the
F1-optimal point on validation, expressed in calibrated-probability units.

**Test metrics (2015-2018).** See the Model Performance tab in the app, or `models/metrics.json`,
for the full comparison table across Logistic Regression, Altman Z baseline, XGBoost (primary),
XGBoost with SMOTE, Decision Tree, Random Forest, and the deployed calibrated model. The test set
has only 119 true positives, so these metrics carry wide uncertainty; a handful of different
outcomes would move them noticeably.

**Limitations.** Trained only on public NYSE/NASDAQ-listed US firms. Predicts filing within one
year of the reported financials, not a multi-year risk estimate. The alive class carries
survivorship bias, since firms that were acquired, delisted, or stopped reporting for reasons
other than bankruptcy are not distinguished from firms that stayed healthy. This is a screening
tool meant to prioritize further review, not an automated decision system.

## Drift policy

New data must pass `src/validate.py` (`validate_for_training`, plus `check_drift` against the
saved training reference stats) before being used for anything. Retraining is a manual
`python -m src.train` run; there is no automated retraining pipeline. The app always scores
against whatever is currently committed in `models/`.

## Repo structure

```
bankruptcy-predictor/
  american_bankruptcy.csv   raw dataset
  src/
    constants.py             shared column lists and split boundaries
    validate.py               schema and label-structure checks
    features.py                relabeling, ratios, trends, winsorization, imputation, clustering
    train.py                    reproduces the notebook pipeline end to end, saves artifacts
    predict.py                  loads artifacts, scores new input
  app/
    streamlit_app.py           three-tab Streamlit app
  tests/                       pytest suite
  models/                      serialized artifacts (committed, not regenerated in CI)
  notebooks/
    AI4All_Group_23C_Project_Code.ipynb   original notebook, untouched
  .github/workflows/ci.yml     ruff + pytest on push and PR
```
