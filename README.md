# Bankruptcy Risk Screener

[![CI](https://github.com/alexreifel/AI4All-Ignite-Group-23C/actions/workflows/ci.yml/badge.svg)](https://github.com/alexreifel/AI4All-Ignite-Group-23C/actions/workflows/ci.yml)

## Overview

You've probably heard of Lehman Brothers, or Enron, or Sears. What you may not have heard of are the hundreds of other public companies that quietly file for bankruptcy every year without ever making the news.

Firms fail for all kinds of reasons: a debt load that outgrows the business, a downturn that never reverses, or just years of shrinking margins nobody caught in time. With that much uncertainty built into the market, how are investors, auditors, and employees supposed to know whether the company they're tied to is thriving or quietly circling the drain?

For nearly 60 years, the answer has been the Altman Z-Score: a single formula, built in 1968 from five financial ratios pulled straight off a balance sheet, that scores a company's bankruptcy risk in one number. It's simple, transparent, and still in use today. But finance has changed a lot since 1968. Is a decades-old linear formula still the sharpest tool available, or can modern machine learning read the same financial statements more accurately? That's the question this project set out to answer.

A tool that reads a US public company's yearly financial statements and estimates the chance it files for bankruptcy within the next year. It runs on a calibrated XGBoost model trained on 20 years of NYSE and NASDAQ filings, and we benchmark it against the Altman Z-Score, the formula the industry has leaned on since 1968, to see whether modern machine learning actually does better.

**Why it matters:** a working early-warning signal gives creditors, investors, and employees months of notice instead of a surprise filing, and a transparent, SHAP-explained tool like this one is a free alternative to opaque commercial credit-risk scores.

**How this evolved:** the project started as exploratory analysis on the raw Kaggle CSV in a single notebook, and the biggest turning point was discovering a label leakage bug, where every historical row of a doomed firm was marked "failed" instead of only its final year. Fixing that reshaped the whole target definition. From there the project grew from a one-off notebook into a tested `src/` package and a Streamlit app, with SHAP explainability added last, once the core model was already validated.

Built by **AI4ALL Ignite Summer 2026, Group 23C**: Michelle Jiang, Alex Reifel, Palak Goindwani, Abdurrahman Oyediran, Rashid Mikidadi, and Edomias Zerihun.

The full analysis lives in `notebooks/` (six staged notebooks, `01_loading_data.ipynb` through `06_evaluation.ipynb`), which stays the source of truth for every transformation, hyperparameter, and threshold used in the app.

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

To retrain the model from scratch, run `python -m src.train`. This reads `data/american_bankruptcy.csv`, runs the whole pipeline again, and overwrites everything in `models/`.

To see the original analysis itself instead of just its output, run the six staged notebooks under `notebooks/` in order (see `notebooks/README.md`).

## Tech Stack

**Language:** Python 3.11.

**Data and modeling:** pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, joblib.

**App:** Streamlit.

**Notebooks and plotting:** Jupyter, Matplotlib, seaborn.

**Testing and linting:** pytest, ruff.

**CI/CD:** GitHub Actions.

## Repo Structure

```
AI4All-Ignite-Group-23C/
  data/
    american_bankruptcy.csv    raw dataset
  notebooks/
    01_loading_data.ipynb
    02_data_cleaning.ipynb
    03_eda.ipynb
    04_feature_engineering.ipynb
    05_model_building.ipynb
    06_evaluation.ipynb
    README.md                  run order and how state passes between notebooks
  src/
    constants.py               shared column lists and split boundaries
    validate.py                schema and label-structure checks
    features.py                relabeling, ratios, trends, winsorization, imputation, clustering
    train.py                   reproduces the notebook pipeline end to end, saves artifacts
    predict.py                 loads artifacts, scores new input
  app/
    streamlit_app.py           three-tab Streamlit app
  tests/                       pytest suite
  models/                      serialized model artifacts (committed, not regenerated in CI)
  requirements.txt
  pyproject.toml
  .github/workflows/ci.yml     ruff + pytest on push and PR
```

## Documentation

**Data source:** Kaggle American Bankruptcy dataset, 78,682 US public company firm-years (1999-2018) from NYSE and NASDAQ filings: [kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset).

**Citations:**

1. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589-609.
2. Shumway, T. (2001). Forecasting Bankruptcy More Accurately: A Simple Hazard Model. *The Journal of Business*, 74(1), 101-124.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.
5. Lombardo, G. et al. (2022). Machine Learning for Bankruptcy Prediction in the American Stock Market. *Future Internet*, 14(8), 244.
6. Administrative Office of the U.S. Courts. (2026). Bankruptcy Filings Rise 11 Percent. uscourts.gov.
7. Hayes, A. (n.d.). Understanding the Altman Z-Score Formula and Its Interpretation. Investopedia.

## Algorithm

**Model type:** a calibrated XGBoost model, a gradient-boosted ensemble of decision trees, used as a binary classifier.

**Inputs:** the engineered feature set built from the 18 raw financial statement fields per firm-year: the Altman Z-Score components, profitability, leverage, and liquidity ratios, year-over-year and 2-year trend features, structural flags like consecutive years of negative net income, and an unsupervised industry cluster label.

**Output:** a calibrated probability that the company files for bankruptcy within the next year, bucketed into Lower, Elevated, or High risk bands.

**Why XGBoost:** we benchmarked it head to head against Logistic Regression, a Decision Tree, Random Forest, XGBoost with SMOTE resampling, and the classical Altman Z formula (see Model Evaluation below), and it won on PR-AUC and F1 at this roughly 1% base rate, while staying tree-based enough to support SHAP explanations for every individual prediction. That combination of the best test-set ranking performance and interpretability is why it's the model deployed in the app, rather than a less transparent model that might trade away the SHAP story.

## Model Card

**Data:** 78,682 US public company firm-years (1999-2018) from the Kaggle American Bankruptcy dataset of NYSE and NASDAQ firms, with 18 raw financial statement fields per firm-year.

**Target definition:** one-year bankruptcy filing. The raw data marks every historical row of a firm that eventually failed as `"failed"`, so we corrected the label: only a firm's final observed year counts as a positive (bankrupt next year), and all prior years are relabeled `alive`. This leaves roughly 609 true positives out of 78,682 rows, about a 0.77% base rate.

**Features:** the Altman Z-Score and its five components, profitability ratios (ROA, EBITDA/TA, margins), leverage ratios (liabilities/TA, long-term debt/TA), liquidity ratios (current ratio, inventory turnover, days sales outstanding), year-over-year and 2-year slope trends masked across year gaps, consecutive years of negative net income, years of firm history, and an unsupervised industry cluster label from K-Means fit on training rows only.

**Model:** XGBoost, tuned by time-series cross-validation with firm-level leakage prevention, so any firm appearing in a fold's validation window is fully excluded from that fold's training rows. The tuning-stage model trains on 1999-2011 data only.

**Calibration:** Platt scaling (sigmoid) fit on the 2012-2014 validation set, wrapping the tuning-stage model rather than a train-plus-validation refit, so the probabilities are calibrated on data the model never trained on. This is the version deployed in the app. The decision threshold is the F1-optimal point on validation, in calibrated-probability units.

**Test metrics (2015-2018):** see the Model Performance tab in the app, or `models/metrics.json`, for the full comparison across Logistic Regression, the Altman Z baseline, XGBoost (primary), XGBoost with SMOTE, Decision Tree, Random Forest, and the deployed calibrated model. The test set has only 119 true positives, so these numbers carry real uncertainty and a handful of different outcomes would move them noticeably.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Altman Z (1968 baseline) | 0.794 | 0.025 | 0.041 | 0.021 | 0.882 |
| Logistic Regression | 0.769 | 0.027 | 0.066 | 0.036 | 0.361 |
| Decision Tree (depth=3) | 0.883 | 0.058 | 0.124 | 0.068 | 0.723 |
| Random Forest | 0.888 | 0.116 | 0.160 | 0.105 | 0.345 |
| XGBoost (SMOTE) | 0.844 | 0.061 | 0.134 | 0.120 | 0.151 |
| XGBoost (primary) | 0.905 | 0.200 | 0.255 | 0.181 | 0.429 |
| **XGBoost (calibrated, deployed)** | 0.902 | 0.189 | 0.223 | 0.169 | 0.328 |

What this reveals: ROC-AUC is inflated by the huge majority class at this base rate, so PR-AUC and F1 are the metrics that actually matter here. XGBoost roughly doubles Logistic Regression's PR-AUC and beats Altman's F1 by about 5x, meaning it ranks true bankruptcies far higher among its top alerts. Altman's headline recall of 0.88 looks strong, but it comes with precision of only 0.02, meaning it flags almost every company and can't actually prioritize review on its own. The deployed, calibrated model trades a little of the primary model's raw F1 for trustworthy probabilities: calibration drops the Brier score from 0.079 raw to 0.0089 calibrated, which is why this version, not the higher-F1 primary model, is what's shipped in the app.

**Limitations:** trained only on public NYSE and NASDAQ US firms. It predicts a filing within one year of the reported financials, not a multi-year risk estimate. The alive class carries survivorship bias, since firms that were acquired, delisted, or simply stopped reporting for non-bankruptcy reasons are not separated from firms that stayed healthy. This is a screening tool meant to prioritize further review, not an automated decision system.

## Impact and Bias

**Positive effects:** a working early-warning signal helps creditors, lenders, investors, and employees plan ahead instead of being surprised by a filing, and it offers a free, explainable alternative to opaque commercial credit-risk scores. Because every prediction comes with a SHAP breakdown, a user can also check whether the reasoning behind a score makes sense rather than trusting a black box.

**Negative effects:** at roughly 17-18% precision, most companies the model flags are false alarms. Careless or public use of a risk score could unfairly damage a healthy company's reputation, financing terms, or stock price. On the other side, false negatives could give lenders or employees false reassurance about a firm that does go on to fail.

**Bias:** the model is trained only on public NYSE and NASDAQ filers from 1999-2018, so it inherits survivorship bias: firms that were quietly delisted or acquired for reasons unrelated to failing are not separated from firms that stayed genuinely healthy, and it has never seen data past 2018, including any COVID-era shock. The unsupervised industry cluster feature could also encode sector-level bias, for example penalizing capital-intensive industries like utilities or manufacturing that naturally run worse liquidity ratios than services firms, even when both are healthy for their own sector.

**Mitigation:** `src/validate.py`'s `check_drift` flags when new input diverges from the training reference stats, and this README positions the tool as a screening and triage aid for human review, not an automated accept or reject decision. Those two guardrails are what keep the biases above from silently compounding once the model is in use.

## Drift Policy

New data has to pass `src/validate.py` (`validate_for_training`, plus `check_drift` against the saved training reference stats) before it's used for anything. Retraining is a manual `python -m src.train` run, with no automated pipeline. The app always scores against whatever is currently committed in `models/`.

## What We Learned

A few things stuck with us by the end of this project.

Accuracy is a trap when the thing you care about almost never happens. Only about 0.77% of the company-years in our data end in bankruptcy, so a model that predicts "everyone survives" scores over 99% and catches nothing. We leaned on PR-AUC and recall instead, which actually reward finding the rare failures.

Cleaning the labels mattered more than any fancy model. The raw data marked every year of a doomed company as "failed," even years when it was doing fine. Fixing that so only the final year counts as bankrupt changed the whole problem, and skipping it would have quietly taught the model nonsense.

How you split the data is a decision, not a detail. Because the same firm shows up across many years, a random split would let the model peek at a company's future. Splitting by time instead, with the 2008 crisis sitting only in the training years, forced the model to prove it still works in calmer periods.

Modern ML really does beat the old formula, but not by magic. Our XGBoost model outscored the 1968 Altman Z-Score on the test set, and SHAP showed it weighs the same financial signals quite differently than Altman did. Still, it flags a lot of false alarms, so we treat it as a screening tool that points humans toward companies worth a closer look, not a final verdict.

## Next Steps

- Retrain and re-validate on 2019 and later data to see how the model handles a post-2018, COVID-era shock it has never encountered.
- Reduce the survivorship-bias scope by extending beyond NYSE and NASDAQ-listed firms, if a suitable data source is found.
- Replace the unsupervised K-Means industry cluster with an explicit, auditable industry code such as SIC or GICS, to make the sector-level bias discussed above easier to inspect and correct.
- Explore cost-sensitive or user-tunable thresholds, since different audiences, like a lender versus a jobseeker, tolerate different false-positive rates, rather than shipping one fixed threshold for everyone.
- Extend the Streamlit app itself, for example with portfolio-level batch monitoring or alerting, building on the CSV batch screening and SHAP explanation work already shipped.
