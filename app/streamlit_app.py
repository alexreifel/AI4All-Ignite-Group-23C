"""Streamlit app for screening companies for one-year bankruptcy risk.

Loads artifacts produced by `python -m src.train` and scores user input
through src.predict.predict_one. Never fits or refits a model.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import predict  # noqa: E402
from src.constants import RAW_DOLLAR_COLS  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
MANUAL_LABEL = "-- Enter manually --"

FIELD_LABELS = {
    "current_assets": "Current assets",
    "cogs": "Cost of goods sold",
    "dep_amort": "Depreciation and amortization",
    "ebitda": "EBITDA",
    "inventory": "Inventory",
    "net_income": "Net income",
    "receivables": "Receivables",
    "market_value": "Market value of equity",
    "net_sales": "Net sales",
    "total_assets": "Total assets",
    "lt_debt": "Long-term debt",
    "ebit": "EBIT",
    "gross_profit": "Gross profit",
    "curr_liabilities": "Current liabilities",
    "retained_earnings": "Retained earnings",
    "total_revenue": "Total revenue",
    "total_liabilities": "Total liabilities",
    "total_opex": "Total operating expenses",
}
BALANCE_SHEET_FIELDS = [
    "current_assets", "inventory", "receivables", "total_assets",
    "curr_liabilities", "lt_debt", "total_liabilities", "retained_earnings", "market_value",
]
INCOME_FIELDS = [
    "net_sales", "total_revenue", "cogs", "gross_profit",
    "ebitda", "ebit", "dep_amort", "net_income", "total_opex",
]

st.set_page_config(page_title="Bankruptcy Risk Screener", layout="wide")


@st.cache_data
def load_metrics():
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def load_example_firms():
    return json.loads((MODELS_DIR / "example_firms.json").read_text())


@st.cache_resource
def load_test_scores():
    return dict(np.load(MODELS_DIR / "test_scores.npz"))


def apply_example():
    choice = st.session_state["example_choice"]
    if choice == MANUAL_LABEL:
        return
    examples = load_example_firms()
    ex = next(e for e in examples if _example_label(e) == choice)
    for field in RAW_DOLLAR_COLS:
        st.session_state[f"cur_{field}"] = float(ex["financials"].get(field, 0.0))
    if ex.get("prev_year"):
        st.session_state["use_prev"] = True
        for field in RAW_DOLLAR_COLS:
            st.session_state[f"prev_{field}"] = float(ex["prev_year"].get(field, 0.0))
    else:
        st.session_state["use_prev"] = False


def _example_label(example):
    return f"{example['company_name']} ({example['year']}) - {example['scenario']}"


st.title("Bankruptcy Risk Screener")
st.caption(
    "This tool screens a company's annual financial statement data to estimate "
    "the likelihood it files for bankruptcy within the next year."
)

tab_screen, tab_perf, tab_about = st.tabs(
    ["Screen a Company", "Model Performance", "About and Limitations"]
)

with tab_screen:
    examples = load_example_firms()
    example_labels = [MANUAL_LABEL] + [_example_label(e) for e in examples]
    st.selectbox(
        "Load an example test-set firm, or enter figures manually below",
        example_labels, key="example_choice", on_change=apply_example,
    )

    st.markdown("Enter the company's most recent annual financial statement figures.")
    col1, col2 = st.columns(2)
    financials = {}
    with col1:
        st.markdown("**Balance sheet**")
        for field in BALANCE_SHEET_FIELDS:
            financials[field] = st.number_input(
                FIELD_LABELS[field], key=f"cur_{field}", format="%.3f"
            )
    with col2:
        st.markdown("**Income statement**")
        for field in INCOME_FIELDS:
            financials[field] = st.number_input(
                FIELD_LABELS[field], key=f"cur_{field}", format="%.3f"
            )

    use_prev = st.checkbox("Include previous year's financials (enables trend features)", key="use_prev")
    prev_year = None
    if use_prev:
        with st.expander("Previous Year Financials", expanded=True):
            prev_year = {}
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                for field in BALANCE_SHEET_FIELDS:
                    prev_year[field] = st.number_input(
                        f"Prior year - {FIELD_LABELS[field]}", key=f"prev_{field}", format="%.3f"
                    )
            with pcol2:
                for field in INCOME_FIELDS:
                    prev_year[field] = st.number_input(
                        f"Prior year - {FIELD_LABELS[field]}", key=f"prev_{field}", format="%.3f"
                    )

    if st.button("Screen this company", type="primary"):
        st.session_state["last_result"] = predict.predict_one(financials, prev_year)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        band_colors = {"Lower Risk": "green", "Elevated": "orange", "High": "red"}

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Calibrated bankruptcy probability", f"{result['probability']:.1%}")
            st.markdown(f"Risk band: :{band_colors[result['risk_band']]}[**{result['risk_band']}**]")
            st.write(result["plain_language"])
        with c2:
            st.metric("Altman Z-Score", f"{result['altman_z']:.2f}")
            st.write(f"Zone: **{result['altman_zone']}**")
            st.caption("Zones: distress below 1.81, gray zone 1.81-2.99, safe above 2.99.")

        st.subheader("Why this prediction")
        exp = shap.Explanation(
            values=np.array(list(result["shap_values"].values())),
            base_values=result["base_value"],
            data=np.array(list(result["feature_values"].values())),
            feature_names=list(result["shap_values"].keys()),
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(exp, max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Shows each feature's contribution to the underlying model's score, in the "
            "model's own units, starting from the average score across the training data."
        )

with tab_perf:
    st.subheader("Model Performance")
    metrics = load_metrics()
    results_df = pd.DataFrame(metrics["results"]).set_index("model")
    results_df = results_df.rename(columns={
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
        "precision": "Precision", "recall": "Recall", "threshold": "Threshold",
    })
    st.dataframe(results_df.round(4))
    st.caption(
        f"Deployed model: {metrics['deployed_model']}. Test set (2015-2018): "
        f"{metrics['test_set']['n_rows']:,} rows, {metrics['test_set']['n_positives']} "
        f"positives ({metrics['test_set']['positive_rate']:.2%})."
    )

    scores = load_test_scores()
    y_te = scores["y_te"]

    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        fig, ax = plt.subplots()
        for name, key in metrics["test_score_keys"].items():
            sc = scores[key]
            p, r, _ = precision_recall_curve(y_te, sc)
            ax.plot(r, p, drawstyle="steps-post",
                    label=f"{name} ({average_precision_score(y_te, sc):.3f})")
        ax.axhline(y_te.mean(), color="gray", ls=":", label="chance")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curves, test set")
        ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    with curve_col2:
        fig, ax = plt.subplots()
        for name, key in metrics["test_score_keys"].items():
            sc = scores[key]
            fpr, tpr, _ = roc_curve(y_te, sc)
            ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, sc):.3f})")
        ax.plot([0, 1], [0, 1], color="gray", ls=":", label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves, test set")
        ax.legend(fontsize=7, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Interactive threshold, deployed model")
    deployed_key = metrics["test_score_keys"][metrics["deployed_model"]]
    deployed_scores = scores[deployed_key]
    thr = st.slider(
        "Decision threshold (calibrated probability)",
        0.0, 1.0, float(metrics["calibrated_threshold"]), 0.01,
    )
    pred = (deployed_scores >= thr).astype(int)
    precision = precision_score(y_te, pred, zero_division=0)
    recall = recall_score(y_te, pred, zero_division=0)
    flagged = int(pred.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision:.3f}")
    m2.metric("Recall", f"{recall:.3f}")
    m3.metric("Firms flagged", f"{flagged:,}")

with tab_about:
    st.subheader("Model card")
    metrics = load_metrics()
    deployed_row = next(r for r in metrics["results"] if r["model"] == metrics["deployed_model"])

    st.markdown(f"""
**Data.** {metrics['n_train_rows'] + metrics['n_val_rows'] + metrics['n_test_rows']:,} US public
company firm-years (1999-2018), from the Kaggle American Bankruptcy dataset of NYSE/NASDAQ-listed firms.

**Target.** One-year bankruptcy filing. The raw data marks every historical row of a firm that
eventually failed as "failed", so the label was corrected: only a firm's final observed year is
counted as a positive (bankrupt next year), and all prior years are relabeled alive.
{metrics['n_relabeled_rows']:,} rows were relabeled this way.

**Features.** Altman Z-Score and its five components, profitability/leverage/liquidity ratios,
year-over-year and 2-year trend features masked across year gaps, consecutive years of negative
net income, years of firm history, and an unsupervised industry cluster label fit on training
data only.

**Model.** XGBoost, hyperparameters tuned by time-series cross-validation with firm-level leakage
prevention. Trained on 1999-2011 data only, so its probabilities can be calibrated on clean,
unseen validation data.

**Calibration.** Platt scaling (sigmoid), fit on the 2012-2014 validation set. This is the model
deployed here.

**Test metrics (2015-2018, {metrics['test_set']['n_positives']} positives out of
{metrics['test_set']['n_rows']:,} rows).** ROC-AUC {deployed_row['roc_auc']:.3f}, PR-AUC
{deployed_row['pr_auc']:.3f}, F1 {deployed_row['f1']:.3f}, precision {deployed_row['precision']:.3f},
recall {deployed_row['recall']:.3f} at the frozen threshold. With only
{metrics['test_set']['n_positives']} true positives in the test set, these metrics carry wide
uncertainty; a handful of different outcomes would move them noticeably.
""")

    st.subheader("Altman Z-Score bankruptcy zones (test set)")
    zone_df = pd.DataFrame(metrics["altman_zone_table"]).T
    zone_df.columns = ["Bankruptcy rate", "Firms"]
    st.dataframe(zone_df)

    st.subheader("Which Altman input matters most: 1968 vs today")
    imp = metrics["importance_comparison"]
    today_pct, altman_pct, labels = imp["today_pct"], imp["altman_pct"], imp["labels"]
    order = sorted(today_pct, key=lambda k: today_pct[k])
    y = np.arange(len(order))
    h = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(y - h / 2, [altman_pct[k] for k in order], height=h,
            color="#B0B0B0", label="1968 (Altman weights)")
    ax.barh(y + h / 2, [today_pct[k] for k in order], height=h,
            color="#4C72B0", label="Today (XGBoost mean SHAP)")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order])
    ax.set_xlabel("Share of importance among the 5 Altman variables (%)")
    ax.set_title("Which Altman input matters most: 1968 vs. today")
    ax.legend(loc="lower right", frameon=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Limitations")
    st.warning(f"""
- Trained only on public NYSE/NASDAQ-listed US firms; results may not generalize to private
  companies, unlisted firms, or non-US markets.
- Predicts filing within one year of the reported financials, not a multi-year risk estimate.
- The alive class carries survivorship bias: firms that were acquired, delisted, or stopped
  reporting for reasons other than bankruptcy are not distinguished from firms that stayed healthy.
- This is a screening tool meant to prioritize further review, not an automated decision system.
- Precision-recall tradeoff at the frozen threshold: precision {deployed_row['precision']:.1%},
  recall {deployed_row['recall']:.1%} on {metrics['test_set']['n_positives']} test-set positives.
  Raising the threshold trades recall for precision and vice versa; see the Model Performance tab.
""")
