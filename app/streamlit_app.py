"""Streamlit app for screening companies for one-year bankruptcy risk.

Loads artifacts produced by `python -m src.train` and scores user input
through src.predict.predict_one / predict_history. Never fits or refits a model.
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
from src.constants import FEATURE_LABELS, RAW_DOLLAR_COLS  # noqa: E402

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

RISK_CHIP_STYLES = {
    "Lower Risk": ("#EAF7EE", "#1E7E34", "#BFE6C8"),
    "Elevated": ("#FFF6E5", "#9A5B00", "#FCE1A8"),
    "High": ("#FDECEC", "#B3261E", "#F5C2C0"),
}

st.set_page_config(page_title="Bankruptcy Risk Screener", page_icon="📊", layout="wide")


@st.cache_data
def load_metrics():
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def load_example_firms():
    return json.loads((MODELS_DIR / "example_firms.json").read_text())


@st.cache_resource
def load_test_scores():
    return dict(np.load(MODELS_DIR / "test_scores.npz"))


def _example_label(example):
    return f"{example['company_name']} ({example['year']}) - {example['scenario']}"


def _year_columns(n_years):
    current_year = pd.Timestamp.now().year
    return [str(y) for y in range(current_year - int(n_years) + 1, current_year + 1)]


def _blank_grid(fields, year_cols):
    return pd.DataFrame(0.0, index=[FIELD_LABELS[f] for f in fields], columns=year_cols)


def _apply_example_to_history():
    choice = st.session_state["hist_example_choice"]
    if choice == MANUAL_LABEL:
        st.session_state["hist_base_bs"] = None
        st.session_state["hist_base_inc"] = None
        return

    examples = load_example_firms()
    ex = next(e for e in examples if _example_label(e) == choice)
    year_cols = _year_columns(st.session_state["hist_num_years"])
    bs = _blank_grid(BALANCE_SHEET_FIELDS, year_cols)
    inc = _blank_grid(INCOME_FIELDS, year_cols)

    for field in BALANCE_SHEET_FIELDS:
        bs.loc[FIELD_LABELS[field], year_cols[-1]] = float(ex["financials"].get(field, 0.0))
    for field in INCOME_FIELDS:
        inc.loc[FIELD_LABELS[field], year_cols[-1]] = float(ex["financials"].get(field, 0.0))

    if ex.get("prev_year") and len(year_cols) >= 2:
        for field in BALANCE_SHEET_FIELDS:
            bs.loc[FIELD_LABELS[field], year_cols[-2]] = float(ex["prev_year"].get(field, 0.0))
        for field in INCOME_FIELDS:
            inc.loc[FIELD_LABELS[field], year_cols[-2]] = float(ex["prev_year"].get(field, 0.0))

    st.session_state["hist_base_bs"] = bs
    st.session_state["hist_base_inc"] = inc
    st.session_state["hist_load_version"] = st.session_state.get("hist_load_version", 0) + 1


def render_badges(labels):
    chips = "".join(
        '<span style="display:inline-block;background:#EEF2FF;color:#2563EB;'
        "border:1px solid #C7D2FE;border-radius:999px;padding:4px 12px;"
        f'margin:0 6px 6px 0;font-size:0.8rem;font-weight:600;">{label}</span>'
        for label in labels
    )
    st.markdown(f'<div style="margin-top:-6px;margin-bottom:14px;">{chips}</div>', unsafe_allow_html=True)


def _risk_chip_html(risk_band):
    bg, fg, border = RISK_CHIP_STYLES[risk_band]
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'border:1px solid {border};border-radius:999px;padding:4px 14px;'
        f'font-size:0.95rem;font-weight:700;">{risk_band}</span>'
    )


def render_result_panel(probability, risk_band, altman_z, altman_zone, plain_language=None):
    if plain_language is None:
        plain_language = (
            f"About {round(probability * 100)} of 100 firms with this profile "
            f"filed for bankruptcy within a year."
        )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Calibrated bankruptcy probability", f"{probability:.1%}")
            st.markdown(_risk_chip_html(risk_band), unsafe_allow_html=True)
            st.caption(plain_language)
        with c2:
            st.metric("Altman Z-Score", f"{altman_z:.2f}")
            st.write(f"Zone: **{altman_zone}**")
            st.caption("Zones: distress below 1.81, gray zone 1.81-2.99, safe above 2.99.")


def render_shap_waterfall(shap_values, base_value, feature_values, feature_labels=None):
    with st.container(border=True):
        st.subheader("Why this prediction")
        feature_names = list(shap_values.keys())
        if feature_labels:
            feature_names = [feature_labels.get(f, f) for f in feature_names]
        exp = shap.Explanation(
            values=np.array(list(shap_values.values())),
            base_values=base_value,
            data=np.array(list(feature_values.values())),
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(exp, max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Shows each feature's contribution to the underlying model's score, in the "
            "model's own units, starting from the average score across the training data."
        )


def _risk_row_style(row):
    bg = RISK_CHIP_STYLES.get(row["risk_band"], (None,))[0]
    return [f"background-color: {bg}" if bg else ""] * len(row)


st.title("Bankruptcy Risk Screener")
st.caption(
    "This tool screens a company's annual financial statement data to estimate "
    "the likelihood it files for bankruptcy within the next year."
)
render_badges([
    "XGBoost · Calibrated", "SHAP Explainable",
    "1999–2018 NYSE/NASDAQ firm-years", "Time-Series Validated",
])

tab_screen, tab_perf, tab_about = st.tabs(
    ["🔎 Screen a Company", "📈 Model Performance", "📄 About and Limitations"]
)

with tab_screen:
    input_mode = st.radio(
        "Input method",
        ["CSV Batch Upload", "Manual Entry"],
        horizontal=True,
        key="screen_input_mode",
    )

    if input_mode == "CSV Batch Upload":
        st.markdown(
            "Upload a CSV with `company_name`, `year`, and the 18 raw financial fields for one or "
            "more companies (readable field names or the dataset's `X1`-`X18` names are both accepted)."
        )
        csv_template = pd.DataFrame(columns=["company_name", "year", *RAW_DOLLAR_COLS])
        st.download_button(
            "Download CSV template",
            csv_template.to_csv(index=False).encode("utf-8"),
            file_name="bankruptcy_screening_template.csv",
            mime="text/csv",
            key="csv_template_download",
        )
        uploaded = st.file_uploader("Choose a CSV file", type="csv", key="csv_upload")
        csv_latest_only = st.checkbox(
            "Return only the latest year for each company", value=True, key="csv_latest_only"
        )
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error("The CSV could not be read.")
                st.exception(exc)
            else:
                st.write("Preview")
                st.dataframe(uploaded_df.head(20), width="stretch", hide_index=True)
                st.session_state["csv_uploaded_df"] = uploaded_df
                if st.button("Run batch predictions", type="primary", key="csv_submit"):
                    try:
                        st.session_state["csv_results"] = predict.predict_history(
                            uploaded_df, latest_only=csv_latest_only
                        )
                    except ValueError as exc:
                        st.session_state.pop("csv_results", None)
                        st.warning(str(exc))

        if "csv_results" in st.session_state:
            csv_results = st.session_state["csv_results"]
            st.divider()
            elevated_or_high = int(csv_results["risk_band"].isin(["Elevated", "High"]).sum())
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Companies screened", f"{len(csv_results):,}")
            with k2:
                st.metric("Elevated / High risk", f"{elevated_or_high:,}")
            with k3:
                st.metric("Average probability", f"{csv_results['probability'].mean():.1%}")

            display_results = csv_results.drop(columns=["threshold"]).copy()
            display_results["probability"] = display_results["probability"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                display_results.style.apply(_risk_row_style, axis=1),
                width="stretch", hide_index=True,
            )
            st.download_button(
                "Download prediction results",
                csv_results.to_csv(index=False).encode("utf-8"),
                file_name="bankruptcy_predictions.csv",
                mime="text/csv",
                key="csv_results_download",
            )

            st.divider()
            st.subheader("Explain a company's score")
            company_names = sorted(csv_results["company_name"].unique())
            selected_company = st.selectbox(
                "Choose a company from the batch results",
                company_names, key="csv_explain_company",
            )
            if st.button("Explain this company", key="csv_explain_submit"):
                try:
                    st.session_state["csv_explanation"] = predict.explain_history_row(
                        st.session_state["csv_uploaded_df"], company_name=selected_company,
                    )
                except ValueError as exc:
                    st.session_state.pop("csv_explanation", None)
                    st.warning(str(exc))

            if "csv_explanation" in st.session_state:
                csv_explanation = st.session_state["csv_explanation"]
                if csv_explanation["company_name"] == selected_company:
                    with st.expander(
                        f"Why: {csv_explanation['company_name']} ({csv_explanation['year']})",
                        expanded=True,
                    ):
                        render_shap_waterfall(
                            csv_explanation["shap_values"], csv_explanation["base_value"],
                            csv_explanation["feature_values"], feature_labels=FEATURE_LABELS,
                        )
                else:
                    st.caption('Selection changed — click "Explain this company" again to refresh.')

    else:
        st.markdown(
            "Enter one company's financial history, most recent year last. Two or more years let "
            "trend features (including the 2-year EBITDA slope) compute exactly as they do in training."
        )
        hist_company_name = st.text_input("Company name", value="Example Company", key="hist_company_name")
        number_of_years = st.number_input(
            "Number of years", min_value=1, max_value=10, value=2, step=1, key="hist_num_years"
        )
        examples = load_example_firms()
        example_labels = [MANUAL_LABEL] + [_example_label(e) for e in examples]
        st.selectbox(
            "Load an example test-set firm, or enter figures manually below",
            example_labels, key="hist_example_choice", on_change=_apply_example_to_history,
        )

        year_cols = _year_columns(number_of_years)
        base_bs = st.session_state.get("hist_base_bs")
        base_inc = st.session_state.get("hist_base_inc")
        if base_bs is None or list(base_bs.columns) != year_cols:
            base_bs = _blank_grid(BALANCE_SHEET_FIELDS, year_cols)
        if base_inc is None or list(base_inc.columns) != year_cols:
            base_inc = _blank_grid(INCOME_FIELDS, year_cols)
        grid_key_suffix = f"{int(number_of_years)}_{st.session_state.get('hist_load_version', 0)}"

        year_col_config = {
            col: st.column_config.NumberColumn(col, format="%.3f") for col in year_cols
        }
        with st.container(border=True):
            st.markdown("**Balance sheet** — rows are line items, columns are years")
            edited_bs = st.data_editor(
                base_bs, width="stretch", column_config=year_col_config,
                key=f"hist_bs_editor_{grid_key_suffix}",
            )
        with st.container(border=True):
            st.markdown("**Income statement** — rows are line items, columns are years")
            edited_inc = st.data_editor(
                base_inc, width="stretch", column_config=year_col_config,
                key=f"hist_inc_editor_{grid_key_suffix}",
            )

        if st.button("Screen this company", type="primary", key="hist_submit"):
            combined = pd.concat([edited_bs, edited_inc])
            history_rows = []
            for year_col in year_cols:
                row = {"company_name": hist_company_name.strip() or "Example Company", "year": int(year_col)}
                for field, label in FIELD_LABELS.items():
                    row[field] = float(combined.loc[label, year_col])
                history_rows.append(row)
            history = pd.DataFrame(history_rows)
            try:
                hist_results = predict.predict_history(history, latest_only=True)
                hist_explanation = predict.explain_history_row(history)
            except ValueError as exc:
                st.session_state.pop("hist_result", None)
                st.warning(str(exc))
            else:
                st.session_state["hist_result"] = (hist_results, hist_explanation)

        if "hist_result" in st.session_state:
            hist_results, hist_explanation = st.session_state["hist_result"]
            latest_row = hist_results.iloc[0]
            render_result_panel(
                latest_row["probability"], latest_row["risk_band"],
                latest_row["altman_z"], latest_row["altman_zone"],
            )
            render_shap_waterfall(
                hist_explanation["shap_values"], hist_explanation["base_value"],
                hist_explanation["feature_values"], feature_labels=FEATURE_LABELS,
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
