"""Loads saved artifacts and scores new input. Never recomputes a
winsorization cap, an imputation median, a cluster fit, or a threshold.
Everything here is either a fixed formula (safe_div ratios, Altman Z) or a
lookup against something train.py already fit and saved to models/.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src import features
from src.constants import ALTMAN_HIGH, ALTMAN_LOW, CLUSTER_COLS, COLMAP, FEATURES_ALL, RAW_DOLLAR_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


@lru_cache(maxsize=4)
def load_artifacts(models_dir=str(DEFAULT_MODELS_DIR)):
    models_dir = Path(models_dir)

    calibrated_model = joblib.load(models_dir / "model_calibrated.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    kmeans = joblib.load(models_dir / "kmeans.joblib")

    caps = json.loads((models_dir / "caps.json").read_text())
    medians = json.loads((models_dir / "medians.json").read_text())
    ta_caps = tuple(json.loads((models_dir / "ta_caps.json").read_text()))
    feature_list = json.loads((models_dir / "feature_list.json").read_text())
    threshold = json.loads((models_dir / "threshold.json").read_text())["calibrated_threshold"]
    reference_stats = json.loads((models_dir / "reference_stats.json").read_text())

    # Recover the exact tuning-stage XGBoost that produces the calibrated
    # probability, so the SHAP explanation is consistent with what's shown.
    underlying_xgb = calibrated_model.calibrated_classifiers_[0].estimator.estimator
    explainer = shap.TreeExplainer(underlying_xgb)

    return {
        "calibrated_model": calibrated_model,
        "scaler": scaler,
        "kmeans": kmeans,
        "caps": caps,
        "medians": medians,
        "ta_caps": ta_caps,
        "feature_list": feature_list,
        "threshold": threshold,
        "reference_stats": reference_stats,
        "explainer": explainer,
    }


def _risk_band(probability, threshold):
    half = threshold / 2
    if probability < half:
        return "Lower Risk"
    if probability < threshold:
        return "Elevated"
    return "High"


def _build_synthetic_history(financials, prev_year):
    rows = []
    if prev_year is not None:
        prev_row = {c: float(prev_year[c]) for c in RAW_DOLLAR_COLS}
        prev_row["company_name"] = "synthetic_firm"
        prev_row["year"] = 0
        rows.append(prev_row)

    current_row = {c: float(financials[c]) for c in RAW_DOLLAR_COLS}
    current_row["company_name"] = "synthetic_firm"
    current_row["year"] = 1 if prev_year is not None else 0
    rows.append(current_row)

    return pd.DataFrame(rows)


def predict_one(financials: dict, prev_year: dict | None = None, models_dir=str(DEFAULT_MODELS_DIR)) -> dict:
    """Score one company. financials/prev_year keys are the readable raw
    dollar column names (current_assets, net_income, ...). If prev_year is
    omitted, trend features are 0 and years_of_history is 1, matching the
    training-time first-observation rule.
    """
    artifacts = load_artifacts(models_dir)

    df = _build_synthetic_history(financials, prev_year)
    winsor_artifacts = {
        "caps": artifacts["caps"],
        "medians": artifacts["medians"],
        "ta_caps": artifacts["ta_caps"],
    }
    df = features.engineer_features(df, fit=False, artifacts=winsor_artifacts)
    df = features.assign_cluster(df, artifacts["scaler"], artifacts["kmeans"], CLUSTER_COLS)

    current = df.iloc[[-1]]
    X = current[FEATURES_ALL]

    probability = float(artifacts["calibrated_model"].predict_proba(X)[:, 1][0])
    threshold = artifacts["threshold"]
    risk_band = _risk_band(probability, threshold)

    altman_z = float(current["altman_z"].iloc[0])
    if altman_z < ALTMAN_LOW:
        altman_zone = "distress (<1.81)"
    elif altman_z < ALTMAN_HIGH:
        altman_zone = "gray (1.81-2.99)"
    else:
        altman_zone = "safe (>2.99)"

    sv = artifacts["explainer"](X)
    shap_values = {feat: float(val) for feat, val in zip(FEATURES_ALL, sv.values[0])}
    feature_values = {feat: float(val) for feat, val in zip(FEATURES_ALL, X.iloc[0].values)}
    base_value = float(np.asarray(sv.base_values).reshape(-1)[0])

    plain_language = (
        f"About {round(probability * 100)} of 100 firms with this profile "
        f"filed for bankruptcy within a year."
    )

    return {
        "probability": probability,
        "risk_band": risk_band,
        "altman_z": altman_z,
        "altman_zone": altman_zone,
        "shap_values": shap_values,
        "feature_values": feature_values,
        "base_value": base_value,
        "plain_language": plain_language,
        "threshold": threshold,
    }


def _altman_zone(altman_z: float) -> str:
    if altman_z < ALTMAN_LOW:
        return "distress (<1.81)"
    if altman_z < ALTMAN_HIGH:
        return "gray (1.81-2.99)"
    return "safe (>2.99)"


def _normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either the readable raw-dollar column names or the dataset's X1-X18 names."""
    df = df.rename(columns=COLMAP).copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a user-supplied multi-year, multi-company financial history into the
    shape features.engineer_features expects. Raises ValueError with a message
    fit for display in the app on any problem.
    """
    df = _normalize_history_columns(df)

    if "company_name" not in df.columns:
        df["company_name"] = "Uploaded company"
    df["company_name"] = df["company_name"].fillna("Unknown company").astype(str)

    required = {"year", *RAW_DOLLAR_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["year", *RAW_DOLLAR_COLS]
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isna().any().any():
        raise ValueError("Every year and financial field must be a number, with no missing values.")
    if (df["total_assets"] <= 0).any():
        raise ValueError("total_assets must be greater than zero in every row.")
    df["year"] = df["year"].astype(int)

    if df.duplicated(["company_name", "year"]).any():
        raise ValueError("Each company can have only one row per year.")

    return df.sort_values(["company_name", "year"]).reset_index(drop=True)


def _engineer_history(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    winsor_artifacts = {
        "caps": artifacts["caps"],
        "medians": artifacts["medians"],
        "ta_caps": artifacts["ta_caps"],
    }
    df = features.engineer_features(df, fit=False, artifacts=winsor_artifacts)
    df = features.assign_cluster(df, artifacts["scaler"], artifacts["kmeans"], CLUSTER_COLS)
    return df


def predict_history(
    history: pd.DataFrame, latest_only: bool = True, models_dir=str(DEFAULT_MODELS_DIR)
) -> pd.DataFrame:
    """Score a multi-year, multi-company financial history in one pass.

    Unlike predict_one (which can carry at most one prior year), trends here
    are computed the same way training does: features.build_trends groups by
    company_name and diffs across the actually-supplied years, so a 2-year
    EBITDA slope only appears when 3+ consecutive years are given.

    Returns one row per (company_name, year) with probability/risk_band/
    altman_z/altman_zone, or one row per company (its most recent year) when
    latest_only=True.
    """
    artifacts = load_artifacts(models_dir)
    df = _validate_history(history)
    df = _engineer_history(df, artifacts)

    X = df[FEATURES_ALL]
    probabilities = artifacts["calibrated_model"].predict_proba(X)[:, 1]
    threshold = artifacts["threshold"]

    result = df[["company_name", "year"]].copy()
    result["probability"] = probabilities
    result["risk_band"] = [_risk_band(p, threshold) for p in probabilities]
    result["altman_z"] = df["altman_z"].to_numpy()
    result["altman_zone"] = [_altman_zone(z) for z in result["altman_z"]]
    result["threshold"] = threshold

    if latest_only:
        latest_idx = result.groupby("company_name")["year"].idxmax()
        result = result.loc[latest_idx].sort_values("company_name").reset_index(drop=True)
    else:
        result = result.sort_values(["company_name", "year"]).reset_index(drop=True)

    return result


def explain_history_row(
    history: pd.DataFrame,
    company_name: str | None = None,
    year: int | None = None,
    models_dir=str(DEFAULT_MODELS_DIR),
) -> dict:
    """SHAP explanation for one row of a multi-year history, engineered
    alongside the rest of that company's supplied history so trends are
    correct. Defaults to the most recent year of the last company in the
    (sorted) history if company_name/year aren't given.
    """
    artifacts = load_artifacts(models_dir)
    df = _validate_history(history)
    df = _engineer_history(df, artifacts)

    if company_name is None:
        company_name = df["company_name"].iloc[-1]
    subset = df[df["company_name"] == company_name]
    if subset.empty:
        raise ValueError(f"No rows for company {company_name!r}")
    row = subset.loc[[subset["year"].idxmax()]] if year is None else subset[subset["year"] == year]
    if row.empty:
        raise ValueError(f"No row for company {company_name!r}, year {year}")

    X = row[FEATURES_ALL]
    sv = artifacts["explainer"](X)
    shap_values = {feat: float(val) for feat, val in zip(FEATURES_ALL, sv.values[0])}
    feature_values = {feat: float(val) for feat, val in zip(FEATURES_ALL, X.iloc[0].values)}
    base_value = float(np.asarray(sv.base_values).reshape(-1)[0])

    return {
        "company_name": company_name,
        "year": int(row["year"].iloc[0]),
        "shap_values": shap_values,
        "feature_values": feature_values,
        "base_value": base_value,
    }
