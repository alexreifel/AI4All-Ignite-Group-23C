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
from src.constants import ALTMAN_HIGH, ALTMAN_LOW, CLUSTER_COLS, FEATURES_ALL, RAW_DOLLAR_COLS

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
