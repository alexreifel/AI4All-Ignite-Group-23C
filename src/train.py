"""Reproduces the notebook's pipeline end to end and serializes deployment artifacts.

This mirrors the six staged notebooks in notebooks/ exactly: same hyperparameter
grids, same random states, same splits, same formulas. The Logistic Regression
step is written here as its own function (fit_logreg), the same way Decision
Tree and Random Forest are, using its own declared feature set (FEATURES_LR)
for both the CV grid and the final fit.

Run as: python -m src.train
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import features, validate
from src.constants import (
    ALTMAN_COEFS,
    ALTMAN_HIGH,
    ALTMAN_LOW,
    CLUSTER_COLS,
    COLMAP,
    EXPECTED_K_OPT,
    FEATURES_ALL,
    FEATURES_DT,
    FEATURES_LR,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "american_bankruptcy.csv"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def evaluate(name, y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def build_grouped_time_folds(df, boundaries, feature_cols):
    """Grouped time-series folds: any firm with a row in a fold's validation
    window is fully excluded from that fold's training rows, to prevent
    firm-level leakage.
    """
    folds = []
    for (tr_start, tr_end, va_start, va_end) in boundaries:
        val_window = df[df["year"].between(va_start, va_end)]
        val_firms = set(val_window["company_name"])
        train_window_raw = df[df["year"].between(tr_start, tr_end)]
        train_window_clean = train_window_raw[~train_window_raw["company_name"].isin(val_firms)]
        folds.append({
            "X_tr": train_window_clean[feature_cols], "y_tr": train_window_clean["target"],
            "X_va": val_window[feature_cols], "y_va": val_window["target"],
        })
    return folds


def load_and_prepare(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns=COLMAP)
    validate.validate_for_training(df)
    df = features.relabel_target(df)
    train_mask = df["year"] <= TRAIN_YEARS[1]
    df, winsor_artifacts = features.engineer_features(df, train_mask=train_mask, fit=True)
    return df, train_mask, winsor_artifacts


def fit_cluster(df, train_mask):
    scaler = StandardScaler().fit(df.loc[train_mask, CLUSTER_COLS])
    Xc_train = scaler.transform(df.loc[train_mask, CLUSTER_COLS])

    k_range = range(2, 11)
    sils = []
    for k in k_range:
        km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc_train)
        sils.append(silhouette_score(Xc_train, km_k.labels_, sample_size=5000, random_state=42))

    k_opt = list(k_range)[int(np.argmax(sils))]
    print(f"Silhouette-selected k = {k_opt} (silhouette={max(sils):.3f})")
    if k_opt != EXPECTED_K_OPT:
        raise ValueError(
            f"Silhouette search selected k={k_opt}, but FEATURES_ALL hard-codes "
            f"industry_cluster_1, which requires k={EXPECTED_K_OPT}. This is a "
            "real fidelity break with the notebook."
        )

    kmeans = KMeans(n_clusters=k_opt, n_init=10, random_state=42).fit(Xc_train)
    df = features.assign_cluster(df, scaler, kmeans, CLUSTER_COLS)
    return df, scaler, kmeans, k_opt


def fit_logreg(df, cv_folds_lr, X_tr_lr, y_tr, X_va_lr, y_va, X_te_lr, y_te):
    C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    lr_cv_results = []
    for C in C_grid:
        fold_scores = []
        for f in cv_folds_lr:
            scaler = StandardScaler().fit(f["X_tr"])
            Xf_tr_s, Xf_va_s = scaler.transform(f["X_tr"]), scaler.transform(f["X_va"])
            lr = LogisticRegression(max_iter=2000, C=C, class_weight="balanced", random_state=42)
            lr.fit(Xf_tr_s, f["y_tr"])
            fold_scores.append(average_precision_score(f["y_va"], lr.predict_proba(Xf_va_s)[:, 1]))
        lr_cv_results.append({"C": C, "pr_auc_mean": np.mean(fold_scores)})
        print(f"LR C={C}: mean PR-AUC={np.mean(fold_scores):.4f}")

    best_C = max(lr_cv_results, key=lambda r: r["pr_auc_mean"])["C"]
    print(f"LR best C: {best_C}")

    final_scaler = StandardScaler().fit(X_tr_lr)
    lr_final = LogisticRegression(max_iter=2000, C=best_C, class_weight="balanced", random_state=42)
    lr_final.fit(final_scaler.transform(X_tr_lr), y_tr)
    lr_thr = best_f1_threshold(y_va, lr_final.predict_proba(final_scaler.transform(X_va_lr))[:, 1])
    sc = lr_final.predict_proba(final_scaler.transform(X_te_lr))[:, 1]
    return evaluate("LogReg (balanced)", y_te, sc, lr_thr), sc


def fit_xgboost_primary(df, cv_folds, X_tr, y_tr, X_va, y_va, X_te, y_te):
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"scale_pos_weight from training data: {spw:.1f}")

    grid = [
        {"max_depth": d, "learning_rate": lr_, "subsample": ss,
         "colsample_bytree": cs, "min_child_weight": mcw}
        for d in (3, 4, 6)
        for lr_ in (0.05, 0.1)
        for ss, cs, mcw in [(0.8, 0.8, 5), (1.0, 1.0, 1)]
    ]

    cv_results = []
    for params in grid:
        fold_scores, fold_rounds = [], []
        for f in cv_folds:
            spw_f = (f["y_tr"] == 0).sum() / max((f["y_tr"] == 1).sum(), 1)
            m = XGBClassifier(n_estimators=2000, early_stopping_rounds=50,
                              scale_pos_weight=spw_f, eval_metric="aucpr",
                              tree_method="hist", random_state=42, **params)
            m.fit(f["X_tr"], f["y_tr"], eval_set=[(f["X_va"], f["y_va"])], verbose=False)
            fold_scores.append(average_precision_score(f["y_va"], m.predict_proba(f["X_va"])[:, 1]))
            fold_rounds.append(m.best_iteration + 1)
        cv_results.append({"params": params, "pr_auc_mean": np.mean(fold_scores),
                            "pr_auc_std": np.std(fold_scores), "n_rounds": int(np.median(fold_rounds))})

    best_cv = max(cv_results, key=lambda r: r["pr_auc_mean"])
    print(f"Best XGBoost CV mean PR-AUC {best_cv['pr_auc_mean']:.4f} "
          f"(+/-{best_cv['pr_auc_std']:.4f}) with {best_cv['params']}, {best_cv['n_rounds']} rounds")

    best = {"params": best_cv["params"], "n_rounds": best_cv["n_rounds"]}
    tuning_fit = XGBClassifier(n_estimators=best["n_rounds"], scale_pos_weight=spw,
                               eval_metric="aucpr", tree_method="hist",
                               random_state=42, **best["params"])
    tuning_fit.fit(X_tr, y_tr, verbose=False)
    best["model"] = tuning_fit
    best["pr_auc"] = average_precision_score(y_va, tuning_fit.predict_proba(X_va)[:, 1])

    xgb_thr = best_f1_threshold(y_va, tuning_fit.predict_proba(X_va)[:, 1])

    X_trva = pd.concat([X_tr, X_va])
    y_trva = pd.concat([y_tr, y_va])
    spw_full = (y_trva == 0).sum() / (y_trva == 1).sum()
    xgb = XGBClassifier(n_estimators=best["n_rounds"], scale_pos_weight=spw_full,
                        eval_metric="aucpr", tree_method="hist",
                        random_state=42, **best["params"])
    xgb.fit(X_trva, y_trva, verbose=False)
    sc = xgb.predict_proba(X_te)[:, 1]
    result = evaluate("XGBoost (primary)", y_te, sc, xgb_thr)
    return result, sc, best, xgb, X_trva, y_trva


def fit_xgboost_smote(best, X_tr, y_tr, X_va, y_va, X_te, y_te):
    X_tr_sm, y_tr_sm = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    print(f"Training set size before SMOTE: {len(y_tr):,} ({y_tr.mean():.3%} positive)")
    print(f"Training set size after SMOTE:  {len(y_tr_sm):,} ({y_tr_sm.mean():.3%} positive)")

    xgb_sm = XGBClassifier(n_estimators=best["n_rounds"], scale_pos_weight=1,
                           eval_metric="aucpr", tree_method="hist",
                           random_state=42, **best["params"])
    xgb_sm.fit(X_tr_sm, y_tr_sm, verbose=False)

    sm_thr = best_f1_threshold(y_va, xgb_sm.predict_proba(X_va)[:, 1])
    sc = xgb_sm.predict_proba(X_te)[:, 1]
    return evaluate("XGBoost (SMOTE)", y_te, sc, sm_thr), sc


def fit_decision_tree(X_tr_dt, y_tr, X_va_dt, y_va, X_te_dt, y_te):
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    dt.fit(X_tr_dt, y_tr)
    dt_thr = best_f1_threshold(y_va, dt.predict_proba(X_va_dt)[:, 1])
    sc = dt.predict_proba(X_te_dt)[:, 1]
    return evaluate("Decision Tree (depth=3)", y_te, sc, dt_thr), sc


def fit_random_forest(X_tr, y_tr, X_va, y_va, X_te, y_te):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_thr = best_f1_threshold(y_va, rf.predict_proba(X_va)[:, 1])
    sc = rf.predict_proba(X_te)[:, 1]
    return evaluate("Random Forest", y_te, sc, rf_thr), sc


def fit_calibration(tuning_model, X_va, y_va, X_te, y_te):
    raw_probs_te = tuning_model.predict_proba(X_te)[:, 1]
    calibrated = CalibratedClassifierCV(FrozenEstimator(tuning_model), method="sigmoid")
    calibrated.fit(X_va, y_va)
    calibrated_probs_va = calibrated.predict_proba(X_va)[:, 1]
    calibrated_probs_te = calibrated.predict_proba(X_te)[:, 1]

    # Threshold re-expressed in calibrated-probability space. Platt scaling
    # is monotonic, so this selects the same val rows as thresholding the
    # raw tuning-stage scores would, just in the units the app displays.
    calibrated_thr = best_f1_threshold(y_va, calibrated_probs_va)

    result = evaluate("XGBoost (calibrated, deployed)", y_te, calibrated_probs_te, calibrated_thr)

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_te, raw_probs_te, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_te, calibrated_probs_te, n_bins=10, strategy="quantile")

    calibration_info = {
        "brier_raw": float(brier_score_loss(y_te, raw_probs_te)),
        "brier_calibrated": float(brier_score_loss(y_te, calibrated_probs_te)),
        "raw": {"prob_pred": prob_pred_raw.tolist(), "prob_true": prob_true_raw.tolist()},
        "calibrated": {"prob_pred": prob_pred_cal.tolist(), "prob_true": prob_true_cal.tolist()},
    }
    return calibrated, calibrated_thr, calibrated_probs_te, result, calibration_info


def importance_comparison(xgb, X_te, X_trva, te):
    explainer = shap.TreeExplainer(xgb)
    sv = explainer(X_te)

    today_importance_all = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURES_ALL)
    today_five = today_importance_all.loc[list(ALTMAN_COEFS.keys())]

    train_means = X_trva[list(ALTMAN_COEFS.keys())].mean()
    altman_contribs = pd.DataFrame({
        name: ALTMAN_COEFS[name] * (te[name] - train_means[name])
        for name in ALTMAN_COEFS
    })
    altman_importance = altman_contribs.abs().mean().sort_values(ascending=False)

    today_pct = today_five / today_five.sum() * 100
    altman_pct = altman_importance / altman_importance.sum() * 100

    labels = {
        "altman_x1": "X1: Working capital / TA",
        "altman_x2": "X2: Retained earnings / TA",
        "altman_x3": "X3: EBIT / TA",
        "altman_x4": "X4: Market equity / liabilities",
        "altman_x5": "X5: Sales / TA",
    }
    return {
        "today_pct": today_pct.to_dict(),
        "altman_pct": altman_pct.to_dict(),
        "labels": labels,
    }


def sanitize_key(name):
    return (name.replace(" ", "_").replace("(", "").replace(")", "")
                .replace(",", "").replace("+", "plus"))


def build_example_firms(df, te, calibrated_probs_te, calibrated_thr):
    """Pick a handful of real test-set firms for the app's example dropdown."""
    from src.constants import RAW_DOLLAR_COLS

    te_meta = te[["company_name", "year", "target"]].reset_index(drop=True)
    te_meta["calibrated_probability"] = calibrated_probs_te

    def firm_financials(company_name, year):
        row = df[(df["company_name"] == company_name) & (df["year"] == year)]
        if row.empty:
            return None
        row = row.iloc[0]
        return {c: float(row[c]) for c in RAW_DOLLAR_COLS}

    def make_example(company_name, year, scenario):
        current = firm_financials(company_name, year)
        prev = firm_financials(company_name, year - 1)
        row = te_meta[(te_meta["company_name"] == company_name) & (te_meta["year"] == year)].iloc[0]
        return {
            "company_name": company_name,
            "year": int(year),
            "scenario": scenario,
            "financials": current,
            "prev_year": prev,
            "true_target": int(row["target"]),
            "calibrated_probability": float(row["calibrated_probability"]),
        }

    examples = []

    c2335 = te_meta[(te_meta["company_name"] == "C_2335")]
    if not c2335.empty:
        row = c2335.iloc[0]
        correct = row["target"] == 1 and row["calibrated_probability"] >= calibrated_thr
        if not correct:
            print(f"WARNING: C_2335 is not a correctly predicted bankruptcy at the frozen "
                  f"threshold (target={row['target']}, prob={row['calibrated_probability']:.4f}, "
                  f"threshold={calibrated_thr:.4f}). Including it anyway as requested, flagging "
                  f"the mismatch for review.")
        examples.append(make_example("C_2335", int(row["year"]), "correctly predicted bankruptcy"))

    healthy_candidates = te_meta[(te_meta["target"] == 0) & (te_meta["company_name"] != "C_2335")]
    healthy_candidates = healthy_candidates.sort_values("calibrated_probability")
    if not healthy_candidates.empty:
        r = healthy_candidates.iloc[0]
        examples.append(make_example(r["company_name"], int(r["year"]), "healthy firm, low risk"))

    elevated_candidates = te_meta[
        (te_meta["target"] == 0) &
        (te_meta["calibrated_probability"] >= calibrated_thr / 2) &
        (te_meta["calibrated_probability"] < calibrated_thr)
    ]
    if not elevated_candidates.empty:
        r = elevated_candidates.sort_values("calibrated_probability", ascending=False).iloc[0]
        examples.append(make_example(r["company_name"], int(r["year"]), "elevated risk, survived"))

    other_positives = te_meta[
        (te_meta["target"] == 1) &
        (te_meta["calibrated_probability"] >= calibrated_thr) &
        (te_meta["company_name"] != "C_2335")
    ]
    if not other_positives.empty:
        r = other_positives.sort_values("calibrated_probability", ascending=False).iloc[0]
        examples.append(make_example(
            r["company_name"], int(r["year"]), "another correctly predicted bankruptcy"))

    missed_positives = te_meta[
        (te_meta["target"] == 1) & (te_meta["calibrated_probability"] < calibrated_thr)
    ]
    if not missed_positives.empty:
        r = missed_positives.sort_values("calibrated_probability", ascending=False).iloc[0]
        examples.append(make_example(r["company_name"], int(r["year"]), "missed bankruptcy (false negative)"))

    return examples


def feature_reference_stats(df, train_mask):
    stats = {}
    for col in FEATURES_ALL:
        s = df.loc[train_mask, col]
        stats[col] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
        }
    return stats


def main(data_path=DEFAULT_DATA_PATH, models_dir=DEFAULT_MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    df, train_mask, winsor_artifacts = load_and_prepare(data_path)

    print("Fitting cluster assignment...")
    df, cluster_scaler, kmeans, k_opt = fit_cluster(df, train_mask)

    reference_stats = feature_reference_stats(df, train_mask)

    tr = df[df["year"].between(*TRAIN_YEARS)]
    va = df[df["year"].between(*VAL_YEARS)]
    te = df[df["year"].between(*TEST_YEARS)]

    X_tr, y_tr = tr[FEATURES_ALL], tr["target"]
    X_va, y_va = va[FEATURES_ALL], va["target"]
    X_te, y_te = te[FEATURES_ALL], te["target"]
    X_tr_lr, X_va_lr, X_te_lr = tr[FEATURES_LR], va[FEATURES_LR], te[FEATURES_LR]
    X_tr_dt, X_va_dt, X_te_dt = tr[FEATURES_DT], va[FEATURES_DT], te[FEATURES_DT]

    print(f"Train: {len(X_tr):,} rows, Val: {len(X_va):,} rows, Test: {len(X_te):,} rows")

    years = np.sort(df.loc[df["year"] <= VAL_YEARS[1], "year"].unique())
    tscv = TimeSeriesSplit(n_splits=5, test_size=2)
    fold_boundaries = [(years[tr_idx[0]], years[tr_idx[-1]], years[va_idx[0]], years[va_idx[-1]])
                        for tr_idx, va_idx in tscv.split(years)]

    cv_folds_lr = build_grouped_time_folds(df, fold_boundaries, FEATURES_LR)
    cv_folds = build_grouped_time_folds(df, fold_boundaries, FEATURES_ALL)

    results = []
    test_scores = {}

    print("\nFitting Logistic Regression...")
    lr_result, lr_scores = fit_logreg(df, cv_folds_lr, X_tr_lr, y_tr, X_va_lr, y_va, X_te_lr, y_te)
    results.append(lr_result)
    test_scores["LogReg (balanced)"] = lr_scores

    print("\nEvaluating Altman Z baseline...")
    z_te = te["altman_z"]
    results.append(evaluate("Altman Z", y_te, -z_te, -ALTMAN_LOW))
    test_scores["Altman Z"] = -z_te.values
    zone = pd.cut(z_te, [-np.inf, ALTMAN_LOW, ALTMAN_HIGH, np.inf],
                  labels=["distress (<1.81)", "gray (1.81-2.99)", "safe (>2.99)"])
    altman_zone_table = te.groupby(zone, observed=True)["target"].agg(["mean", "size"])
    altman_zone_table_out = {str(k): {"rate": float(v["mean"]), "n": int(v["size"])}
                              for k, v in altman_zone_table.iterrows()}

    print("\nFitting XGBoost (primary)...")
    xgb_result, xgb_scores, best, xgb, X_trva, y_trva = fit_xgboost_primary(
        df, cv_folds, X_tr, y_tr, X_va, y_va, X_te, y_te)
    results.append(xgb_result)
    test_scores["XGBoost (primary)"] = xgb_scores

    print("\nFitting XGBoost (SMOTE)...")
    sm_result, sm_scores = fit_xgboost_smote(best, X_tr, y_tr, X_va, y_va, X_te, y_te)
    results.append(sm_result)
    test_scores["XGBoost (SMOTE)"] = sm_scores

    print("\nFitting Decision Tree...")
    dt_result, dt_scores = fit_decision_tree(X_tr_dt, y_tr, X_va_dt, y_va, X_te_dt, y_te)
    results.append(dt_result)
    test_scores["Decision Tree (depth=3)"] = dt_scores

    print("\nFitting Random Forest...")
    rf_result, rf_scores = fit_random_forest(X_tr, y_tr, X_va, y_va, X_te, y_te)
    results.append(rf_result)
    test_scores["Random Forest"] = rf_scores

    print("\nFitting calibration...")
    tuning_model = best["model"]
    calibrated, calibrated_thr, calibrated_probs_te, calibrated_result, calibration_info = \
        fit_calibration(tuning_model, X_va, y_va, X_te, y_te)
    results.append(calibrated_result)
    test_scores["XGBoost (calibrated, deployed)"] = calibrated_probs_te

    print("\nComputing 1968-vs-today importance comparison...")
    importance = importance_comparison(xgb, X_te, X_trva, te)

    print("\nSelecting example firms...")
    example_firms = build_example_firms(df, te, calibrated_probs_te, calibrated_thr)

    print("\nSaving artifacts...")
    joblib.dump(calibrated, models_dir / "model_calibrated.joblib")
    joblib.dump(cluster_scaler, models_dir / "scaler.joblib")
    joblib.dump(kmeans, models_dir / "kmeans.joblib")

    (models_dir / "caps.json").write_text(json.dumps(winsor_artifacts["caps"], indent=2))
    (models_dir / "medians.json").write_text(json.dumps(winsor_artifacts["medians"], indent=2))
    (models_dir / "ta_caps.json").write_text(json.dumps(winsor_artifacts["ta_caps"]))
    (models_dir / "feature_list.json").write_text(json.dumps({
        "features_all": FEATURES_ALL, "features_lr": FEATURES_LR, "features_dt": FEATURES_DT,
        "cluster_cols": CLUSTER_COLS,
    }, indent=2))
    (models_dir / "threshold.json").write_text(json.dumps({"calibrated_threshold": calibrated_thr}, indent=2))
    (models_dir / "reference_stats.json").write_text(json.dumps(reference_stats, indent=2))
    (models_dir / "example_firms.json").write_text(json.dumps(example_firms, indent=2))

    test_score_keys = {name: sanitize_key(name) for name in test_scores}
    npz_payload = {"y_te": y_te.values}
    npz_payload.update({sanitize_key(name): np.asarray(arr) for name, arr in test_scores.items()})
    np.savez(models_dir / "test_scores.npz", **npz_payload)

    metrics = {
        "results": results,
        "deployed_model": "XGBoost (calibrated, deployed)",
        "calibrated_threshold": calibrated_thr,
        "test_set": {
            "n_rows": int(len(y_te)), "n_positives": int(y_te.sum()),
            "positive_rate": float(y_te.mean()),
        },
        "altman_zone_table": altman_zone_table_out,
        "calibration": calibration_info,
        "importance_comparison": importance,
        "k_opt": k_opt,
        "test_score_keys": test_score_keys,
        "n_train_rows": int(len(X_tr)), "n_val_rows": int(len(X_va)), "n_test_rows": int(len(X_te)),
        "n_relabeled_rows": int((df["status_label"] == "failed").sum() - df["target"].sum()),
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nDone. Test set results:")
    print(pd.DataFrame(results).set_index("model").round(4).to_string())
    print(f"\nDeployed model (calibrated) threshold: {calibrated_thr:.4f}")


if __name__ == "__main__":
    main()
