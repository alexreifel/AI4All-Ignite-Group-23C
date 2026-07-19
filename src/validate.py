"""Schema and label-structure checks.

The relabeling logic in features.relabel_target assumes that a firm's raw
status_label is constant across all of its rows (a firm is either "alive"
in every row, or "failed" in every row, right up to its last observed
year). The notebook only prints how many firms break that assumption.
Here it is a hard check, so a future dataset version that changes label
structure fails loudly instead of silently mislabeling the target.
"""

from src.constants import RAW_DOLLAR_COLS


def check_schema(df):
    """Check required columns, nulls, and duplicate firm-year rows."""
    errors = []

    required = {"company_name", "status_label", "year"} | set(RAW_DOLLAR_COLS)
    missing_cols = required - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    if errors:
        raise ValueError("Schema check failed:\n" + "\n".join(errors))

    if df["company_name"].isnull().any():
        errors.append("company_name has null values")
    if df["year"].isnull().any():
        errors.append("year has null values")

    nulls = df[RAW_DOLLAR_COLS].isnull().sum()
    if nulls.sum() > 0:
        errors.append(f"Missing values found in raw columns:\n{nulls[nulls > 0]}")

    dupes = df.duplicated(["company_name", "year"]).sum()
    if dupes > 0:
        errors.append(f"Duplicate (company_name, year) rows: {dupes}")

    bad_labels = set(df["status_label"].unique()) - {"alive", "failed"}
    if bad_labels:
        errors.append(f"Unexpected status_label values: {bad_labels}")

    if errors:
        raise ValueError("Schema check failed:\n" + "\n".join(errors))


def check_label_structure(df):
    """Check that each firm's raw status_label is constant across its rows.

    This is the assumption features.relabel_target depends on: every
    historical row of a failed firm is marked failed, so we can safely
    relabel all but the final observed row to alive.
    """
    labels_per_firm = df.groupby("company_name")["status_label"].nunique()
    inconsistent = labels_per_firm[labels_per_firm > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            f"Label structure check failed: {len(inconsistent)} firms have "
            f"more than one distinct status_label across their rows: "
            f"{inconsistent.index.tolist()[:10]}. The relabeling logic "
            f"assumes status_label is constant per firm."
        )


def validate_for_training(df):
    """Run schema and label-structure checks, raise on any failure."""
    check_schema(df)
    check_label_structure(df)


def check_drift(df, reference_stats, feature_cols):
    """Compare a new dataset's per-feature stats against training reference stats.

    reference_stats is expected to be a dict of
    {feature: {"mean": ..., "std": ..., "min": ..., "max": ...}}
    as saved by train.py. Returns a list of human-readable warning strings,
    empty if nothing looks unusual. Does not raise, since drift is a signal
    to review before retraining, not a hard failure.
    """
    warnings = []
    for col in feature_cols:
        if col not in reference_stats:
            warnings.append(f"{col}: no reference stats available")
            continue
        if col not in df.columns:
            warnings.append(f"{col}: missing from new data")
            continue

        ref = reference_stats[col]
        new_mean = df[col].mean()
        ref_std = ref["std"] if ref["std"] > 0 else 1e-9
        z = abs(new_mean - ref["mean"]) / ref_std
        if z > 3:
            warnings.append(
                f"{col}: new mean {new_mean:.4g} is {z:.1f} std away from "
                f"training mean {ref['mean']:.4g}"
            )

        new_min, new_max = df[col].min(), df[col].max()
        if new_min < ref["min"] or new_max > ref["max"]:
            warnings.append(
                f"{col}: new range [{new_min:.4g}, {new_max:.4g}] falls "
                f"outside training range [{ref['min']:.4g}, {ref['max']:.4g}]"
            )

    return warnings
