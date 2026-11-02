"""
analytics.py
------------
All data logic for the Student Academic Performance Analytics app:
sample-data generation, cleaning/validation, KPI derivation, rule-based
risk identification and the machine-learning early-warning model.

Kept separate from app.py so the UI code stays short and this logic can be
tested on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "student_id",
    "course",
    "semester",
    "attendance_pct",
    "assessment1",
    "assessment2",
    "assignment_completion_pct",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[3:]
FEATURES = ["attendance_pct", "assessment1", "assignment_completion_pct"]

ATTENDANCE_LABELS = ["<60%", "60–74%", "75–89%", "90–100%"]
PERFORMANCE_LABELS = [
    "Fail (<40)",
    "Pass (40–49)",
    "Good (50–59)",
    "Very good (60–74)",
    "Distinction (75+)",
]

# Friendly alternative headers that are mapped onto the required names
_ALIASES = {
    "id": "student_id",
    "studentid": "student_id",
    "student": "student_id",
    "roll_no": "student_id",
    "subject": "course",
    "sem": "semester",
    "term": "semester",
    "attendance": "attendance_pct",
    "attendance_percentage": "attendance_pct",
    "attendance_percent": "attendance_pct",
    "assessment_1": "assessment1",
    "assessment_2": "assessment2",
    "assignment_completion": "assignment_completion_pct",
    "assignment_pct": "assignment_completion_pct",
    "assignment_completion_percentage": "assignment_completion_pct",
    "assignment_completion_percent": "assignment_completion_pct",
}

RISK_ACTIONS = {
    "Low attendance": "Attendance counselling / mentor check-in",
    "Low marks": "Tutoring or remedial sessions",
    "Low assignment completion": "Assignment follow-up and deadline plan",
    "Failing": "Priority academic advising",
}


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
def generate_sample_data(n_students: int = 180, seed: int = 42) -> pd.DataFrame:
    """Synthetic student records (no real people). Deliberately a little messy
    (duplicates, blanks, out-of-range values) so the cleaning step has work to do."""
    rng = np.random.default_rng(seed)
    course_offset = {"BDA": 0, "OS": -4, "DBMS": 3, "CN": -2, "ML": -1}
    courses = list(course_offset)
    semesters = {"2025-2": 0, "2026-1": 2}

    rows = []
    for i in range(n_students):
        sid = f"S{101 + i}"
        ability = rng.normal(0, 1)
        engagement = 0.6 * ability + rng.normal(0, 0.8)
        for sem, sem_bonus in semesters.items():
            course = str(rng.choice(courses))
            att = np.clip(82 + 9 * engagement + rng.normal(0, 6), 30, 100)
            a1 = np.clip(
                66 + 15 * ability + 0.25 * (att - 82) + course_offset[course]
                + sem_bonus + rng.normal(0, 6),
                15, 100,
            )
            a2 = np.clip(a1 + 2 + 0.15 * (att - 82) + rng.normal(0, 6), 10, 100)
            asg = np.clip(86 + 10 * engagement + rng.normal(0, 8), 20, 100)
            rows.append(
                {
                    "student_id": sid,
                    "course": course,
                    "semester": sem,
                    "attendance_pct": int(round(att)),
                    "assessment1": int(round(a1)),
                    "assessment2": int(round(a2)),
                    "assignment_completion_pct": int(round(asg)),
                }
            )
    df = pd.DataFrame(rows)

    # --- inject realistic data-quality problems (Veracity!) ---
    dup = df.sample(3, random_state=seed).copy()
    dup["student_id"] = dup["student_id"].str.lower().radd(" ") + " "  # ' s105 '
    df = pd.concat([df, dup], ignore_index=True)
    df = df.astype({c: float for c in NUMERIC_COLUMNS})
    idx = np.random.default_rng(seed + 1).choice(len(df), size=8, replace=False)
    df.loc[idx[:3], "attendance_pct"] = np.nan
    df.loc[idx[3:5], "assessment2"] = np.nan
    df.loc[idx[5], "attendance_pct"] = 105
    df.loc[idx[6], "assessment1"] = -5
    df.loc[idx[7], "assignment_completion_pct"] = np.nan
    return df


# --------------------------------------------------------------------------
# Cleaning & validation
# --------------------------------------------------------------------------
def _normalise_name(col: str) -> str:
    c = str(col).strip().lower()
    for ch in ("(", ")"):
        c = c.replace(ch, "")
    for ch in (" ", "-", "%", "/", "."):
        c = c.replace(ch, "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardise headers/text, enforce numeric ranges, drop duplicates.
    Raises ValueError if required columns are missing."""
    df = raw.copy()
    rename_map = {}
    for col in df.columns:
        new = _normalise_name(col)
        new = _ALIASES.get(new, new)
        if new != col:
            rename_map[col] = new
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
            + ".  Required: " + ", ".join(REQUIRED_COLUMNS)
        )

    report = {"rows_in": int(len(df)), "renamed_columns": rename_map}

    # Text fields
    for col in ("student_id", "course", "semester"):
        df[col] = df[col].astype("string").str.strip()
    df["student_id"] = df["student_id"].str.upper()
    df["course"] = df["course"].str.upper()

    # Rows without a usable key are dropped
    key_ok = df[["student_id", "course", "semester"]].notna().all(axis=1)
    key_ok &= (df[["student_id", "course", "semester"]] != "").all(axis=1)
    report["dropped_missing_keys"] = int((~key_ok).sum())
    df = df[key_ok]

    for col in ("student_id", "course", "semester"):
        df[col] = df[col].astype(str)

    # Numeric fields + range validation (0–100)
    invalid = 0
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & ~df[col].between(0, 100)
        invalid += int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_values_blanked"] = invalid

    # Duplicates (after standardising text, so ' s105 ' == 'S105')
    before = len(df)
    df = df.drop_duplicates(subset=["student_id", "course", "semester"], keep="first")
    report["duplicates_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["missing_by_column"] = {c: int(df[c].isna().sum()) for c in NUMERIC_COLUMNS}
    return df, report


# --------------------------------------------------------------------------
# Metrics & risk
# --------------------------------------------------------------------------
def derive_metrics(df: pd.DataFrame, pass_mark: float = 40) -> pd.DataFrame:
    df = df.copy()
    df["average_marks"] = df[["assessment1", "assessment2"]].mean(axis=1, skipna=True)
    df["improvement"] = df["assessment2"] - df["assessment1"]
    df["pass_status"] = np.where(
        df["average_marks"].isna(),
        "Unknown",
        np.where(df["average_marks"] >= pass_mark, "Pass", "Fail"),
    )
    df["attendance_band"] = pd.cut(
        df["attendance_pct"], bins=[0, 60, 75, 90, 100.0001], right=False,
        labels=ATTENDANCE_LABELS,
    )
    df["performance_band"] = pd.cut(
        df["average_marks"], bins=[0, 40, 50, 60, 75, 100.0001], right=False,
        labels=PERFORMANCE_LABELS,
    )
    return df


def compute_risk(
    df: pd.DataFrame,
    attendance_thr: float = 75,
    marks_thr: float = 50,
    assignment_thr: float = 70,
    pass_mark: float = 40,
) -> pd.DataFrame:
    """Transparent rule-based flags (same rules as the notebook) + risk level."""
    df = df.copy()
    flags = pd.DataFrame(
        {
            "Low attendance": df["attendance_pct"] < attendance_thr,
            "Low marks": df["average_marks"] < marks_thr,
            "Low assignment completion": df["assignment_completion_pct"] < assignment_thr,
            "Failing": df["average_marks"] < pass_mark,
        }
    )
    n_rule_flags = flags[["Low attendance", "Low marks", "Low assignment completion"]].sum(axis=1)
    df["risk_flags"] = n_rule_flags.astype(int)
    df["risk_level"] = np.where(
        (n_rule_flags >= 2) | flags["Failing"],
        "High",
        np.where(n_rule_flags == 1, "Medium", "Low"),
    )
    if df.empty:
        df["risk_reasons"] = pd.Series(dtype=str)
        df["suggested_action"] = pd.Series(dtype=str)
        return df

    cols = list(flags.columns)
    reasons, actions = [], []
    for row in flags.to_numpy():
        hit = [c for c, v in zip(cols, row) if v]
        reasons.append("; ".join(hit) if hit else "—")
        actions.append("; ".join(RISK_ACTIONS[c] for c in hit) if hit else "No action needed")
    df["risk_reasons"] = reasons
    df["suggested_action"] = actions
    return df


def sort_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["risk_flags", "average_marks", "attendance_pct"],
        ascending=[False, True, True],
        na_position="last",
    )


def course_kpis(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("course").agg(
        students=("student_id", "nunique"),
        avg_attendance=("attendance_pct", "mean"),
        avg_marks=("average_marks", "mean"),
        avg_assignment=("assignment_completion_pct", "mean"),
        high_risk=("risk_level", lambda s: int((s == "High").sum())),
    )
    known = df[df["pass_status"] != "Unknown"]
    pr = known.groupby("course")["pass_status"].apply(lambda s: (s == "Pass").mean() * 100)
    g = g.join(pr.rename("pass_rate"))
    return g.round(2).reset_index()


def pass_rate(df: pd.DataFrame) -> float:
    known = df[df["pass_status"] != "Unknown"]
    return float((known["pass_status"] == "Pass").mean() * 100) if len(known) else float("nan")


# --------------------------------------------------------------------------
# Machine learning: early-warning model
# --------------------------------------------------------------------------
def train_risk_models(data: pd.DataFrame, pass_mark: float = 40, test_size: float = 0.25,
                      seed: int = 42) -> dict:
    """Predict whether a student will FAIL Assessment 2 (score < pass_mark) using
    only attendance, Assessment 1 and assignment completion.

    Using Assessment 2 as the target (rather than the average of both tests) avoids
    data leakage: the target is not part of the inputs.

    `data` must contain FEATURES + 'assessment2'. Returns a dict with ok=False and a
    message when there is not enough data."""
    labelled = data.dropna(subset=["assessment2"]).copy()
    y = (labelled["assessment2"] < pass_mark).astype(int)
    X = labelled[FEATURES]
    counts = y.value_counts()
    n_fail, n_pass = int(counts.get(1, 0)), int(counts.get(0, 0))
    if len(labelled) < 30 or n_fail < 5 or n_pass < 5:
        return {
            "ok": False,
            "message": (
                f"Not enough data to train a reliable model (need ≥30 rows with ≥5 students "
                f"in each outcome; found {len(labelled)} rows: {n_fail} failing / {n_pass} passing "
                f"at pass mark {pass_mark:g}). Upload more records or adjust the pass mark."
            ),
        }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=seed,
            ),
        ),
    }

    rows, confusion, importance = [], {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_te, pred),
                "Precision (at-risk)": precision_score(y_te, pred, zero_division=0),
                "Recall (at-risk)": recall_score(y_te, pred, zero_division=0),
                "F1 (at-risk)": f1_score(y_te, pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_te, proba),
            }
        )
        confusion[name] = confusion_matrix(y_te, pred, labels=[0, 1])
        est = model[-1]
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=FEATURES)
        else:
            imp = pd.Series(np.abs(est.coef_[0]), index=FEATURES)
            imp = imp / imp.sum()
        importance[name] = imp

    metrics = pd.DataFrame(rows).round(3)
    best_name = metrics.sort_values(["F1 (at-risk)", "ROC AUC"], ascending=False).iloc[0]["Model"]

    # Out-of-fold probabilities for every labelled row (honest, not in-sample)
    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_proba = cross_val_predict(clone(models[best_name]), X, y, cv=cv, method="predict_proba")[:, 1]
    final_model = clone(models[best_name]).fit(X, y)

    prob = pd.Series(np.nan, index=data.index, dtype=float)
    prob.loc[labelled.index] = cv_proba
    unlabelled = data.index.difference(labelled.index)
    if len(unlabelled):
        prob.loc[unlabelled] = final_model.predict_proba(data.loc[unlabelled, FEATURES])[:, 1]

    return {
        "ok": True,
        "metrics": metrics,
        "best_name": best_name,
        "confusion": confusion,
        "importance": importance,
        "final_model": final_model,
        "prob": prob,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "at_risk_rate": float(y.mean()),
        "baseline_accuracy": float(1 - y.mean()),
    }
