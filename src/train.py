"""
Trains the fraud detector in two stages, per the project roadmap's
Phase 2 ("baseline -> real model"):

  1. Logistic regression  -> establishes the precision/recall floor.
  2. Gradient-boosted trees (XGBoost if installed, otherwise sklearn's
     HistGradientBoostingClassifier as a drop-in fallback) -> the model
     you'd actually ship.

Both are evaluated on the same held-out test set so the lift from
stage 1 to stage 2 is a fair comparison, not an artifact of different
data splits.

Usage:
    python src/train.py --data data/transactions.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Supports both `python -m src.train` and legacy `python src/train.py`.
    from .features import FEATURE_COLUMNS, LABEL_COLUMN, AVG_ORDER_VALUE, AVG_FRAUD_LOSS
except ImportError:  # pragma: no cover
    from features import FEATURE_COLUMNS, LABEL_COLUMN, AVG_ORDER_VALUE, AVG_FRAUD_LOSS

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


def load_split(data_path: str, test_size: float = 0.2, validation_size: float = 0.2, seed: int = 7):
    """Return train, validation, and test partitions with stratification."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("dataset is empty")
    missing = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    if y.nunique() != 2:
        raise ValueError("dataset must contain both fraud and non-fraud labels")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val, y_train_val, test_size=relative_validation_size,
        stratify=y_train_val, random_state=seed
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_baseline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def build_boosted():
    # Fraud is rare, so weight the positive class instead of resampling —
    # keeps the training distribution honest to what the model will see
    # in production.
    if HAVE_XGB:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            scale_pos_weight=15,  # ~ (1 - fraud_rate) / fraud_rate, tune to your real rate
            random_state=7,
        )
    print("xgboost not installed — falling back to sklearn's "
          "HistGradientBoostingClassifier. Same interface, install "
          "xgboost for the version this pipeline is designed around.")
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.06,
        random_state=7,
    )


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "expected_cost": round(fp * AVG_ORDER_VALUE + fn * AVG_FRAUD_LOSS, 2),
    }


def pick_cost_minimizing_threshold(y_true, y_proba) -> dict:
    """Scans thresholds and returns the one with the lowest expected
    dollar cost — see features.py for the cost constants. This is the
    threshold you should actually ship, not the one that maximizes F1."""
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [evaluate_at_threshold(y_true, y_proba, t) for t in candidates]
    return min(scored, key=lambda r: r["expected_cost"])


PROJECT_ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "transactions.csv"))
    ap.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--reports-dir", type=str, default=str(PROJECT_ROOT / "reports"))
    args = ap.parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    X_train, X_validation, X_test, y_train, y_validation, y_test = load_split(args.data)

    results = {}

    # --- Stage 1: baseline ---
    baseline = build_baseline()
    baseline.fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    results["baseline_logistic_regression"] = {
        "auc_pr": round(average_precision_score(y_test, baseline_proba), 4),
        "at_threshold_0.5": evaluate_at_threshold(y_test, baseline_proba, 0.5),
    }
    joblib.dump(baseline, Path(args.models_dir) / "baseline_model.joblib")

    # --- Stage 2: boosted ---
    boosted = build_boosted()
    if HAVE_XGB:
        boosted.set_params(scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))
        boosted.fit(X_train, y_train)
    else:
        sample_weight = np.where(y_train.to_numpy() == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
        boosted.fit(X_train, y_train, sample_weight=sample_weight)
    validation_proba = boosted.predict_proba(X_validation)[:, 1]
    boosted_proba = boosted.predict_proba(X_test)[:, 1]

    selected = pick_cost_minimizing_threshold(y_validation, validation_proba)
    final = evaluate_at_threshold(y_test, boosted_proba, selected["threshold"])
    results["boosted_model"] = {
        "model_type": "XGBoost" if HAVE_XGB else "HistGradientBoostingClassifier (fallback)",
        "auc_pr": round(average_precision_score(y_test, boosted_proba), 4),
        "at_threshold_0.5": evaluate_at_threshold(y_test, boosted_proba, 0.5),
        "threshold_selection": {"split": "validation", **selected},
        "at_cost_minimizing_threshold": final,
    }
    joblib.dump(boosted, Path(args.models_dir) / "fraud_model.joblib")

    # Precision-recall curve points, saved for evaluate.py / the dashboard
    p, r, t = precision_recall_curve(y_test, boosted_proba)
    pr_curve = [{"threshold": round(float(th), 3), "precision": round(float(pi), 4), "recall": round(float(ri), 4)}
                for pi, ri, th in zip(p[:-1], r[:-1], t)]
    results["precision_recall_curve"] = pr_curve

    with open(Path(args.reports_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Baseline (logistic regression) ===")
    print(json.dumps(results["baseline_logistic_regression"]["at_threshold_0.5"], indent=2))
    print(f"AUC-PR: {results['baseline_logistic_regression']['auc_pr']}")

    print(f"\n=== Boosted model ({results['boosted_model']['model_type']}) ===")
    print("At default threshold 0.5:")
    print(json.dumps(results["boosted_model"]["at_threshold_0.5"], indent=2))
    print("Validation-selected threshold, evaluated on untouched test set:")
    print(json.dumps(results["boosted_model"]["at_cost_minimizing_threshold"], indent=2))
    print(f"AUC-PR: {results['boosted_model']['auc_pr']}")
    print(f"\nSaved models to {args.models_dir}/, metrics to {args.reports_dir}/metrics.json")


if __name__ == "__main__":
    main()
