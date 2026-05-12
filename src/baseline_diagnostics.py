from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, create_features
from utils import MODEL_DIR, PROCESSED_DATA_PATH, REPORT_DIR, ensure_directories, load_csv


def classification_metrics(y_true, y_probability, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probability)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": round(float(average_precision_score(y_true, y_probability)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "specificity": round(float(specificity), 4),
        "negative_class_recall": round(float(specificity), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def load_threshold(default: float = 0.5, target: str = TARGET_COLUMN) -> float:
    path = Path("experiments") / ("risk_best_threshold.json" if target == "at_risk_user" else "best_threshold.json")
    if not path.exists():
        return default
    try:
        return float(json.loads(path.read_text()).get("threshold", default))
    except Exception:
        return default


def run_baseline_diagnostics(seed: int = 42, target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    ensure_directories([REPORT_DIR])
    data = load_csv(data_path)
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        data = create_features(data)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")
    X, y = data[FEATURE_COLUMNS], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    rows: list[dict[str, object]] = []
    threshold = load_threshold(target=target)

    model_path = MODEL_DIR / ("risk_random_forest_tuned.joblib" if target == "at_risk_user" and (MODEL_DIR / "risk_random_forest_tuned.joblib").exists() else ("risk_random_forest.joblib" if target == "at_risk_user" else "random_forest.joblib"))
    if model_path.exists():
        artifact = joblib.load(model_path)
        y_probability = artifact["model"].predict_proba(X_test)[:, 1]
        rows.append({"model": f"trained_{target}_random_forest_tuned_threshold_0_5", "threshold": 0.5, **classification_metrics(y_test, y_probability, 0.5)})
        rows.append({"model": f"trained_{target}_random_forest_tuned_calibrated", "threshold": threshold, **classification_metrics(y_test, y_probability, threshold)})

    dummy_configs = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ("dummy_stratified", DummyClassifier(strategy="stratified", random_state=seed)),
        ("dummy_constant_positive", DummyClassifier(strategy="constant", constant=1, random_state=seed)),
    ]
    for name, dummy in dummy_configs:
        dummy.fit(X_train, y_train)
        if hasattr(dummy, "predict_proba"):
            y_probability = dummy.predict_proba(X_test)[:, 1]
        else:
            y_probability = dummy.predict(X_test)
        rows.append({"model": name, "threshold": 0.5, **classification_metrics(y_test, y_probability, 0.5)})

    output = pd.DataFrame(rows)
    output_path = REPORT_DIR / ("risk_baseline_diagnostics.csv" if target == "at_risk_user" else "baseline_diagnostics.csv")
    output.to_csv(output_path, index=False)
    print(json.dumps({"csv": str(output_path), "rows": output.to_dict(orient="records")}, indent=2))
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare model against dummy baselines.")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baseline_diagnostics(seed=args.seed, target=args.target, data_path=args.data)


if __name__ == "__main__":
    main()
