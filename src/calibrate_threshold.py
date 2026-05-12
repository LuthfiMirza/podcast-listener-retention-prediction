from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, create_features
from utils import FIGURE_DIR, PROCESSED_DATA_PATH, PROJECT_ROOT, REPORT_DIR, ensure_directories, load_csv


def risk_mode(target: str) -> bool:
    return target == "at_risk_user"


def best_model_path(target: str) -> Path:
    return PROJECT_ROOT / "experiments" / ("risk_best_model.pkl" if risk_mode(target) else "best_model.pkl")


def threshold_path(target: str) -> Path:
    return PROJECT_ROOT / "experiments" / ("risk_best_threshold.json" if risk_mode(target) else "best_threshold.json")


def load_test_predictions(target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> tuple[pd.Series, np.ndarray]:
    model_path = best_model_path(target)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run python src/tune.py first.")

    artifact = joblib.load(model_path)
    data = create_features(load_csv(data_path))
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")
    X, y = data[FEATURE_COLUMNS], data[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    probabilities = artifact["model"].predict_proba(X_test)[:, 1]
    return y_test, probabilities


def evaluate_thresholds(metric: str = "f1", target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> dict[str, str | float]:
    path = threshold_path(target)
    ensure_directories([REPORT_DIR, FIGURE_DIR, path.parent])
    y_test, probabilities = load_test_predictions(target, data_path)
    roc_auc = float(roc_auc_score(y_test, probabilities))
    rows = []

    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "specificity": specificity,
                "negative_class_recall": specificity,
                "roc_auc": roc_auc,
            }
        )

    metrics = pd.DataFrame(rows)
    best_row = metrics.sort_values([metric, "f1", "precision"], ascending=False).iloc[0]

    csv_path = REPORT_DIR / ("risk_threshold_metrics.csv" if risk_mode(target) else "threshold_metrics.csv")
    chart_path = FIGURE_DIR / ("risk_threshold_tradeoff.png" if risk_mode(target) else "threshold_tradeoff.png")
    metrics.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 6))
    for column in ["precision", "recall", "f1", "accuracy", "balanced_accuracy", "specificity"]:
        plt.plot(metrics["threshold"], metrics[column], label=column)
    plt.axvline(best_row["threshold"], color="black", linestyle="--", alpha=0.65, label="best threshold")
    plt.title(f"Threshold Tradeoff Optimized for {metric.upper()} ({target})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()

    payload = {
        "threshold": round(float(best_row["threshold"]), 2),
        "target": target,
        "optimized_metric": metric,
        "precision": round(float(best_row["precision"]), 4),
        "recall": round(float(best_row["recall"]), 4),
        "f1": round(float(best_row["f1"]), 4),
        "accuracy": round(float(best_row["accuracy"]), 4),
        "balanced_accuracy": round(float(best_row["balanced_accuracy"]), 4),
        "specificity": round(float(best_row["specificity"]), 4),
        "roc_auc": round(roc_auc, 4),
    }
    path.write_text(json.dumps(payload, indent=2))
    return {"csv": str(csv_path), "chart": str(chart_path), **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate classification threshold for the best model.")
    parser.add_argument("--metric", choices=["f1", "recall", "precision"], default="f1")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    print(json.dumps(evaluate_thresholds(metric=args.metric, target=args.target, data_path=args.data), indent=2))


if __name__ == "__main__":
    main()
