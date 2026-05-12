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
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
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
from utils import FIGURE_DIR, MODEL_DIR, PROCESSED_DATA_PATH, REPORT_DIR, ensure_directories, load_csv


def load_threshold(default: float = 0.5, target: str = TARGET_COLUMN) -> float:
    path = Path("experiments") / ("risk_best_threshold.json" if target == "at_risk_user" else "best_threshold.json")
    if not path.exists():
        return default
    try:
        return float(json.loads(path.read_text()).get("threshold", default))
    except Exception:
        return default


def target_distribution(y: pd.Series) -> dict[str, dict[str, float]]:
    counts = y.value_counts().sort_index()
    total = counts.sum()
    return {
        str(label): {"count": int(count), "percentage": round(float(count / total * 100), 2)}
        for label, count in counts.items()
    }


def calculate_metrics(y_true, y_probability, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probability)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_probability)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "specificity": round(float(specificity), 4),
        "negative_class_recall": round(float(specificity), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def generate_report(model_name: str, target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> dict[str, object]:
    ensure_directories([FIGURE_DIR, REPORT_DIR])
    stem = f"risk_{model_name}" if target == "at_risk_user" else model_name
    artifact_path = MODEL_DIR / f"{stem}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Train the model first: python src/train.py --model {model_name}")

    artifact = joblib.load(artifact_path)
    data = create_features(load_csv(data_path))
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")
    X, y = data[FEATURE_COLUMNS], data[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_probability = artifact["model"].predict_proba(X_test)[:, 1]
    threshold = load_threshold(default=0.5, target=target)
    y_pred = (y_probability >= threshold).astype(int)

    confusion_path = FIGURE_DIR / ("risk_confusion_matrix.png" if target == "at_risk_user" else f"{model_name}_confusion_matrix.png")
    roc_path = FIGURE_DIR / ("risk_roc_curve.png" if target == "at_risk_user" else f"{model_name}_roc_curve.png")
    importance_path = FIGURE_DIR / ("risk_feature_importance.png" if target == "at_risk_user" else f"{model_name}_feature_importance.png")

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix @ threshold={threshold:.2f}")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_probability)
    plt.title(f"{model_name} ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()

    model = artifact["model"]
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        importance = pd.Series(estimator.feature_importances_, index=artifact["features"]).sort_values()
    elif hasattr(estimator, "coef_"):
        importance = pd.Series(abs(estimator.coef_[0]), index=artifact["features"]).sort_values()
    else:
        importance = pd.Series(dtype=float)

    if not importance.empty:
        sns.barplot(x=importance.values, y=importance.index)
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(importance_path, dpi=160)
        plt.close()

    metrics = calculate_metrics(y_test, y_probability, threshold=threshold)
    report = {
        "metrics": metrics,
        "target_distribution": target_distribution(y_test),
        "confusion_matrix": str(confusion_path),
        "roc_curve": str(roc_path),
        "feature_importance": str(importance_path) if importance_path.exists() else "not_available",
    }
    (REPORT_DIR / ("risk_evaluation_report.json" if target == "at_risk_user" else f"{model_name}_evaluation_report.json")).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model evaluation plots.")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.model, target=args.target, data_path=args.data), indent=2))


if __name__ == "__main__":
    main()
