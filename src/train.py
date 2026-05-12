from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, create_features
from seed_everything import seed_everything
from utils import MODEL_DIR, PROCESSED_DATA_PATH, REPORT_DIR, ensure_directories, load_csv


def artifact_stem(model_name: str, target: str) -> str:
    return f"risk_{model_name}" if target == "at_risk_user" else model_name


def build_model(model_name: str, random_state: int = 42):
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("Install xgboost to train the xgboost model.")
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("Install lightgbm to train the lightgbm model.")
        return LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_predictions(y_true, y_probability, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
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


def get_model_params(model) -> dict:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    return estimator.get_params()


def log_mlflow_run(model_name: str, model, metrics: dict[str, float], artifact_paths: list[Path], seed: int, target: str) -> None:
    if mlflow is None:
        return
    mlflow.set_experiment("user-engagement-prediction")
    with mlflow.start_run(run_name=f"train_{model_name}_{target}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target)
        mlflow.log_param("seed", seed)
        mlflow.log_params(get_model_params(model))
        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(model, artifact_path="model")


def load_training_frame(data_path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    data = load_csv(data_path)
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        data = create_features(data)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")
    return data[FEATURE_COLUMNS], data[target]


def train_model(model_name: str, seed: int = 42, target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> dict[str, float]:
    seed_everything(seed)
    X, y = load_training_frame(data_path, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = build_model(model_name, random_state=seed)
    model.fit(X_train, y_train)
    y_probability = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_probability)

    ensure_directories([MODEL_DIR, REPORT_DIR])
    stem = artifact_stem(model_name, target)
    model_path = MODEL_DIR / f"{stem}.joblib"
    predictions_path = REPORT_DIR / f"{stem}_predictions.csv"
    metrics_path = REPORT_DIR / f"{stem}_metrics.json"
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "target": target, "data_path": str(data_path)}, model_path)
    pd.DataFrame({"actual": y_test, "predicted_probability": y_probability}).to_csv(
        predictions_path, index=False
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log_mlflow_run(model_name, model, metrics, [model_path, predictions_path, metrics_path], seed, target)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an engagement prediction model.")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        default="xgboost",
    )
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_model(args.model, seed=args.seed, target=args.target, data_path=args.data)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
