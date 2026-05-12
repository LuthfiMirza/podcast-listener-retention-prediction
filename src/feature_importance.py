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
import shap
from sklearn.model_selection import train_test_split

from feature_engineering import create_features, split_features_target
from utils import FIGURE_DIR, PROCESSED_DATA_PATH, PROJECT_ROOT, REPORT_DIR, ensure_directories, load_csv

BEST_MODEL_PATH = PROJECT_ROOT / "experiments" / "best_model.pkl"


def unwrap_estimator(model):
    return model.named_steps["model"] if hasattr(model, "named_steps") else model


def compute_shap_importance(sample_size: int = 500) -> dict[str, str]:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Missing experiments/best_model.pkl. Run python src/tune.py first.")

    ensure_directories([REPORT_DIR, FIGURE_DIR])
    artifact = joblib.load(BEST_MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["features"]
    data = create_features(load_csv(PROCESSED_DATA_PATH))
    X, y = split_features_target(data)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    estimator = unwrap_estimator(model)
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[:, :, -1]
    except Exception:
        background = shap.sample(X, min(100, len(X)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(X_sample)
        shap_values = explanation.values[:, :, -1] if explanation.values.ndim == 3 else explanation.values

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    csv_path = REPORT_DIR / "feature_importance.csv"
    plot_path = FIGURE_DIR / "shap_top20.png"
    importance.to_csv(csv_path, index=False)

    top20 = importance.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, max(5, len(top20) * 0.35)))
    plt.barh(top20["feature"], top20["mean_abs_shap"])
    plt.title("Top 20 Features by Mean Absolute SHAP")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    return {"csv": str(csv_path), "plot": str(plot_path), "top_feature": str(importance.iloc[0]["feature"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance for the best model.")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(compute_shap_importance(sample_size=args.sample_size), indent=2))


if __name__ == "__main__":
    main()
