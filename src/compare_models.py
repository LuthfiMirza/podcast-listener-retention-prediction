from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, average_precision_score, balanced_accuracy_score

from utils import FIGURE_DIR, REPORT_DIR, ensure_directories

METRIC_COLUMNS = ["model", "roc_auc", "pr_auc", "precision", "recall", "f1_score", "balanced_accuracy"]


def enrich_metrics_from_predictions(model_name: str, metrics: dict) -> dict:
    predictions_path = REPORT_DIR / f"{model_name}_predictions.csv"
    if not predictions_path.exists():
        return metrics
    predictions = pd.read_csv(predictions_path)
    y_true = predictions["actual"]
    y_probability = predictions["predicted_probability"]
    y_pred = (y_probability >= 0.5).astype(int)
    metrics.setdefault("pr_auc", round(float(average_precision_score(y_true, y_probability)), 4))
    metrics.setdefault("balanced_accuracy", round(float(balanced_accuracy_score(y_true, y_pred)), 4))
    return metrics


def load_metric_files(prefix: str = "") -> pd.DataFrame:
    rows = []
    pattern = f"{prefix}*_metrics.json" if prefix else "*_metrics.json"
    for path in sorted(REPORT_DIR.glob(pattern)):
        if not prefix and path.name.startswith("risk_"):
            continue
        metrics = json.loads(path.read_text())
        model_name = path.name.removesuffix("_metrics.json")
        rows.append({"model": model_name, **enrich_metrics_from_predictions(model_name, metrics)})
    if not rows:
        raise FileNotFoundError("No metric files found. Train models first with python src/train.py --model <name>.")
    frame = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[METRIC_COLUMNS].sort_values("roc_auc", ascending=False)


def plot_roc_curves(metrics: pd.DataFrame, prefix: str = "") -> Path:
    output_path = FIGURE_DIR / ("risk_roc_comparison.png" if prefix else "roc_comparison.png")
    plt.figure(figsize=(8, 6))
    for model_name in metrics["model"]:
        predictions_path = REPORT_DIR / f"{model_name}_predictions.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        RocCurveDisplay.from_predictions(
            predictions["actual"],
            predictions["predicted_probability"],
            name=model_name,
            ax=plt.gca(),
        )
    plt.title("Risk ROC Curve Comparison" if prefix else "ROC Curve Comparison")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def build_dashboard(target: str = "returned_within_7_days") -> dict[str, str]:
    ensure_directories([REPORT_DIR, FIGURE_DIR])
    prefix = "risk_" if target == "at_risk_user" else ""
    metrics = load_metric_files(prefix=prefix)
    csv_path = REPORT_DIR / ("risk_model_comparison.csv" if prefix else "model_comparison.csv")
    html_path = REPORT_DIR / ("risk_model_comparison.html" if prefix else "model_comparison.html")
    roc_path = plot_roc_curves(metrics, prefix=prefix)

    metrics.to_csv(csv_path, index=False)
    html_table = metrics.to_html(index=False, float_format="{:.4f}".format, classes="metrics")
    html_path.write_text(
        f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Listener Retention Model Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2937; }}
    table.metrics {{ border-collapse: collapse; min-width: 720px; }}
    .metrics th, .metrics td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: right; }}
    .metrics th:first-child, .metrics td:first-child {{ text-align: left; }}
    .metrics th {{ background: #f3f4f6; }}
    img {{ max-width: 900px; width: 100%; margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>Listener Retention Model Comparison</h1>
  {html_table}
  <h2>ROC Curves</h2>
  <img src="figures/roc_comparison.png" alt="ROC curve comparison">
</body>
</html>
""".strip()
    )
    return {"csv": str(csv_path), "html": str(html_path), "roc_plot": str(roc_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model comparison dashboard.")
    parser.add_argument("--target", default="returned_within_7_days")
    args = parser.parse_args()
    print(json.dumps(build_dashboard(target=args.target), indent=2))


if __name__ == "__main__":
    main()
