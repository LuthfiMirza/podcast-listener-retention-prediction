from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils import PROJECT_ROOT, REPORT_DIR, ensure_directories

DRIFT_REPORT_PATH = REPORT_DIR / "drift_report.json"
RETRAIN_LOG_PATH = PROJECT_ROOT / "experiments" / "retrain_log.jsonl"
DEFAULT_METRICS_PATH = REPORT_DIR / "random_forest_tuned_metrics.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_current_auc() -> float | None:
    for path in [DEFAULT_METRICS_PATH, REPORT_DIR / "logistic_regression_metrics.json"]:
        metrics = load_json(path)
        if "roc_auc" in metrics:
            return float(metrics["roc_auc"])
    threshold_metrics = load_json(PROJECT_ROOT / "experiments" / "best_threshold.json")
    return float(threshold_metrics["roc_auc"]) if "roc_auc" in threshold_metrics else None


def should_retrain(threshold_auc: float) -> tuple[bool, dict]:
    drift_report = load_json(DRIFT_REPORT_PATH)
    high_drift = int(drift_report.get("high_drift_features", 0))
    auc = get_current_auc()
    reasons = []
    if high_drift > 2:
        reasons.append(f"high_drift_features={high_drift}")
    if auc is not None and auc < threshold_auc:
        reasons.append(f"roc_auc={auc:.4f}<threshold_auc={threshold_auc:.4f}")
    return bool(reasons), {"high_drift_features": high_drift, "roc_auc": auc, "reasons": reasons}


def run_pipeline(tune_trials: int) -> list[str]:
    commands = [
        [sys.executable, "data/generate_synthetic_data.py"],
        [sys.executable, "src/data_pipeline.py"],
        [sys.executable, "src/feature_engineering.py"],
        [sys.executable, "src/train.py", "--model", "random_forest"],
        [sys.executable, "src/tune.py", "--model", "random_forest", "--trials", str(tune_trials)],
        [sys.executable, "src/calibrate_threshold.py", "--metric", "f1"],
    ]
    executed = []
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        executed.append(" ".join(command))
    return executed


def append_log(event: dict) -> None:
    ensure_directories([RETRAIN_LOG_PATH.parent])
    with RETRAIN_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger retraining when drift or model degradation exceeds limits.")
    parser.add_argument("--threshold-auc", type=float, default=0.92)
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    triggered, context = should_retrain(args.threshold_auc)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggered": triggered,
        "dry_run": args.dry_run,
        "threshold_auc": args.threshold_auc,
        **context,
    }
    if triggered and not args.dry_run:
        event["commands"] = run_pipeline(args.tune_trials)
    append_log(event)
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
