from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import PROCESSED_DATA_PATH, ensure_directories, load_csv

DEFAULT_OUTPUT = Path("data/processed/engagement_features_with_risk_target.csv")


def create_risk_target(output: str | None = None) -> dict[str, object]:
    data = load_csv(PROCESSED_DATA_PATH)
    required = {"returned_within_7_days", "days_since_last_session", "completion_rate", "skip_rate"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for at_risk_user target: {sorted(missing)}")

    # Business intervention target calibrated to a practical 15%-40% positive range.
    # A listener is at risk if they did not return, show inactivity with weaker completion,
    # or combine high skipping with moderate/low completion.
    data["at_risk_user"] = (
        (data["returned_within_7_days"] == 0)
        | ((data["days_since_last_session"] > 7) & (data["completion_rate"] < 0.80))
        | ((data["skip_rate"] > 0.25) & (data["completion_rate"] < 0.75))
    ).astype(int)

    output_path = Path(output) if output else DEFAULT_OUTPUT
    ensure_directories([output_path.parent])
    data.to_csv(output_path, index=False)
    counts = data["at_risk_user"].value_counts().sort_index()
    distribution = {
        str(label): {"count": int(count), "percentage": round(float(count / len(data) * 100), 2)}
        for label, count in counts.items()
    }
    summary = {"output": str(output_path), "rows": int(len(data)), "at_risk_distribution": distribution}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create optional business-focused at_risk_user target without changing the main pipeline.")
    parser.add_argument("--output", default=None, help=f"Optional output CSV path. Defaults to {DEFAULT_OUTPUT}")
    args = parser.parse_args()
    create_risk_target(args.output)


if __name__ == "__main__":
    main()
