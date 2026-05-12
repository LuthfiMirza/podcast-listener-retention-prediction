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

from feature_engineering import TARGET_COLUMN
from utils import FIGURE_DIR, PROCESSED_DATA_PATH, REPORT_DIR, ensure_directories, load_csv


def detect_target(data: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in data.columns:
            raise ValueError(f"Target column {requested!r} not found")
        return requested
    if "returned_within_7_days" in data.columns:
        return "returned_within_7_days"
    if "engaged" in data.columns:
        return "engaged"
    raise ValueError("No supported target column found: returned_within_7_days or engaged")


def output_paths(target: str) -> tuple[Path, Path]:
    if target == "at_risk_user":
        return REPORT_DIR / "risk_target_distribution.csv", FIGURE_DIR / "risk_target_distribution.png"
    return REPORT_DIR / "target_distribution.csv", FIGURE_DIR / "target_distribution.png"


def build_target_distribution(data_path: Path = PROCESSED_DATA_PATH, target: str | None = None) -> dict[str, object]:
    ensure_directories([REPORT_DIR, FIGURE_DIR])
    data = load_csv(data_path)
    target_column = detect_target(data, target)
    counts = data[target_column].value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    distribution = pd.DataFrame(
        {
            "target": counts.index.astype(str),
            "count": counts.values.astype(int),
            "percentage": (counts.values / total * 100).round(2),
        }
    )

    csv_path, plot_path = output_paths(target_column)
    distribution.to_csv(csv_path, index=False)

    labels = [f"{target_column}={value}" for value in distribution["target"]]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, distribution["count"], color=["#22c55e", "#ef4444"][: len(distribution)])
    for bar, pct in zip(bars, distribution["percentage"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.2f}%", ha="center", va="bottom")
    plt.title(f"Target Distribution: {target_column}")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "target_column": target_column,
        "rows": total,
        "distribution": distribution.to_dict(orient="records"),
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect binary target distribution.")
    parser.add_argument("--target", default=None)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    build_target_distribution(args.data, args.target)


if __name__ == "__main__":
    main()
