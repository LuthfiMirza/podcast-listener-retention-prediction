from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from data_connector import detect_schema, apply_schema_mapping
from feature_engineering import FEATURE_COLUMNS, create_features
from utils import FIGURE_DIR, PROCESSED_DATA_PATH, PROJECT_ROOT, REPORT_DIR, ensure_directories, load_csv

DRIFT_JSON_PATH = REPORT_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORT_DIR / "drift_report.html"


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        quantiles = np.linspace(reference.min(), reference.max() + 1e-6, bins + 1)
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    ref_counts = pd.cut(reference, quantiles, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(current, quantiles, include_lowest=True).value_counts(sort=False)
    ref_pct = (ref_counts / len(reference)).replace(0, 1e-6)
    cur_pct = (cur_counts / len(current)).replace(0, 1e-6)
    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def drift_level(psi: float) -> str:
    if psi > 0.2:
        return "HIGH"
    if psi >= 0.1:
        return "MEDIUM"
    return "LOW"


def prepare_new_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    schema = detect_schema(list(data.columns))
    mapped = apply_schema_mapping(data, schema)
    return create_features(mapped)


def generate_drift_report(new_data_path: str | Path) -> dict:
    ensure_directories([REPORT_DIR, FIGURE_DIR])
    reference = create_features(load_csv(PROCESSED_DATA_PATH))
    current = prepare_new_data(Path(new_data_path))
    feature_rows = []

    for feature in FEATURE_COLUMNS:
        if feature not in current.columns or feature not in reference.columns:
            continue
        ref_values = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur_values = pd.to_numeric(current[feature], errors="coerce").dropna()
        ks_statistic, ks_pvalue = ks_2samp(ref_values, cur_values) if not cur_values.empty else (0.0, 1.0)
        psi = population_stability_index(ref_values, cur_values)
        feature_rows.append(
            {
                "feature": feature,
                "psi": round(psi, 6),
                "ks_statistic": round(float(ks_statistic), 6),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "drift_level": drift_level(psi),
            }
        )

    high_count = sum(row["drift_level"] == "HIGH" for row in feature_rows)
    medium_count = sum(row["drift_level"] == "MEDIUM" for row in feature_rows)
    report = {
        "new_data_path": str(new_data_path),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "high_drift_features": high_count,
        "medium_drift_features": medium_count,
        "features": feature_rows,
    }
    DRIFT_JSON_PATH.write_text(json.dumps(report, indent=2))
    write_html(report)
    return report


def write_html(report: dict) -> None:
    rows = "\n".join(
        f"<tr><td>{row['feature']}</td><td>{row['psi']:.4f}</td><td>{row['ks_statistic']:.4f}</td><td>{row['ks_pvalue']:.4f}</td><td>{row['drift_level']}</td></tr>"
        for row in report["features"]
    )
    DRIFT_HTML_PATH.write_text(
        f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Drift Report</title>
<style>body{{font-family:Arial;margin:40px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f3f4f6}}</style>
</head><body>
<h1>Drift Report</h1>
<p>Rows: reference={report['reference_rows']}, current={report['current_rows']}</p>
<p>HIGH drift features: {report['high_drift_features']} | MEDIUM drift features: {report['medium_drift_features']}</p>
<table><thead><tr><th>Feature</th><th>PSI</th><th>KS Statistic</th><th>KS p-value</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor feature drift with PSI and KS tests.")
    parser.add_argument("--new-data", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_drift_report(args.new_data), indent=2))


if __name__ == "__main__":
    main()
