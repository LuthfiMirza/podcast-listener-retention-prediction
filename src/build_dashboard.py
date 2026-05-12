from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT, REPORT_DIR, ensure_directories

DASHBOARD_PATH = REPORT_DIR / "monitoring_dashboard.html"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prediction_volume() -> list[dict]:
    batch_dir = REPORT_DIR / "batch_results"
    rows = []
    for path in sorted(batch_dir.glob("*.csv")) if batch_dir.exists() else []:
        try:
            rows.append({"file": path.name, "rows": len(pd.read_csv(path))})
        except Exception:
            rows.append({"file": path.name, "rows": 0})
    return rows


def build_dashboard() -> dict[str, str]:
    ensure_directories([REPORT_DIR])
    metrics = load_json(PROJECT_ROOT / "experiments" / "best_threshold.json", {})
    drift = load_json(REPORT_DIR / "drift_report.json", {"features": []})
    retrain_history = load_jsonl(PROJECT_ROOT / "experiments" / "retrain_log.jsonl")
    volume = prediction_volume()
    payload = {
        "metrics": metrics,
        "drift": drift,
        "retrain_history": retrain_history,
        "prediction_volume": volume,
    }
    data_json = json.dumps(payload)
    DASHBOARD_PATH.write_text(
        f"""
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Monitoring Dashboard</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#111827}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{border:1px solid #ddd;border-radius:8px;padding:16px;min-width:150px;background:#f9fafb}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f3f4f6}}.HIGH{{color:#b91c1c;font-weight:bold}}.MEDIUM{{color:#b45309;font-weight:bold}}.LOW{{color:#047857;font-weight:bold}}</style>
</head>
<body>
<h1>User Engagement Monitoring Dashboard</h1>
<div id="app"></div>
<script id="dashboard-data" type="application/json">{data_json}</script>
<script>
const data = JSON.parse(document.getElementById('dashboard-data').textContent);
const metricKeys = ['roc_auc','f1','recall','precision','accuracy','threshold'];
function table(rows, columns) {{
  if (!rows.length) return '<p>No data available.</p>';
  return '<table><thead><tr>' + columns.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + columns.map(c => `<td class="${{r[c] || ''}}">${{r[c] ?? ''}}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}}
const metricCards = metricKeys.map(k => `<div class="card"><strong>${{k}}</strong><br>${{data.metrics[k] ?? 'n/a'}}</div>`).join('');
const driftRows = data.drift.features || [];
const retrainRows = data.retrain_history || [];
const volumeRows = data.prediction_volume || [];
document.getElementById('app').innerHTML = `
  <h2>Current Model Metrics</h2><div class="cards">${{metricCards}}</div>
  <h2>Drift Status</h2>${{table(driftRows, ['feature','psi','ks_statistic','ks_pvalue','drift_level'])}}
  <h2>Retrain History</h2>${{table(retrainRows, ['timestamp','triggered','dry_run','threshold_auc','high_drift_features','roc_auc','reasons'])}}
  <h2>Prediction Volume</h2>${{table(volumeRows, ['file','rows'])}}
`;
</script>
</body></html>
""".strip()
    )
    return {"dashboard": str(DASHBOARD_PATH)}


def main() -> None:
    print(json.dumps(build_dashboard(), indent=2))


if __name__ == "__main__":
    main()
