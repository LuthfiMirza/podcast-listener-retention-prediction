from __future__ import annotations

import pandas as pd

import src.drift_monitor as drift
from src.drift_monitor import drift_level, population_stability_index


def test_high_drift_psi_detection():
    reference = pd.Series([0.1] * 500 + [0.2] * 500)
    current = pd.Series([0.9] * 500 + [1.0] * 500)
    psi = population_stability_index(reference, current)
    assert psi > 0.2
    assert drift_level(psi) == "HIGH"


def test_generate_drift_report_outputs(monkeypatch, tmp_path):
    base = pd.DataFrame(
        {
            "sessions": [1, 2, 3, 4],
            "avg_session_duration": [10, 11, 12, 13],
            "skip_rate": [0.1, 0.2, 0.1, 0.2],
            "replay_rate": [0.1, 0.1, 0.2, 0.2],
            "content_variety_score": [2, 3, 4, 5],
            "days_since_last_session": [1, 2, 3, 4],
            "completion_rate": [0.7, 0.8, 0.9, 0.7],
            "peak_hour_activity": [0, 1, 0, 1],
            "engaged": [1, 1, 0, 1],
        }
    )
    new_path = tmp_path / "new.csv"
    shifted = base.copy()
    shifted["skip_rate"] = [0.9, 0.95, 0.9, 0.95]
    shifted.to_csv(new_path, index=False)
    monkeypatch.setattr(drift, "PROCESSED_DATA_PATH", new_path)
    monkeypatch.setattr(drift, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(drift, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(drift, "DRIFT_JSON_PATH", tmp_path / "drift_report.json")
    monkeypatch.setattr(drift, "DRIFT_HTML_PATH", tmp_path / "drift_report.html")
    report = drift.generate_drift_report(new_path)
    assert report["current_rows"] == 4
    assert (tmp_path / "drift_report.json").exists()
    assert (tmp_path / "drift_report.html").exists()
