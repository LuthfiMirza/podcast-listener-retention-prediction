from __future__ import annotations

import numpy as np

import src.calibrate_threshold as calibrate


def test_threshold_sweep_valid_range(monkeypatch, tmp_path):
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.8, 0.95])
    monkeypatch.setattr(calibrate, "load_test_predictions", lambda: (y_true, probabilities))
    monkeypatch.setattr(calibrate, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(calibrate, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(calibrate, "BEST_THRESHOLD_PATH", tmp_path / "best_threshold.json")
    result = calibrate.evaluate_thresholds(metric="f1")
    assert 0.10 <= result["threshold"] <= 0.90
    assert 0 <= result["f1"] <= 1
    assert (tmp_path / "threshold_metrics.csv").exists()
    assert (tmp_path / "best_threshold.json").exists()
