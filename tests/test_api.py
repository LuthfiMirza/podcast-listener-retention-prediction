from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)
PAYLOAD = {
    "sessions": 10,
    "avg_session_duration": 32.5,
    "skip_rate": 0.22,
    "replay_rate": 0.12,
    "content_variety_score": 7,
    "days_since_last_session": 2,
    "completion_rate": 0.76,
    "peak_hour_activity": 1,
}


def test_health_and_schema():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    schema = client.get("/schema")
    assert schema.status_code == 200
    assert "skip_rate" in schema.json()["fields"]


def test_predict_v1_and_validation():
    response = client.post("/v1/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["engagement_probability"] <= 1
    assert body["risk_segment"] in {"high_engagement", "likely_engaged", "at_risk"}
    invalid = client.post("/v1/predict", json={**PAYLOAD, "skip_rate": 2})
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_failed"


def test_batch_predict_and_metrics():
    response = client.post("/batch-predict", json=[PAYLOAD, PAYLOAD])
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 2
    assert body["status"] == "completed"
    assert len(body["predictions"]) == 2
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "user_engagement_model_auc" in metrics.text
