# Podcast Listener At-Risk Retention Prediction API

FastAPI service for scoring listener churn/intervention risk from behavioral features.

By default, the API loads the at-risk model:

- `experiments/risk_best_model.pkl`
- `experiments/risk_best_threshold.json`

If the risk model is unavailable, it falls back to the legacy retention-likelihood model:

- `experiments/best_model.pkl`
- `experiments/best_threshold.json`

## Run Locally

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --reload --port 8000
```

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/sample_request.json
```

Example response fields:

```json
{
  "risk_probability": 0.87,
  "threshold": 0.41,
  "predicted_at_risk": 1,
  "risk_segment": "high_risk",
  "recommended_action": "personalized content recommendation and push notification",
  "model_target": "at_risk_user"
}
```

## Health and Metrics

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```
