from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from api.schemas import BatchPredictRequest, FIELD_SPECS, PredictRequest  # noqa: E402
from feature_engineering import FEATURE_COLUMNS, create_features  # noqa: E402
from utils import ensure_directories  # noqa: E402

RISK_MODEL_PATH = PROJECT_ROOT / "experiments" / "risk_best_model.pkl"
RISK_THRESHOLD_PATH = PROJECT_ROOT / "experiments" / "risk_best_threshold.json"
RISK_METADATA_PATH = PROJECT_ROOT / "experiments" / "risk_model_metadata.json"
BEST_MODEL_PATH = PROJECT_ROOT / "experiments" / "best_model.pkl"
BEST_THRESHOLD_PATH = PROJECT_ROOT / "experiments" / "best_threshold.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "experiments" / "model_metadata.json"
BATCH_RESULTS_DIR = PROJECT_ROOT / "reports" / "batch_results"

app = FastAPI(title="Podcast Listener At-Risk Retention Prediction API", version="2.0.0")
PREDICTION_COUNTER = 0
BATCH_COUNTER = 0


class PredictionResponse(BaseModel):
    risk_probability: float
    threshold: float
    predicted_at_risk: int
    risk_segment: str
    recommended_action: str
    model_target: str
    engagement_probability: Optional[float] = None
    predicted_label: Optional[int] = None


class BatchPredictionResponse(BaseModel):
    status: str
    rows: int
    result_file: Optional[str] = None
    predictions: Optional[list[PredictionResponse]] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": errors})


def active_paths() -> dict[str, Path | str]:
    if RISK_MODEL_PATH.exists():
        return {
            "model": RISK_MODEL_PATH,
            "threshold": RISK_THRESHOLD_PATH,
            "metadata": RISK_METADATA_PATH,
            "target": "at_risk_user",
        }
    return {
        "model": BEST_MODEL_PATH,
        "threshold": BEST_THRESHOLD_PATH,
        "metadata": MODEL_METADATA_PATH,
        "target": "returned_within_7_days",
    }


def load_model() -> dict[str, Any]:
    paths = active_paths()
    model_path = paths["model"]
    if not isinstance(model_path, Path) or not model_path.exists():
        raise FileNotFoundError("Missing model artifact. Run tuning first.")
    return joblib.load(model_path)


def load_metadata() -> dict[str, Any]:
    paths = active_paths()
    metadata_path = paths["metadata"]
    if not isinstance(metadata_path, Path) or not metadata_path.exists():
        return {"model_version": "unknown", "auc": None, "target": paths["target"]}
    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("target", paths["target"])
    return metadata


def load_threshold() -> float:
    paths = active_paths()
    threshold_path = paths["threshold"]
    if isinstance(threshold_path, Path) and threshold_path.exists():
        payload = json.loads(threshold_path.read_text())
        return float(payload.get("threshold", 0.5))
    return float(load_metadata().get("threshold", 0.5))


def classify_risk(probability: float, threshold: float) -> str:
    if probability >= 0.75:
        return "high_risk"
    if probability >= threshold:
        return "medium_risk"
    return "low_risk"


def recommended_action(segment: str) -> str:
    actions = {
        "high_risk": "personalized content recommendation and push notification",
        "medium_risk": "softer content nudge or creator recommendation",
        "low_risk": "no immediate intervention",
    }
    return actions[segment]


def score_rows(rows: list[PredictRequest]) -> list[PredictionResponse]:
    artifact = load_model()
    threshold = load_threshold()
    target = str(active_paths()["target"])
    frame = pd.DataFrame([row.model_dump() for row in rows])
    features = create_features(frame)[FEATURE_COLUMNS]
    probabilities = artifact["model"].predict_proba(features)[:, 1]
    responses = []
    for probability in probabilities:
        probability = float(probability)
        segment = classify_risk(probability, threshold)
        predicted = int(probability >= threshold)
        response = PredictionResponse(
            risk_probability=round(probability, 6),
            threshold=round(threshold, 4),
            predicted_at_risk=predicted,
            risk_segment=segment,
            recommended_action=recommended_action(segment),
            model_target=target,
            engagement_probability=round(probability, 6) if target != "at_risk_user" else None,
            predicted_label=predicted if target != "at_risk_user" else None,
        )
        responses.append(response)
    return responses


def save_batch_results(predictions: list[PredictionResponse]) -> Path:
    ensure_directories([BATCH_RESULTS_DIR])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = BATCH_RESULTS_DIR / f"batch_predictions_{timestamp}.csv"
    pd.DataFrame([prediction.model_dump() for prediction in predictions]).to_csv(output_path, index=False)
    return output_path


def process_batch(rows: list[PredictRequest]) -> None:
    predictions = score_rows(rows)
    save_batch_results(predictions)


@app.get("/health")
def health() -> dict[str, Any]:
    metadata = load_metadata()
    return {
        "status": "ok",
        "model_version": metadata.get("model_version", "unknown"),
        "model_target": metadata.get("target", active_paths()["target"]),
        "threshold": load_threshold(),
        "auc": metadata.get("auc"),
        "model_path": str(active_paths()["model"]),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {"fields": FIELD_SPECS, "max_batch_size": 1000}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    metadata = load_metadata()
    metric_prefix = "listener_retention"
    return "\n".join([
        f"{metric_prefix}_model_auc {metadata.get('auc', 0) or 0}",
        f"{metric_prefix}_threshold {load_threshold()}",
        f"{metric_prefix}_predictions_total {PREDICTION_COUNTER}",
        f"{metric_prefix}_batches_total {BATCH_COUNTER}",
    ]) + "\n"


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    global PREDICTION_COUNTER
    try:
        prediction = score_rows([payload])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    PREDICTION_COUNTER += 1
    return prediction


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_v1(payload: PredictRequest) -> PredictionResponse:
    return predict(payload)


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(payload: Union[BatchPredictRequest, list[PredictRequest]], background_tasks: BackgroundTasks) -> BatchPredictionResponse:
    global BATCH_COUNTER, PREDICTION_COUNTER
    rows = payload.rows if isinstance(payload, BatchPredictRequest) else payload
    if not rows:
        raise HTTPException(status_code=400, detail="Batch cannot be empty.")
    if len(rows) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 rows.")
    BATCH_COUNTER += 1
    PREDICTION_COUNTER += len(rows)
    try:
        if len(rows) > 100:
            background_tasks.add_task(process_batch, rows)
            return BatchPredictionResponse(status="accepted", rows=len(rows), result_file=str(BATCH_RESULTS_DIR))
        predictions = score_rows(rows)
        output_path = save_batch_results(predictions)
        return BatchPredictionResponse(status="completed", rows=len(rows), result_file=str(output_path), predictions=predictions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
