from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

FIELD_SPECS: dict[str, dict[str, Any]] = {
    "sessions": {"type": "integer", "minimum": 0, "required": True, "description": "Number of user sessions."},
    "avg_session_duration": {"type": "number", "minimum": 0, "required": True, "description": "Average session duration in minutes."},
    "skip_rate": {"type": "number", "minimum": 0, "maximum": 1, "required": True, "description": "Skipped content ratio."},
    "replay_rate": {"type": "number", "minimum": 0, "maximum": 1, "required": True, "description": "Replay interaction ratio."},
    "content_variety_score": {"type": "integer", "minimum": 0, "required": True, "description": "Unique content/genre variety score."},
    "days_since_last_session": {"type": "integer", "minimum": 0, "required": True, "description": "Days since the user was last active."},
    "completion_rate": {"type": "number", "minimum": 0, "maximum": 1, "required": True, "description": "Average content completion ratio."},
    "peak_hour_activity": {"type": "integer", "minimum": 0, "maximum": 1, "required": True, "description": "Whether the user is active during peak hours."},
}


class PredictRequest(BaseModel):
    sessions: int = Field(..., ge=0)
    avg_session_duration: float = Field(..., ge=0)
    skip_rate: float = Field(..., ge=0, le=1)
    replay_rate: float = Field(..., ge=0, le=1)
    content_variety_score: int = Field(..., ge=0)
    days_since_last_session: int = Field(..., ge=0)
    completion_rate: float = Field(..., ge=0, le=1)
    peak_hour_activity: int = Field(..., ge=0, le=1)

    @field_validator("skip_rate", "replay_rate", "completion_rate")
    @classmethod
    def validate_rate(cls, value: float, info):
        if not 0 <= value <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return value

    @field_validator("peak_hour_activity")
    @classmethod
    def validate_binary(cls, value: int):
        if value not in (0, 1):
            raise ValueError("peak_hour_activity must be either 0 or 1")
        return value


class BatchPredictRequest(BaseModel):
    rows: list[PredictRequest] = Field(..., min_length=1, max_length=1000)
