from __future__ import annotations

import numpy as np
import pandas as pd

from utils import PROCESSED_DATA_PATH, ensure_directories, load_csv

FEATURE_COLUMNS = [
    "sessions",
    "avg_session_duration",
    "skip_rate",
    "replay_rate",
    "content_variety_score",
    "days_since_last_session",
    "completion_rate",
    "peak_hour_activity",
    "engagement_intensity",
    "skip_completion_gap",
    "recency_risk_score",
    "high_skip_user",
    "low_completion_user",
    "listen_completion_ratio",
    "engagement_depth",
    "is_power_listener",
    "content_affinity_score",
    "time_of_day_encoded",
]
TARGET_COLUMN = "returned_within_7_days"


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    features["engagement_intensity"] = (
        features["avg_session_duration"] * features["sessions"] / (features["days_since_last_session"] + 1)
    )
    features["skip_completion_gap"] = features["skip_rate"] - features["completion_rate"]
    features["recency_risk_score"] = np.log1p(features["days_since_last_session"]) * (1 + features["skip_rate"])
    features["high_skip_user"] = (features["skip_rate"] >= 0.60).astype(int)
    features["low_completion_user"] = (features["completion_rate"] < 0.40).astype(int)

    if "episode_duration_minutes" not in features.columns:
        features["episode_duration_minutes"] = features["avg_session_duration"].clip(lower=1)
    for column in ["liked_content", "shared_content", "commented", "creator_followed", "replay_count"]:
        if column not in features.columns:
            features[column] = 0
    if "content_type" not in features.columns:
        features["content_type"] = "podcast"
    if "listening_time_of_day" not in features.columns:
        features["listening_time_of_day"] = "evening"

    affinity_weights = {"podcast": 1.00, "audiobook": 0.92, "live": 0.82, "radio": 0.68, "video": 0.55}
    time_mapping = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    features["listen_completion_ratio"] = features["completion_rate"] * features["episode_duration_minutes"]
    features["engagement_depth"] = features[["liked_content", "shared_content", "commented", "creator_followed"]].sum(axis=1)
    features["is_power_listener"] = ((features["completion_rate"] > 0.80) & (features["replay_count"] > 1)).astype(int)
    features["content_affinity_score"] = features["content_type"].map(affinity_weights).fillna(0.60) * (1 + features["completion_rate"])
    features["time_of_day_encoded"] = features["listening_time_of_day"].map(time_mapping).fillna(2).astype(int)
    return features


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]


def main() -> None:
    data = load_csv(PROCESSED_DATA_PATH)
    featured = create_features(data)
    ensure_directories([PROCESSED_DATA_PATH.parent])
    featured.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved feature-engineered data to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
