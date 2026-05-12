from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, ensure_directories, load_csv

REQUIRED_COLUMNS = {
    "user_id",
    "sessions",
    "avg_session_duration",
    "skip_rate",
    "replay_rate",
    "content_variety_score",
    "days_since_last_session",
    "completion_rate",
    "peak_hour_activity",
    "engaged",
}


def preprocess_interactions(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates(subset=["user_id"])
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "user_id"]
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=list(REQUIRED_COLUMNS))

    rate_columns = ["skip_rate", "replay_rate", "completion_rate"]
    cleaned[rate_columns] = cleaned[rate_columns].clip(0, 1)
    cleaned["days_since_last_session"] = cleaned["days_since_last_session"].clip(lower=0)
    cleaned["peak_hour_activity"] = cleaned["peak_hour_activity"].astype(int).clip(0, 1)
    cleaned["engaged"] = cleaned["engaged"].astype(int).clip(0, 1)
    return cleaned.sort_values("user_id").reset_index(drop=True)


def main() -> None:
    raw_data = load_csv(RAW_DATA_PATH)
    processed = preprocess_interactions(raw_data)
    ensure_directories([PROCESSED_DATA_PATH.parent])
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
