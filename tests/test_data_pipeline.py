from __future__ import annotations

import pandas as pd

from src.data_pipeline import preprocess_interactions


def sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "sessions": [5, 5, 3, None],
            "avg_session_duration": [20.0, 20.0, 10.0, 5.0],
            "skip_rate": [1.2, 1.2, -0.1, 0.2],
            "replay_rate": [0.1, 0.1, 0.2, 0.3],
            "content_variety_score": [4, 4, 2, 1],
            "days_since_last_session": [2, 2, -3, 1],
            "completion_rate": [0.8, 0.8, 1.5, 0.4],
            "peak_hour_activity": [1, 1, 2, 0],
            "engaged": [1, 1, 0, 1],
        }
    )


def test_preprocess_drops_nulls_duplicates_and_clips_values():
    cleaned = preprocess_interactions(sample_raw())
    assert len(cleaned) == 2
    assert cleaned["user_id"].tolist() == [1, 2]
    assert cleaned["skip_rate"].between(0, 1).all()
    assert cleaned["completion_rate"].between(0, 1).all()
    assert cleaned["days_since_last_session"].min() >= 0
    assert cleaned["peak_hour_activity"].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(cleaned["engaged"])



def test_preprocess_raises_for_missing_columns():
    import pytest
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_interactions(pd.DataFrame({"user_id": [1]}))
