from __future__ import annotations

import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, create_features, split_features_target


def test_create_features_shape_and_values():
    data = pd.DataFrame(
        {
            "sessions": [4],
            "avg_session_duration": [20.0],
            "skip_rate": [0.75],
            "replay_rate": [0.1],
            "content_variety_score": [3],
            "days_since_last_session": [3],
            "completion_rate": [0.25],
            "peak_hour_activity": [1],
            "engaged": [0],
        }
    )
    featured = create_features(data)
    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert featured.loc[0, "engagement_intensity"] == 20.0
    assert featured.loc[0, "skip_completion_gap"] == 0.5
    assert featured.loc[0, "high_skip_user"] == 1
    assert featured.loc[0, "low_completion_user"] == 1
    X, y = split_features_target(featured)
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert y.tolist() == [0]
