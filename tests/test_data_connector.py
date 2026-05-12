from __future__ import annotations

import pandas as pd

from src.data_connector import CSVConnector, apply_schema_mapping, detect_schema


def test_csv_connector_and_schema_alias_mapping(tmp_path):
    path = tmp_path / "users.csv"
    pd.DataFrame(
        {
            "session_count": [3],
            "avg_duration": [12.5],
            "skip_ratio": [0.2],
            "replay_ratio": [0.1],
            "unique_genres": [5],
            "days_inactive": [2],
            "completion": [0.8],
            "is_peak_hour": [1],
            "target": [1],
        }
    ).to_csv(path, index=False)
    data = CSVConnector(path).load()
    schema = detect_schema(list(data.columns))
    mapped = apply_schema_mapping(data, schema)
    assert schema["missing_required"] == []
    assert "avg_session_duration" in mapped.columns
    assert mapped.loc[0, "engaged"] == 1


def test_build_connector_csv_and_missing_source_args(tmp_path):
    import argparse
    import pytest
    from src.data_connector import build_connector, CSVConnector

    csv_args = argparse.Namespace(source="csv", path=tmp_path / "x.csv", connection_uri=None, query=None, project=None)
    assert isinstance(build_connector(csv_args), CSVConnector)
    pg_args = argparse.Namespace(source="postgresql", path=None, connection_uri=None, query=None, project=None)
    with pytest.raises(ValueError):
        build_connector(pg_args)
    bq_args = argparse.Namespace(source="bigquery", path=None, connection_uri=None, query=None, project=None)
    with pytest.raises(ValueError):
        build_connector(bq_args)
