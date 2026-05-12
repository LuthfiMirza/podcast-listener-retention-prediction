from __future__ import annotations

import argparse
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering import FEATURE_COLUMNS
from utils import PROJECT_ROOT, ensure_directories

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
SCHEMA_PATH = EXPERIMENT_DIR / "column_schema.json"
BASE_FEATURES = [
    "sessions",
    "avg_session_duration",
    "skip_rate",
    "replay_rate",
    "content_variety_score",
    "days_since_last_session",
    "completion_rate",
    "peak_hour_activity",
]
OPTIONAL_TARGETS = ["engaged", "returned_within_3_days", "returned_within_7_days", "user_id"]
EXPECTED_COLUMNS = BASE_FEATURES + OPTIONAL_TARGETS
ALIASES = {
    "sessions": ["session_count", "num_sessions", "total_sessions", "session"],
    "avg_session_duration": ["average_session_duration", "avg_duration", "session_duration", "duration"],
    "skip_rate": ["skip_ratio", "skips_rate", "skip"],
    "replay_rate": ["replay_ratio", "replays_rate", "replay"],
    "content_variety_score": ["variety_score", "content_variety", "unique_genres", "genre_count"],
    "days_since_last_session": ["recency", "days_inactive", "last_seen_days", "days_since_active"],
    "completion_rate": ["complete_rate", "avg_completion", "completion", "completion_ratio"],
    "peak_hour_activity": ["is_peak_hour", "peak_activity", "peak_hour"],
    "engaged": ["label", "target", "is_engaged", "engagement_label"],
    "user_id": ["userid", "customer_id", "account_id"],
}


class DataConnector(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load source data as a pandas DataFrame."""


class CSVConnector(DataConnector):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Missing CSV file: {self.path}")
        return pd.read_csv(self.path)


class PostgreSQLConnector(DataConnector):
    def __init__(self, connection_uri: str, query: str):
        self.connection_uri = connection_uri
        self.query = query

    def load(self) -> pd.DataFrame:
        from sqlalchemy import create_engine

        engine = create_engine(self.connection_uri)
        with engine.connect() as connection:
            return pd.read_sql(self.query, connection)


class BigQueryConnector(DataConnector):
    def __init__(self, query: str, project: str | None = None):
        self.query = query
        self.project = project

    def load(self) -> pd.DataFrame:
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install google-cloud-bigquery to use BigQueryConnector.") from exc

        client = bigquery.Client(project=self.project)
        return client.query(self.query).to_dataframe()


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def detect_schema(columns: list[str]) -> dict[str, Any]:
    normalized = {normalize_column(column): column for column in columns}
    mapping: dict[str, str | None] = {}

    for expected in EXPECTED_COLUMNS:
        candidates = [expected, *ALIASES.get(expected, [])]
        match = next((normalized.get(normalize_column(candidate)) for candidate in candidates if normalize_column(candidate) in normalized), None)
        mapping[expected] = match

    missing_required = [column for column in BASE_FEATURES if mapping[column] is None]
    return {"mapping": mapping, "missing_required": missing_required, "source_columns": columns}


def apply_schema_mapping(data: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    rename_map = {source: target for target, source in schema["mapping"].items() if source is not None}
    mapped = data.rename(columns=rename_map)
    required = [column for column in BASE_FEATURES if column in mapped.columns]
    optional = [column for column in OPTIONAL_TARGETS if column in mapped.columns]
    return mapped[required + optional]


def build_connector(args: argparse.Namespace) -> DataConnector:
    if args.source == "csv":
        return CSVConnector(args.path)
    if args.source == "postgresql":
        if not args.connection_uri or not args.query:
            raise ValueError("PostgreSQL source requires --connection-uri and --query.")
        return PostgreSQLConnector(args.connection_uri, args.query)
    if args.source == "bigquery":
        if not args.query:
            raise ValueError("BigQuery source requires --query.")
        return BigQueryConnector(args.query, project=args.project)
    raise ValueError(f"Unsupported source: {args.source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load real data and infer project column schema.")
    parser.add_argument("--source", choices=["csv", "postgresql", "bigquery"], required=True)
    parser.add_argument("--path", help="CSV file path.")
    parser.add_argument("--connection-uri", help="SQLAlchemy PostgreSQL connection URI.")
    parser.add_argument("--query", help="SQL query for PostgreSQL or BigQuery.")
    parser.add_argument("--project", help="BigQuery project ID.")
    args = parser.parse_args()

    connector = build_connector(args)
    data = connector.load()
    schema = detect_schema(list(data.columns))
    preview = apply_schema_mapping(data, schema)
    ensure_directories([EXPERIMENT_DIR])
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(json.dumps({"rows": len(data), "mapped_columns": list(preview.columns), "schema_path": str(SCHEMA_PATH), **schema}, indent=2))


if __name__ == "__main__":
    main()
