from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import PROJECT_ROOT, ensure_directories

LASTFM_DIR = PROJECT_ROOT / "data" / "external" / "lastfm"
REPORT_DIR = PROJECT_ROOT / "reports" / "external_validation"
PROFILE_PATH = REPORT_DIR / "lastfm_profile.json"
MAX_HISTORY_ROWS = 100_000
MAX_PLAYS_ROWS = 200_000
FILES = {
    "timestamp": "userid-timestamp-artid-artname-traid-traname.tsv",
    "plays": "usersha1-artmbid-artname-plays.tsv",
    "userid_profile": "userid-profile.csv",
    "usersha1_profile_csv": "usersha1-profile.csv",
    "usersha1_profile_tsv": "usersha1-profile.tsv",
}
COLUMN_NAMES = {
    "timestamp": ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
    "plays": ["user_sha1", "artist_id", "artist_name", "plays"],
    "userid_profile": ["user_id", "gender", "age", "country", "registered"],
    "usersha1_profile_csv": ["user_sha1", "gender", "age", "country", "registered"],
    "usersha1_profile_tsv": ["user_sha1", "gender", "age", "country", "registered"],
}


def detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:100_000]
    for encoding in ["utf-8", "latin-1"]:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def separator_for(path: Path) -> str:
    return "\t" if path.suffix == ".tsv" else ","


def manual_read(path: Path, label: str, max_rows: int | None = None) -> tuple[pd.DataFrame, int, int]:
    encoding = detect_encoding(path)
    delimiter = separator_for(path)
    rows = []
    malformed = 0
    total_seen = 0
    expected = len(COLUMN_NAMES[label])
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row_number, row in enumerate(reader):
            if label == "userid_profile" and row_number == 0 and row and row[0] == "#id":
                continue
            total_seen += 1
            if len(row) != expected:
                malformed += 1
                if len(row) > expected:
                    row = row[: expected - 1] + [delimiter.join(row[expected - 1 :])]
                else:
                    row = row + [None] * (expected - len(row))
            if max_rows is None or len(rows) < max_rows:
                rows.append(row)
            if max_rows is not None and len(rows) >= max_rows and label in {"timestamp", "plays"}:
                break
    return pd.DataFrame(rows, columns=COLUMN_NAMES[label]), total_seen, malformed


def basic_summary(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_rows = MAX_HISTORY_ROWS if label == "timestamp" else (MAX_PLAYS_ROWS if label == "plays" else None)
    sample, rows_seen, malformed = manual_read(path, label, max_rows=max_rows)
    summary = {
        "path": str(path),
        "encoding": detect_encoding(path),
        "separator": "tab" if separator_for(path) == "\t" else "comma",
        "shape": [int(rows_seen), int(sample.shape[1])],
        "shape_note": "sample-limited for large file" if max_rows is not None else "full loaded file",
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "sampled_rows_for_stats": int(len(sample)),
        "malformed_rows_seen": int(malformed),
        "columns": list(sample.columns),
        "dtypes_sample": {column: str(dtype) for column, dtype in sample.dtypes.items()},
        "nulls_sample": {column: int(value) for column, value in sample.replace("", pd.NA).isna().sum().items()},
        "sample_rows": sample.head(5).replace("", "<NA>").fillna("<NA>").to_dict(orient="records"),
    }
    return sample, summary


def timestamp_summary(data: pd.DataFrame) -> dict[str, Any]:
    parsed = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    valid = data.assign(_timestamp=parsed).dropna(subset=["_timestamp"])
    per_user = valid.groupby("user_id").size()
    return {
        "note": f"Stats computed on first {len(data):,} rows for interactive inspection.",
        "parsed_timestamp_nulls_sample": int(parsed.isna().sum()),
        "date_range_sample": [str(valid["_timestamp"].min()), str(valid["_timestamp"].max())] if not valid.empty else [None, None],
        "active_users_sample": int(valid["user_id"].nunique()),
        "unique_artists_sample": int(valid["artist_name"].nunique(dropna=True)),
        "total_listens_sample": int(len(valid)),
        "tracks_per_user_stats_sample": {key: float(value) for key, value in per_user.describe().to_dict().items()},
    }


def plays_summary(data: pd.DataFrame) -> dict[str, Any]:
    plays = pd.to_numeric(data["plays"], errors="coerce")
    per_user = data.assign(_plays=plays).groupby("user_sha1")["_plays"].sum()
    top_artists = data.assign(_plays=plays).groupby("artist_name", dropna=True)["_plays"].sum().sort_values(ascending=False).head(10)
    return {
        "note": f"Stats computed on first {len(data):,} rows.",
        "play_count_distribution_sample": {key: float(value) for key, value in plays.describe().to_dict().items()},
        "top_artists_sample": {str(index): int(value) for index, value in top_artists.items()},
        "median_plays_per_user_sample": float(per_user.median()),
        "users_sample": int(data["user_sha1"].nunique()),
        "artists_sample": int(data["artist_name"].nunique(dropna=True)),
    }


def profile_summary(data: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["age_distribution"] = {name: float(value) for name, value in pd.to_numeric(data["age"], errors="coerce").describe().to_dict().items()}
    for column in ["gender", "country"]:
        result[f"{column}_distribution"] = {str(index): int(value) for index, value in data[column].replace("", pd.NA).value_counts(dropna=False).head(20).items()}
    return result


def inspect_all() -> dict[str, Any]:
    ensure_directories([REPORT_DIR])
    profile: dict[str, Any] = {"files": {}}
    for label, filename in FILES.items():
        path = LASTFM_DIR / filename
        if not path.exists():
            profile["files"][label] = {"missing": True, "path": str(path)}
            continue
        data, summary = basic_summary(path, label)
        if label == "timestamp":
            summary["timestamp_summary"] = timestamp_summary(data)
        elif label == "plays":
            summary["plays_summary"] = plays_summary(data)
        elif "profile" in label:
            summary["profile_summary"] = profile_summary(data)
        profile["files"][label] = summary
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, default=str))
    return profile


def print_summary(profile: dict[str, Any]) -> None:
    print(f"Saved summary: {PROFILE_PATH}")
    for label, summary in profile["files"].items():
        print(f"\n=== {label} ===")
        if summary.get("missing"):
            print(f"MISSING: {summary['path']}")
            continue
        print(f"path: {summary['path']}")
        print(f"encoding: {summary['encoding']} | separator: {summary['separator']}")
        print(f"shape: {summary['shape']} ({summary['shape_note']}) | sampled_rows_for_stats: {summary['sampled_rows_for_stats']} | malformed_rows_seen: {summary['malformed_rows_seen']} | file_size_mb: {summary['file_size_mb']}")
        print(f"columns: {summary['columns']}")
        print(f"dtypes_sample: {summary['dtypes_sample']}")
        print(f"nulls_sample: {summary['nulls_sample']}")
        print("sample rows:")
        print(pd.DataFrame(summary["sample_rows"]).to_string(index=False))
        for extra_key in ["timestamp_summary", "plays_summary", "profile_summary"]:
            if extra_key in summary:
                print(f"{extra_key}:")
                print(json.dumps(summary[extra_key], indent=2, default=str))


def main() -> None:
    profile = inspect_all()
    print_summary(profile)


if __name__ == "__main__":
    main()
