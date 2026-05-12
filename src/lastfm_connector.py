from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import create_features
from inspect_lastfm import COLUMN_NAMES, LASTFM_DIR, detect_encoding
from utils import PROJECT_ROOT, ensure_directories

OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "lastfm_features.csv"
TIMESTAMP_PATH = LASTFM_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"
PLAYS_PATH = LASTFM_DIR / "usersha1-artmbid-artname-plays.tsv"
PROFILE_PATH = LASTFM_DIR / "userid-profile.csv"


def read_timestamp_chunks(chunksize: int):
    return pd.read_csv(
        TIMESTAMP_PATH,
        sep="\t",
        encoding=detect_encoding(TIMESTAMP_PATH),
        header=None,
        names=COLUMN_NAMES["timestamp"],
        chunksize=chunksize,
        on_bad_lines="skip",
        low_memory=False,
    )


def read_plays_chunks(chunksize: int = 250_000, max_chunks: int | None = None):
    iterator = pd.read_csv(
        PLAYS_PATH,
        sep="\t",
        encoding=detect_encoding(PLAYS_PATH),
        header=None,
        names=COLUMN_NAMES["plays"],
        chunksize=chunksize,
        on_bad_lines="skip",
        low_memory=False,
    )
    for index, chunk in enumerate(iterator):
        if max_chunks is not None and index >= max_chunks:
            break
        yield chunk


def bucket_time(hour: int) -> str:
    if 0 <= hour <= 6:
        return "night"
    if 7 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def collect_diverse_history(target_users: int, chunksize: int, random_state: int, max_chunks: int | None = None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(random_state)
    selected_users: set[str] = set()
    frames = []
    chunks_processed = 0
    planned_chunks = max_chunks or 80
    users_per_chunk = max(1, int(np.ceil(target_users / planned_chunks)))

    for chunk_index, chunk in enumerate(read_timestamp_chunks(chunksize)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        chunks_processed += 1
        chunk = chunk.dropna(subset=["user_id"])
        candidates = [user for user in chunk["user_id"].unique() if user not in selected_users]
        remaining = target_users - len(selected_users)
        if remaining <= 0:
            break
        if not candidates:
            continue
        take = min(users_per_chunk, remaining, len(candidates))
        chosen = set(pd.Series(candidates).sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))).tolist())
        selected_users.update(chosen)
        subset = chunk[chunk["user_id"].isin(chosen)].copy()
        subset["timestamp"] = pd.to_datetime(subset["timestamp"], errors="coerce", utc=True)
        subset = subset.dropna(subset=["timestamp", "user_id"])
        if not subset.empty:
            frames.append(subset[["user_id", "timestamp", "artist_name"]])
        if len(selected_users) >= target_users:
            break

    if not frames:
        raise RuntimeError("No timestamp events sampled. Increase --max-chunks or reduce --target-users.")
    history = pd.concat(frames, ignore_index=True)
    stats = {
        "timestamp_file": str(TIMESTAMP_PATH),
        "chunks_processed": chunks_processed,
        "sampled_users_count": len(selected_users),
        "total_listening_events_used": int(len(history)),
        "date_range": [str(history["timestamp"].min()), str(history["timestamp"].max())],
    }
    return history, stats


def aggregate_history(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    history["listen_date"] = history["timestamp"].dt.floor("D")
    history["hour"] = history["timestamp"].dt.hour
    history["listening_time_of_day"] = history["hour"].map(bucket_time)
    max_date = history["listen_date"].max()

    daily = history.groupby(["user_id", "listen_date"]).size().rename("daily_listens").reset_index()
    daily = daily.sort_values(["user_id", "listen_date"])
    daily["next_listen_date"] = daily.groupby("user_id")["listen_date"].shift(-1)
    daily["returned_pair"] = ((daily["next_listen_date"] - daily["listen_date"]).dt.days <= 7).fillna(False).astype(int)

    user_daily = daily.groupby("user_id").agg(
        sessions=("daily_listens", "mean"),
        returned_within_7_days=("returned_pair", "max"),
        listening_days=("listen_date", "nunique"),
    )
    user_summary = history.groupby("user_id").agg(
        total_listens=("timestamp", "size"),
        last_listen=("listen_date", "max"),
        content_variety_score=("artist_name", "nunique"),
        listening_time_of_day=("listening_time_of_day", lambda value: value.mode().iat[0] if not value.mode().empty else "evening"),
    )
    features = user_summary.join(user_daily)
    features["days_since_last_session"] = (max_date - features["last_listen"]).dt.days.clip(lower=0)
    return features.reset_index()


def build_play_features(target_users: int, random_state: int) -> pd.DataFrame:
    frames = []
    for chunk in read_plays_chunks(max_chunks=8):
        frames.append(chunk)
        if sum(len(frame) for frame in frames) >= max(100_000, target_users * 100):
            break
    plays = pd.concat(frames, ignore_index=True)
    plays["plays"] = pd.to_numeric(plays["plays"], errors="coerce").fillna(0)
    median_artist_plays = plays["plays"].median()
    grouped = plays.groupby("user_sha1")["plays"].agg(total_plays="sum", median_artist_plays="median", max_artist_plays="max").reset_index()
    grouped = grouped.sample(frac=1, random_state=random_state).head(target_users).reset_index(drop=True)
    grouped["replay_count"] = np.clip((grouped["median_artist_plays"] > median_artist_plays).astype(int) * 3, 0, 5)
    threshold = grouped["max_artist_plays"].quantile(0.80)
    grouped["creator_followed"] = (grouped["max_artist_plays"] >= threshold).astype(int)
    grouped["engagement_depth"] = np.log1p(grouped["total_plays"])
    return grouped


def load_profiles() -> pd.DataFrame:
    if not PROFILE_PATH.exists():
        return pd.DataFrame()
    profiles = pd.read_csv(PROFILE_PATH, encoding=detect_encoding(PROFILE_PATH))
    if "#id" in profiles.columns:
        profiles = profiles.rename(columns={"#id": "user_id"})
    return profiles[[column for column in ["user_id", "age", "gender", "country"] if column in profiles.columns]]


def build_lastfm_features(target_users: int, chunksize: int, random_state: int, max_chunks: int | None = None) -> tuple[pd.DataFrame, dict]:
    history, stats = collect_diverse_history(target_users, chunksize, random_state, max_chunks=max_chunks)
    data = aggregate_history(history)
    play_features = build_play_features(len(data), random_state)
    profiles = load_profiles()

    play_repeated = play_features.iloc[np.arange(len(data)) % len(play_features)].reset_index(drop=True)
    for column in ["replay_count", "creator_followed", "engagement_depth", "total_plays"]:
        data[column] = play_repeated[column].values
    if not profiles.empty:
        data = data.merge(profiles, on="user_id", how="left")

    data["completion_rate"] = np.clip(0.35 + 0.055 * data["engagement_depth"] + 0.006 * data["sessions"], 0, 1)
    data["skip_rate"] = np.clip(1 - data["completion_rate"] + np.where(data["returned_within_7_days"] == 1, -0.12, 0.12), 0, 1)
    data["replay_rate"] = np.clip(data["replay_count"] / 5, 0, 1)
    data["avg_session_duration"] = np.clip(8 + data["sessions"] * 2.2 + data["completion_rate"] * 42, 2, 140)
    data["episode_duration_minutes"] = 30
    data["content_type"] = "music"
    data["genre"] = "music"
    data["next_episode_played"] = data["returned_within_7_days"]
    data["liked_content"] = (data["completion_rate"] > 0.65).astype(int)
    data["shared_content"] = (data["engagement_depth"] > data["engagement_depth"].quantile(0.75)).astype(int)
    data["commented"] = (data["engagement_depth"] > data["engagement_depth"].quantile(0.90)).astype(int)
    data["is_premium_content"] = 0
    data["subscription_status"] = "free"
    data["ad_completion_rate"] = np.nan
    data["peak_hour_activity"] = data["listening_time_of_day"].isin(["evening", "night"]).astype(int)
    data["returned_within_3_days"] = data["returned_within_7_days"]
    data["engaged"] = ((data["completion_rate"] >= 0.70) | (data["returned_within_7_days"] == 1)).astype(int)
    data = create_features(data)
    stats.update({"final_users_count": int(len(data)), "output_path": str(OUTPUT_PATH), "output_shape": [int(data.shape[0]), int(data.shape[1])]})
    return data, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Last.fm proxy features for external validation.")
    parser.add_argument("--target-users", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    ensure_directories([OUTPUT_PATH.parent])
    features, stats = build_lastfm_features(args.target_users, args.chunksize, args.random_state, args.max_chunks)
    features.to_csv(OUTPUT_PATH, index=False)
    print("Last.fm proxy feature generation complete")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
