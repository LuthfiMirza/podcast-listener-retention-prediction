from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_PATH = RAW_DIR / "interaction_logs.csv"

CONTENT_TYPES = ["podcast", "radio", "audiobook", "live", "video"]
CONTENT_TYPE_WEIGHTS = [0.45, 0.25, 0.15, 0.10, 0.05]
GENRES = ["comedy", "religion", "education", "lifestyle", "horror", "true_crime", "news", "music"]
GENRE_WEIGHTS = [0.22, 0.17, 0.15, 0.13, 0.10, 0.09, 0.08, 0.06]
TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
TIME_WEIGHTS = [0.20, 0.18, 0.42, 0.20]
DEVICES = ["mobile", "desktop", "smart_speaker"]
DEVICE_WEIGHTS = [0.82, 0.10, 0.08]
SUBSCRIPTIONS = ["free", "premium_monthly", "premium_yearly"]
SUBSCRIPTION_WEIGHTS = [0.72, 0.20, 0.08]

# Calibration note:
# These synthetic distributions were adjusted with Last.fm HetRec 2011 as a public
# listening-behavior proxy. Last.fm is not Noice production data and does not
# contain native podcast completion or skip semantics, so calibration targets
# behavioral realism rather than exact distribution matching.


def _episode_duration(content_type: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    options = {
        "podcast": ([30, 45, 60, 90], [0.74, 0.14, 0.09, 0.03]),
        "radio": ([15, 30, 45, 60], [0.08, 0.78, 0.10, 0.04]),
        "audiobook": ([30, 45, 60, 90], [0.58, 0.16, 0.16, 0.10]),
        "live": ([30, 60, 90], [0.76, 0.16, 0.08]),
        "video": ([15, 30, 45], [0.12, 0.78, 0.10]),
    }
    return np.array([rng.choice(options[value][0], p=options[value][1]) for value in content_type])


def _days_since_last_session(n_users: int, rng: np.random.Generator) -> np.ndarray:
    segment = rng.choice(["active", "warm", "dormant"], n_users, p=[0.54, 0.31, 0.15])
    days = np.zeros(n_users, dtype=int)
    active = segment == "active"
    warm = segment == "warm"
    dormant = segment == "dormant"
    days[active] = rng.integers(0, 8, active.sum())
    days[warm] = rng.integers(8, 31, warm.sum())
    days[dormant] = rng.integers(31, 121, dormant.sum())
    return days


def _engagement_segment(n_users: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(["casual", "regular", "power"], n_users, p=[0.25, 0.45, 0.30])


def generate_synthetic_interactions(n_users: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = np.arange(1, n_users + 1)

    content_type = rng.choice(CONTENT_TYPES, n_users, p=CONTENT_TYPE_WEIGHTS)
    genre = rng.choice(GENRES, n_users, p=GENRE_WEIGHTS)
    segment = _engagement_segment(n_users, rng)
    episode_duration_minutes = _episode_duration(content_type, rng)
    subscription_status = rng.choice(SUBSCRIPTIONS, n_users, p=SUBSCRIPTION_WEIGHTS)
    is_paid_user = subscription_status != "free"
    is_premium_content = rng.binomial(1, np.where(is_paid_user, 0.34, 0.10))

    completion_rate = np.empty(n_users)
    completion_rate[segment == "casual"] = rng.beta(4.5, 2.2, (segment == "casual").sum())
    completion_rate[segment == "regular"] = rng.beta(7.5, 1.7, (segment == "regular").sum())
    completion_rate[segment == "power"] = rng.beta(12.0, 1.25, (segment == "power").sum())
    content_boost = np.select(
        [content_type == "podcast", content_type == "audiobook", content_type == "radio", content_type == "live"],
        [0.05, 0.04, -0.03, 0.02],
        default=-0.05,
    )
    genre_boost = np.select(
        [genre == "comedy", genre == "religion", genre == "education", genre == "horror", genre == "true_crime"],
        [0.04, 0.04, 0.03, 0.02, 0.02],
        default=0.0,
    )
    completion_rate = np.clip(completion_rate + content_boost + genre_boost + rng.normal(0, 0.055, n_users), 0, 1)
    skip_rate = np.clip(0.70 - 0.78 * completion_rate + rng.normal(0, 0.09, n_users), 0, 1)

    sessions = rng.negative_binomial(n=2, p=0.085, size=n_users) + 1
    sessions += np.where(segment == "regular", rng.poisson(14, n_users), 0)
    sessions += np.where(segment == "power", rng.poisson(42, n_users), 0)
    sessions = np.clip(sessions, 1, 220).astype(int)

    content_variety_score = rng.lognormal(mean=3.05, sigma=1.05, size=n_users)
    content_variety_score *= np.where(segment == "power", 1.9, np.where(segment == "regular", 1.25, 0.75))
    content_variety_score = np.clip(content_variety_score, 1, 220).astype(int)

    replay_count = np.clip(rng.poisson(0.10 + 0.55 * completion_rate + 0.012 * np.sqrt(sessions)), 0, 5).astype(int)
    replay_rate = np.clip(replay_count / 5, 0, 1)
    total_interactions = sessions + replay_count + content_variety_score + rng.poisson(4 + 10 * completion_rate, n_users)
    engagement_depth = np.log1p(total_interactions)

    next_episode_played = rng.binomial(1, np.clip(0.10 + 0.45 * completion_rate + 0.07 * engagement_depth - 0.22 * skip_rate, 0.02, 0.95))
    creator_followed = rng.binomial(1, np.clip(0.06 + 0.18 * completion_rate + 0.08 * replay_rate + 0.025 * engagement_depth, 0.02, 0.72))
    liked_content = rng.binomial(1, np.clip(0.72 + 0.24 * completion_rate + 0.015 * engagement_depth, 0.20, 0.995))
    shared_content = rng.binomial(1, np.clip(0.04 + 0.12 * completion_rate + 0.06 * liked_content, 0.01, 0.42))
    commented = rng.binomial(1, np.clip(0.02 + 0.08 * completion_rate + 0.05 * creator_followed, 0.01, 0.32))

    days_since_last_session = _days_since_last_session(n_users, rng)
    listening_time_of_day = rng.choice(TIME_OF_DAY, n_users, p=TIME_WEIGHTS)
    device_type = rng.choice(DEVICES, n_users, p=DEVICE_WEIGHTS)
    peak_hour_activity = np.isin(listening_time_of_day, ["evening", "night"]).astype(int)

    recency_component = -0.010 * days_since_last_session
    retention_score = (
        1.35
        + 0.105 * np.sqrt(sessions)
        + 0.085 * np.log1p(content_variety_score)
        + 0.42 * engagement_depth
        + 1.55 * completion_rate
        - 1.40 * skip_rate
        + 0.36 * replay_rate
        + 0.42 * creator_followed
        + 0.34 * next_episode_played
        + 0.18 * is_premium_content * completion_rate
        + 0.16 * (subscription_status == "premium_yearly")
        + recency_component
        + rng.normal(0, 0.55, n_users)
    )
    return_probability = 1 / (1 + np.exp(-retention_score))
    returned_within_7_days = rng.binomial(1, return_probability)
    returned_within_3_days = returned_within_7_days.astype(int)

    avg_session_duration = rng.lognormal(mean=3.85, sigma=0.42, size=n_users)
    avg_session_duration *= 0.95 + completion_rate + np.where(segment == "power", 0.22, 0)
    avg_session_duration = np.clip(avg_session_duration, 2, 140)

    ad_completion_rate = np.where(
        subscription_status == "free",
        np.clip(completion_rate - 0.10 * skip_rate + rng.normal(0, 0.16, n_users), 0, 1),
        np.nan,
    )
    engaged = ((completion_rate >= 0.70) | (returned_within_7_days == 1)).astype(int)

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "content_type": content_type,
            "genre": genre,
            "episode_duration_minutes": episode_duration_minutes,
            "sessions": sessions,
            "avg_session_duration": avg_session_duration.round(2),
            "completion_rate": completion_rate.round(4),
            "skip_rate": skip_rate.round(4),
            "replay_count": replay_count,
            "replay_rate": replay_rate.round(4),
            "next_episode_played": next_episode_played,
            "creator_followed": creator_followed,
            "liked_content": liked_content,
            "shared_content": shared_content,
            "commented": commented,
            "is_premium_content": is_premium_content,
            "subscription_status": subscription_status,
            "ad_completion_rate": np.round(ad_completion_rate, 4),
            "listening_time_of_day": listening_time_of_day,
            "device_type": device_type,
            "content_variety_score": content_variety_score,
            "days_since_last_session": days_since_last_session,
            "peak_hour_activity": peak_hour_activity,
            "returned_within_3_days": returned_within_3_days,
            "returned_within_7_days": returned_within_7_days.astype(int),
            "engaged": engaged.astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibrated Noice-style synthetic listener data.")
    parser.add_argument("--n-users", type=int, default=2500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_interactions(n_users=args.n_users, seed=args.random_state)
    data.to_csv(args.output, index=False)
    print(f"Generated {len(data):,} calibrated Noice-style listener rows at {args.output}")


if __name__ == "__main__":
    main()
