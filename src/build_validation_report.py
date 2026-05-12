from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT, ensure_directories

REPORT_DIR = PROJECT_ROOT / "reports" / "external_validation"
PROFILE_PATH = REPORT_DIR / "lastfm_profile.json"
COMPARISON_PATH = REPORT_DIR / "synthetic_vs_lastfm.csv"
SUMMARY_PATH = REPORT_DIR / "synthetic_vs_lastfm_summary.json"
OUTPUT_PATH = REPORT_DIR / "lastfm_report.html"

MAPPING_ROWS = [
    ("timestamp.user_id", "user_id", "Listener identifier for session aggregation"),
    ("timestamp.timestamp", "sessions, days_since_last_session, returned_within_7_days, listening_time_of_day", "Temporal listening behavior proxy"),
    ("timestamp.artist_name", "content_variety_score", "Unique artists as content variety proxy"),
    ("plays.plays", "replay_count, creator_followed, engagement_depth", "Repeated artist plays as affinity/depth proxy"),
    ("userid-profile age/gender/country", "age, gender, country", "Demographic analysis only"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def classify_similarity(comparison: pd.DataFrame) -> tuple[str, list[str]]:
    if comparison.empty or comparison["psi"].dropna().empty:
        return "not enough comparable features", ["Generate Last.fm proxy features first."]
    low = int((comparison["psi"] < 0.1).sum())
    medium = int(((comparison["psi"] >= 0.1) & (comparison["psi"] <= 0.2)).sum())
    high = int((comparison["psi"] > 0.2).sum())
    if low >= max(3, high):
        finding = "Synthetic data is a partial but useful approximation for selected engagement signals."
    else:
        finding = "Synthetic data is not yet a close approximation across all Last.fm proxy features."
    recommendations = []
    high_rows = comparison.sort_values("psi", ascending=False).head(5)
    for _, row in high_rows.iterrows():
        recommendations.append(f"Adjust synthetic feature '{row['feature']}' because PSI={row['psi']:.3f} vs Last.fm proxy.")
    if high:
        recommendations.append("Increase Last.fm sampling coverage because the current timestamp sample is sorted by user and small at user level.")
    return finding, recommendations


def build_report() -> dict[str, str]:
    ensure_directories([REPORT_DIR])
    profile = load_json(PROFILE_PATH)
    summary = load_json(SUMMARY_PATH)
    comparison = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else pd.DataFrame()
    finding, recommendations = classify_similarity(comparison)

    timestamp = profile.get("files", {}).get("timestamp", {}).get("timestamp_summary", {})
    overview = {
        "users": timestamp.get("active_users_sample"),
        "date_range": timestamp.get("date_range_sample"),
        "total_listens": timestamp.get("total_listens_sample"),
        "lastfm_proxy_rows": summary.get("lastfm_proxy_rows"),
        "sample_note": timestamp.get("note"),
    }
    mapping_table = pd.DataFrame(MAPPING_ROWS, columns=["Last.fm source", "Project schema", "Purpose"]).to_html(index=False)
    comparison_table = comparison.sort_values("psi", na_position="last").head(20).to_html(index=False, float_format="{:.4f}".format) if not comparison.empty else "<p>No comparison data.</p>"
    rec_items = "".join(f"<li>{item}</li>" for item in recommendations)

    OUTPUT_PATH.write_text(
        f"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Last.fm External Validation Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#111827}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f3f4f6}}.note{{background:#fff7ed;border:1px solid #fed7aa;padding:12px;border-radius:6px}}</style>
</head><body>
<h1>Last.fm External Validation Report</h1>
<h2>Dataset Overview</h2>
<ul>
<li>Users in inspected timestamp sample: {overview['users']}</li>
<li>Date range sample: {overview['date_range']}</li>
<li>Total listens inspected: {overview['total_listens']}</li>
<li>Generated Last.fm proxy rows: {overview['lastfm_proxy_rows']}</li>
</ul>
<p class="note">{overview['sample_note']} The timestamp file is large and sorted by user, so this validation is a public-data sanity check, not a full replacement for real Noice data.</p>
<h2>Column Mapping</h2>
{mapping_table}
<h2>Distribution Comparison</h2>
{comparison_table}
<h2>Key Finding</h2>
<p>{finding}</p>
<h2>Recommendations</h2>
<ul>{rec_items}</ul>
<h2>Figures</h2>
<ul>
<li><a href="figures/days_since_last_session_overlay.png">Days since last session overlay</a></li>
<li><a href="figures/retention_rate_comparison.png">Retention rate comparison</a></li>
<li><a href="figures/engagement_depth_distribution.png">Engagement depth distribution</a></li>
<li><a href="figures/listening_time_of_day_heatmap.png">Listening time heatmap</a></li>
</ul>
</body></html>
""".strip()
    )
    print(json.dumps({"report": str(OUTPUT_PATH), "finding": finding, "recommendations": recommendations}, indent=2))
    return {"report": str(OUTPUT_PATH)}


if __name__ == "__main__":
    build_report()
