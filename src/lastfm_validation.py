from __future__ import annotations

import json
import os
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from drift_monitor import population_stability_index
from utils import PROJECT_ROOT, ensure_directories, load_csv

SYNTHETIC_PATH = PROJECT_ROOT / "data" / "processed" / "engagement_features.csv"
LASTFM_PATH = PROJECT_ROOT / "data" / "external" / "lastfm_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "external_validation"
FIGURE_DIR = REPORT_DIR / "figures"
COMPARISON_PATH = REPORT_DIR / "synthetic_vs_lastfm.csv"
SUMMARY_PATH = REPORT_DIR / "synthetic_vs_lastfm_summary.json"


def compare_feature(synthetic: pd.DataFrame, real: pd.DataFrame, feature: str) -> dict:
    syn = pd.to_numeric(synthetic[feature], errors="coerce").dropna()
    rel = pd.to_numeric(real[feature], errors="coerce").dropna()
    ks_pvalue = float(ks_2samp(syn, rel).pvalue) if len(syn) and len(rel) else None
    psi = population_stability_index(syn, rel) if len(syn) and len(rel) else None
    return {
        "feature": feature,
        "synthetic_mean": syn.mean(),
        "synthetic_median": syn.median(),
        "synthetic_std": syn.std(),
        "lastfm_mean": rel.mean(),
        "lastfm_median": rel.median(),
        "lastfm_std": rel.std(),
        "ks_pvalue": ks_pvalue,
        "psi": psi,
    }


def plot_outputs(synthetic: pd.DataFrame, real: pd.DataFrame) -> None:
    ensure_directories([FIGURE_DIR])
    plt.figure(figsize=(8, 5))
    plt.hist(synthetic["days_since_last_session"].dropna(), bins=20, alpha=0.55, label="Synthetic", density=True)
    plt.hist(real["days_since_last_session"].dropna(), bins=10, alpha=0.55, label="Last.fm proxy", density=True)
    plt.title("Days Since Last Session: Synthetic vs Last.fm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "days_since_last_session_overlay.png", dpi=160)
    plt.close()

    retention = pd.DataFrame(
        {
            "dataset": ["Synthetic", "Last.fm proxy"],
            "returned_within_7_days": [synthetic["returned_within_7_days"].mean(), real["returned_within_7_days"].mean()],
        }
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=retention, x="dataset", y="returned_within_7_days")
    plt.ylim(0, 1)
    plt.title("7-Day Return Rate Comparison")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "retention_rate_comparison.png", dpi=160)
    plt.close()

    if "engagement_depth" in synthetic.columns and "engagement_depth" in real.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(synthetic["engagement_depth"].dropna(), bins=20, alpha=0.55, label="Synthetic", density=True)
        plt.hist(real["engagement_depth"].dropna(), bins=10, alpha=0.55, label="Last.fm proxy", density=True)
        plt.title("Engagement Depth Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "engagement_depth_distribution.png", dpi=160)
        plt.close()

    time_counts = pd.DataFrame(
        {
            "Synthetic": synthetic["listening_time_of_day"].value_counts(normalize=True),
            "Last.fm proxy": real["listening_time_of_day"].value_counts(normalize=True),
        }
    ).fillna(0).reindex(["morning", "afternoon", "evening", "night"]).fillna(0)
    plt.figure(figsize=(6, 4))
    sns.heatmap(time_counts.T, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Listening Time of Day Share")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "listening_time_of_day_heatmap.png", dpi=160)
    plt.close()


def validate() -> dict:
    ensure_directories([REPORT_DIR, FIGURE_DIR])
    synthetic = load_csv(SYNTHETIC_PATH)
    real = load_csv(LASTFM_PATH)
    shared = sorted(set(synthetic.columns).intersection(real.columns))
    numeric_shared = [feature for feature in shared if pd.api.types.is_numeric_dtype(synthetic[feature]) and pd.api.types.is_numeric_dtype(real[feature])]
    rows = [compare_feature(synthetic, real, feature) for feature in numeric_shared]
    comparison = pd.DataFrame(rows).sort_values("psi", na_position="last")
    comparison.to_csv(COMPARISON_PATH, index=False)
    plot_outputs(synthetic, real)
    summary = {
        "synthetic_rows": int(len(synthetic)),
        "lastfm_proxy_rows": int(len(real)),
        "shared_features": shared,
        "numeric_shared_features": numeric_shared,
        "median_psi": float(comparison["psi"].median()) if not comparison.empty else None,
        "features_with_psi_below_0_1": int((comparison["psi"] < 0.1).sum()) if not comparison.empty else 0,
        "comparison_path": str(COMPARISON_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Top PSI rows:")
    print(comparison.head(10).to_string(index=False))
    return summary


if __name__ == "__main__":
    validate()
