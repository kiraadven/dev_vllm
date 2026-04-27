#!/usr/bin/env python3
"""
Decode step interval analysis & binary classification.

Reads the CSV produced by the step profiler in core.py and:
  1. Shows basic statistics of decode-only step intervals.
  2. Trains a simple binary classifier (normal vs long) on the features
     (num_scheduled_tokens, num_requests, has_new_req, prev_interval, etc.).
  3. Outputs classification report + feature importances.
  4. Saves visualisation plots.

Usage:
    python yqn/analyze_step_intervals.py \
        /path/to/step_profiler/decode1.csv \
        /path/to/step_profiler/decode2.csv \
        [--out-dir yqn/figures/step_intervals]

    # Or use defaults (reads from logs_dp_2p2d_nixl/step_profiler/decode1.csv
    # and decode2.csv):
    python yqn/analyze_step_intervals.py

The CSV is expected to have columns:
    step_index, timestamp_mono, interval_ms,
    has_new_req, num_scheduled_tokens, num_requests
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP_PROFILER_DIR = (
    REPO_ROOT / "yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/step_profiler"
)
DEFAULT_CSV_PATHS = (
    DEFAULT_STEP_PROFILER_DIR / "decode1.csv",
    DEFAULT_STEP_PROFILER_DIR / "decode2.csv",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "yqn/figures/step_intervals"


# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop the very first row (interval_ms == 0, no previous step)
    df = df[df["interval_ms"] > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# 2. Statistics
# ---------------------------------------------------------------
def print_stats(df: pd.DataFrame) -> dict:
    decode_df = df[df["has_new_req"] == 0]
    prefill_df = df[df["has_new_req"] == 1]

    print("=" * 60)
    print("         Decode Step Interval Statistics")
    print("=" * 60)
    print(f"Total steps       : {len(df)}")
    print(f"  decode-only     : {len(decode_df)}")
    print(f"  with prefill    : {len(prefill_df)}")
    print()

    if len(decode_df) == 0:
        print("No decode-only steps found. Cannot proceed.")
        sys.exit(1)

    desc = decode_df["interval_ms"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    print("Decode step interval_ms:")
    print(desc.to_string())
    print()

    median = decode_df["interval_ms"].median()
    threshold = 2.0 * median
    n_normal = (decode_df["interval_ms"] <= threshold).sum()
    n_long = len(decode_df) - n_normal
    pct_l = 100.0 * n_long / len(decode_df) if len(decode_df) else 0

    print(f"Classification threshold = 2.0 × median = {threshold:.2f} ms")
    print(f"  normal : {n_normal:>6d} ({100 - pct_l:5.1f}%)")
    print(f"  long   : {n_long:>6d} ({pct_l:5.1f}%)")
    print()

    return {"median": median, "threshold": threshold}


# ---------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------
def build_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add derived features and a binary label."""
    df = df.copy()

    # Label: 1 = long, 0 = normal
    df["label"] = (df["interval_ms"] > threshold).astype(int)

    # Previous step's interval (lag-1)
    df["prev_interval_ms"] = df["interval_ms"].shift(1).fillna(0)

    # Rolling mean of last 5 intervals
    df["rolling_mean_5"] = (
        df["interval_ms"].rolling(window=5, min_periods=1).mean()
    )

    # Delta between current tokens and previous tokens
    df["token_delta"] = df["num_scheduled_tokens"].diff().fillna(0)

    # Delta between current num_requests and previous
    df["req_delta"] = df["num_requests"].diff().fillna(0)

    return df


# ---------------------------------------------------------------
# 4. Train classifier
# ---------------------------------------------------------------
FEATURE_COLS = [
    "has_new_req",
    "num_scheduled_tokens",
    "num_requests",
    "prev_interval_ms",
    "rolling_mean_5",
    "token_delta",
    "req_delta",
]


def train_classifier(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["label"].values

    if y.sum() == 0 or y.sum() == len(y):
        print("Only one class present, skipping classifier training.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=" * 60)
    print("        Classification Report (test set)")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred, target_names=["normal", "long"],
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    # Feature importances
    importances = clf.feature_importances_
    print("Feature importances:")
    for name, imp in sorted(
        zip(FEATURE_COLS, importances), key=lambda x: -x[1]
    ):
        print(f"  {name:<25s} {imp:.4f}")
    print()

    return clf, X_test, y_test, y_pred


# ---------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------
def plot_all(df: pd.DataFrame, stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    threshold = stats["threshold"]

    # --- (a) Time series of interval_ms with threshold line ---
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = np.where(df["label"] == 1, "red", "steelblue")
    ax.scatter(df["step_index"], df["interval_ms"],
               c=colors, s=2, alpha=0.6)
    ax.axhline(y=threshold, color="orange", linestyle="--",
               label=f"threshold = {threshold:.1f} ms")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Interval (ms)")
    ax.set_title("Decode Step Intervals (red = long)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_intervals_timeseries.png"), dpi=150)
    plt.close(fig)

    # --- (b) Histogram ---
    decode_df = df[df["has_new_req"] == 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(decode_df["interval_ms"], bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(x=threshold, color="red", linestyle="--",
               label=f"threshold = {threshold:.1f} ms")
    ax.set_xlabel("Interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Decode Step Interval Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_intervals_hist.png"), dpi=150)
    plt.close(fig)

    # --- (c) Interval vs num_scheduled_tokens scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["num_scheduled_tokens"], df["interval_ms"],
               c=colors, s=4, alpha=0.5)
    ax.set_xlabel("num_scheduled_tokens")
    ax.set_ylabel("Interval (ms)")
    ax.set_title("Interval vs Scheduled Tokens (red = long)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interval_vs_tokens.png"), dpi=150)
    plt.close(fig)

    # --- (d) Feature importance bar chart (if classifier exists) ---
    # Plotted in plot_feature_importance separately

    print(f"Plots saved to {out_dir}/")


def plot_feature_importance(clf, out_dir: str):
    if clf is None:
        return
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(FEATURE_COLS)),
           importances[indices], align="center", alpha=0.8)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=30,
                       ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances (GradientBoosting)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)


def analyze_one_csv(csv_path: Path, out_dir: Path):
    print("=" * 80)
    print(f"Analyzing {csv_path}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    df = load_data(csv_path)
    stats = print_stats(df)

    df = build_features(df, stats["threshold"])

    clf, X_test, y_test, y_pred = train_classifier(df)

    plot_all(df, stats, out_dir)
    plot_feature_importance(clf, out_dir)


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze decode step intervals from vLLM step profiler."
    )
    parser.add_argument(
        "csv_paths", nargs="*", type=Path, default=list(DEFAULT_CSV_PATHS),
        help=(
            "One or more step profiler CSV paths "
            f"(default: {DEFAULT_CSV_PATHS[0]}, {DEFAULT_CSV_PATHS[1]})"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    missing_paths = [path for path in args.csv_paths if not path.is_file()]
    if missing_paths:
        for path in missing_paths:
            print(f"CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

    for csv_path in args.csv_paths:
        print(f"Reading step profiler CSV: {csv_path}")
        analyze_one_csv(csv_path, args.out_dir / csv_path.stem)

    print("Done.")


if __name__ == "__main__":
    main()
