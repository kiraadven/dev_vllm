"""
ShareGPT-X dataset analyzer.

Two-stage analysis
──────────────────
1. analyze()
       Whole-dataset (post-2024-01-01) token-length distribution figures
       + distribution_stats.json.

2. plot_day_and_window_rates(seed)
     a. Randomly pick 3 days from [2024-01-01, 2025-08-31].
        For each day: plot the 24-h per-second arrival rate  → 3 figures.
     b. Combine the 3 days into a 72-h timeline.
        Find one 20-min window with HIGH / MEDIUM / LOW request count.
        Plot each window's per-second arrival rate  → 3 figures.

Usage
─────
    analyzer = ShareGPTXDataset()
    analyzer.analyze()                    # whole-dataset distribution
    analyzer.plot_day_and_window_rates()  # 3 daily + 3 snapshot rate plots
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from base_dataset import DatasetAnalyzer

_DEFAULT_JSONL = Path("/data/yqn/ShareGPT-X/ChatGPT-Simple.jsonl")

_TS_MIN = datetime.datetime(2024, 1,  1, tzinfo=datetime.timezone.utc).timestamp()
_TS_MAX = datetime.datetime(2025, 8, 31, tzinfo=datetime.timezone.utc).timestamp()

_LEVEL_COLORS = {
    "low":    "#2166AC",
    "medium": "#e08000",
    "high":   "#D6604D",
}

DAY_S      = 86_400.0   # seconds per day
SNAPSHOT_S =  1_200.0   # 20-minute snapshot window


class ShareGPTXDataset(DatasetAnalyzer):
    """
    Analyzer for the ShareGPT-X dataset (ChatGPT-Simple.jsonl, Door #2).

    Parameters
    ----------
    jsonl_path  : path to ChatGPT-Simple.jsonl
    output_root : root directory for figures / stats
    """

    def __init__(
        self,
        jsonl_path: Optional[Path | str] = None,
        output_root: Optional[Path | str] = None,
        max_input_tokens: int = 32_768,
        max_output_tokens: int = 32_768,
        length_divisor: int = 10,
    ) -> None:
        super().__init__(output_root=output_root)
        if max_input_tokens <= 0 or max_output_tokens <= 0:
            raise ValueError("max_input_tokens/max_output_tokens must be positive.")
        if length_divisor <= 0:
            raise ValueError("length_divisor must be positive.")
        self.jsonl_path = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.length_divisor = int(length_divisor)
        self._arrive_times: Optional[np.ndarray] = None
        self._abs_arrive_times: Optional[np.ndarray] = None

    # ── Abstract property ─────────────────────────────────────────────────────
    @property
    def dataset_name(self) -> str:
        return "sharegpt_x"

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_pairs(self) -> List[Tuple[int, int]]:
        """
        Load all post-2024-01-01 records from ChatGPT-Simple.jsonl.

        Caches:
          self._abs_arrive_times  — absolute Unix timestamps (float64)
          self._arrive_times      — relative seconds from t0   (float64)

        Returns list of (input_tokens, output_tokens) pairs.
        """
        print(f"[{self.dataset_name}] Reading {self.jsonl_path} …")
        print(
            f"[{self.dataset_name}] Token handling: "
            f"input<= {self.max_input_tokens}, output<= {self.max_output_tokens}, "
            f"scaled by 1/{self.length_divisor}."
        )
        tok = self.tokenizer

        records: List[Tuple[float, int, int]] = []

        with open(self.jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                arrive_ts: Optional[float] = None
                input_parts: List[str] = []
                output_parts: List[str] = []

                for turn in item.get("conversations", []):
                    role    = turn.get("role", {}).get("role", "")
                    kind    = turn.get("kind", "")
                    created = turn.get("created")
                    raw     = turn.get("content", {}).get("content", [])
                    text    = " ".join(c for c in raw if isinstance(c, str)).strip()

                    if role == "user" and kind == "text" and text:
                        if arrive_ts is None and created is not None:
                            ts = float(created)
                            if ts > 1e11:   # milliseconds → seconds
                                ts /= 1000.0
                            arrive_ts = ts
                        input_parts.append(text)
                    elif role == "assistant" and kind == "text" and text:
                        output_parts.append(text)

                if arrive_ts is None or not input_parts or not output_parts:
                    continue
                if arrive_ts < _TS_MIN:
                    continue

                in_len = self._scaled_token_count(
                    tok,
                    " ".join(input_parts),
                    max_tokens=self.max_input_tokens,
                )
                out_len = self._scaled_token_count(
                    tok,
                    " ".join(output_parts),
                    max_tokens=self.max_output_tokens,
                )

                if in_len > 0 and out_len > 0:
                    records.append((arrive_ts, in_len, out_len))

        records.sort(key=lambda r: r[0])

        abs_arr = np.array([r[0] for r in records], dtype=np.float64)
        self._abs_arrive_times = abs_arr
        self._arrive_times     = abs_arr - abs_arr[0]

        span_days = self._arrive_times[-1] / DAY_S
        print(
            f"[{self.dataset_name}] Loaded {len(records):,} pairs "
            f"spanning {span_days:.1f} days."
        )
        return [(r[1], r[2]) for r in records]

    def _scaled_token_count(self, tok, text: str, max_tokens: int) -> int:
        """Count tokens with truncation safety, then downscale by length_divisor."""
        encoded = tok(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        raw_len = len(encoded["input_ids"])
        if raw_len <= 0:
            return 0
        # Keep positive lengths after scaling for valid request/response pairs.
        return max(1, (raw_len + self.length_divisor - 1) // self.length_divisor)

    # ── Cached properties ─────────────────────────────────────────────────────
    @property
    def abs_arrive_times(self) -> np.ndarray:
        """Absolute Unix timestamps of arrivals (sorted ascending)."""
        if self._abs_arrive_times is None:
            _ = self.pairs
        return self._abs_arrive_times  # type: ignore[return-value]

    @property
    def arrive_times(self) -> np.ndarray:
        """Relative arrival times (seconds from t = 0, sorted ascending)."""
        if self._arrive_times is None:
            _ = self.pairs
        return self._arrive_times  # type: ignore[return-value]

    # ── Main rate-analysis entry point ────────────────────────────────────────
    def plot_day_and_window_rates(self, seed: int = 42) -> None:
        """
        Step 1 — Randomly pick 3 days from [2024-01-01, 2025-08-31].
                  Plot each day's 24-h per-second arrival rate (3 figures).

        Step 2 — Combine the 3 days into a single 72-h timeline.
                  Find one HIGH / MEDIUM / LOW 20-min window by request count.
                  Plot each window's per-second arrival rate (3 figures).
        """
        abs_ts = self.abs_arrive_times
        rng    = np.random.default_rng(seed)

        # Total integer days available in [_TS_MIN, _TS_MAX)
        n_days_range = int((_TS_MAX - _TS_MIN) / DAY_S)
        chosen_offsets = np.sort(rng.choice(n_days_range, size=3, replace=False))

        day_arrivals: List[np.ndarray] = []   # per-day relative arrival arrays
        day_labels:   List[str]        = []

        print(f"\n[{self.dataset_name}] ── Step 1: 3 random days ──")
        for i, offset in enumerate(chosen_offsets):
            d_start = _TS_MIN + int(offset) * DAY_S
            d_end   = d_start + DAY_S
            mask    = (abs_ts >= d_start) & (abs_ts < d_end)
            rel     = abs_ts[mask] - d_start          # seconds from midnight

            dt       = datetime.datetime.utcfromtimestamp(d_start)
            date_str = dt.strftime("%Y-%m-%d")
            day_labels.append(date_str)
            day_arrivals.append(rel)

            print(f"  Day {i + 1}: {date_str}  n={len(rel):,} requests")
            self._plot_day_rate(rel, date_str, day_idx=i + 1)

        # ── Step 2 ────────────────────────────────────────────────────────────
        print(f"\n[{self.dataset_name}] ── Step 2: 20-min HIGH / MEDIUM / LOW windows ──")
        # Merge into one 72-h timeline: day k → [k*DAY_S, (k+1)*DAY_S)
        combined = np.concatenate([
            arr + k * DAY_S for k, arr in enumerate(day_arrivals)
        ])
        combined.sort()

        if len(combined) < 10:
            print("  Too few combined requests — skipping window plots.")
            return

        self._pick_and_plot_windows(combined)

    # ── Window selection & plotting ───────────────────────────────────────────
    def _pick_and_plot_windows(self, combined: np.ndarray) -> None:
        """
        Divide the 72-h combined timeline into 20-min windows, rank by
        request count, and pick one HIGH / MEDIUM / LOW window each.
        """
        total_s  = float(combined[-1])
        n_wins   = max(1, int(total_s // SNAPSHOT_S))
        win_starts = np.arange(n_wins, dtype=np.float64) * SNAPSHOT_S

        # Request count per window
        win_counts = np.array([
            int(np.searchsorted(combined, t + SNAPSHOT_S, side="right"))
            - int(np.searchsorted(combined, t,            side="left"))
            for t in win_starts
        ], dtype=np.float64)

        nonzero = win_counts[win_counts > 0]
        if len(nonzero) == 0:
            nonzero = win_counts

        p33 = float(np.percentile(nonzero, 33))
        p66 = float(np.percentile(nonzero, 66))

        bands: Dict[str, np.ndarray] = {
            "low":    win_starts[win_counts <= p33],
            "medium": win_starts[(win_counts > p33) & (win_counts <= p66)],
            "high":   win_starts[win_counts > p66],
        }
        fallback_targets = {"low": p33 / 2, "medium": (p33 + p66) / 2, "high": p66 * 1.5}

        rng = np.random.default_rng(0)
        for level, candidates in bands.items():
            if len(candidates) == 0:
                # Fallback: closest window to target count
                t_start = float(win_starts[
                    np.argmin(np.abs(win_counts - fallback_targets[level]))
                ])
            else:
                t_start = float(rng.choice(candidates))

            mask    = (combined >= t_start) & (combined < t_start + SNAPSHOT_S)
            win_arr = combined[mask] - t_start

            print(
                f"  {level.upper():6s}: t={t_start / 3600:.2f} h  "
                f"n={len(win_arr):,} requests"
            )
            self._plot_snapshot_rate(win_arr, level)

    # ── Figure helpers ────────────────────────────────────────────────────────
    def _plot_day_rate(
        self,
        arrive: np.ndarray,   # seconds from midnight, range [0, DAY_S)
        date_str: str,
        day_idx: int,
    ) -> None:
        """Bar chart: per-second request count over one 24-hour day."""
        n_bins = int(DAY_S)   # 86 400 one-second bins
        counts, edges = np.histogram(arrive, bins=n_bins, range=(0.0, DAY_S))
        t_hours = (edges[:-1] + 0.5) / 3600.0   # bin centres in hours

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(t_hours, counts, width=1.0 / 3600.0,
               color="#555599", alpha=0.70, edgecolor="none")

        mean_rate = float(counts.mean())
        peak_rate = float(counts.max())
        ax.axhline(mean_rate, color=self._COLORS["mean"], ls="--", lw=1.5,
                   label=f"Mean = {mean_rate:.2f} req/s")
        ax.axhline(peak_rate, color=self._COLORS["p99"],  ls=":",  lw=1.5,
                   label=f"Peak = {peak_rate:.0f} req/s")

        ax.set_xlabel("Time (hours, UTC)")
        ax.set_ylabel("Request Rate (req/s)")
        ax.set_xlim(0, 24)
        ax.set_title(
            f"{self.dataset_name}: 1-s Arrival Rate  —  {date_str}\n"
            f"{len(arrive):,} requests",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right", framealpha=0.85)
        plt.tight_layout()

        stem = self.fig_dir / f"day{day_idx}_rate_{date_str}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png")
        plt.close(fig)
        print(f"  → {stem}.png")

    def _plot_snapshot_rate(
        self,
        arrive: np.ndarray,   # seconds from window start
        level: str,
    ) -> None:
        """Bar chart: per-second request count over one 20-min snapshot."""
        if len(arrive) < 2:
            print(f"  [{level}] Too few requests — skipping snapshot plot.")
            return

        n_bins = int(SNAPSHOT_S)   # 1 200 one-second bins
        counts, edges = np.histogram(arrive, bins=n_bins, range=(0.0, SNAPSHOT_S))
        t_centers = edges[:-1] + 0.5

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(t_centers, counts, width=1.0,
               color=_LEVEL_COLORS[level], alpha=0.75, edgecolor="none")

        mean_rate = float(counts.mean())
        peak_rate = float(counts.max())
        ax.axhline(mean_rate, color=self._COLORS["mean"], ls="--", lw=1.5,
                   label=f"Mean = {mean_rate:.2f} req/s")
        ax.axhline(peak_rate, color=self._COLORS["p99"],  ls=":",  lw=1.5,
                   label=f"Peak = {peak_rate:.0f} req/s")

        ax.set_xlabel("Time (seconds from window start)")
        ax.set_ylabel("Request Rate (req/s)")
        ax.set_title(
            f"{self.dataset_name}  [{level} workload]: 1-s Arrival Rate  "
            f"(20-min snapshot)\n{len(arrive):,} requests",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right", framealpha=0.85)
        plt.tight_layout()

        stem = self.fig_dir / f"snapshot_rate_{level}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png")
        plt.close(fig)
        print(f"  → {stem}.png")

