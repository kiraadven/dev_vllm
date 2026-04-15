"""
Base class for dataset distribution analysis and trace generation.

Subclasses must implement:
  - dataset_name (property) → str
  - _load_pairs()           → List[Tuple[int, int]]  (input_len, output_len)

Usage example
─────────────
    analyzer = MyDataset(output_root="experiments")
    stats = analyzer.analyze()          # plots + JSON stats
    analyzer.generate_traces(n_requests=2000)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from transformers import AutoTokenizer

# ── Publication-quality style ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.6,
})

EXPERIMENTS_DIR = Path(__file__).parent


class DatasetAnalyzer(ABC):
    """
    Abstract base for dataset distribution analysis and Poisson-arrival trace generation.

    Directory layout created under *output_root*:
        figures/<dataset_name>/length_distribution.{pdf,png}
        stats/<dataset_name>/distribution_stats.json
        traces/<dataset_name>/<dataset_name>_rate<X>p<Y>.csv
    """

    # ── Class-level configuration (override in subclasses if needed) ──────────
    TOKENIZER_PATH: str = "/data/yqn/Qwen1.5-MoE-A2.7B"
    POISSON_RATES: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5]

    _COLORS = {
        "prompt": "#2166AC",
        "output": "#D6604D",
        "mean":   "#1a9641",
        "median": "#e08000",
        "p99":    "#d7191c",
    }

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self, output_root: Optional[Path | str] = None) -> None:
        root = Path(output_root) if output_root else EXPERIMENTS_DIR
        self.fig_dir   = root / "figures" / self.dataset_name
        self.trace_dir = root / "traces"  / self.dataset_name
        self.stats_dir = root / "stats"   / self.dataset_name
        for d in (self.fig_dir, self.trace_dir, self.stats_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._tokenizer: Optional[AutoTokenizer] = None
        self._pairs: Optional[List[Tuple[int, int]]] = None

    # ── Abstract interface ────────────────────────────────────────────────────
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Unique, filesystem-safe identifier for this dataset."""

    @abstractmethod
    def _load_pairs(self) -> List[Tuple[int, int]]:
        """
        Load the dataset and return a list of (input_tokens, output_tokens) pairs.
        Called once; result is cached in self._pairs.
        """

    # ── Tokenizer (lazy) ─────────────────────────────────────────────────────
    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            print(f"[{self.dataset_name}] Loading tokenizer from {self.TOKENIZER_PATH} …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)
        return self._tokenizer

    # ── Data access (lazy + cached) ───────────────────────────────────────────
    @property
    def pairs(self) -> List[Tuple[int, int]]:
        if self._pairs is None:
            self._pairs = self._load_pairs()
        return self._pairs

    @property
    def prompt_lengths(self) -> np.ndarray:
        return np.array([p for p, _ in self.pairs], dtype=np.int32)

    @property
    def output_lengths(self) -> np.ndarray:
        return np.array([o for _, o in self.pairs], dtype=np.int32)

    # ── Public API ────────────────────────────────────────────────────────────
    def analyze(self) -> dict:
        """
        Compute summary statistics, save them to JSON, and render distribution figures.
        Returns the statistics dictionary.
        """
        stats = self._compute_stats()
        self._save_stats(stats)
        self._plot_distributions(stats)
        self._print_summary(stats)
        return stats

    def generate_traces(
        self,
        n_requests: int = 1000,
        seed: int = 42,
        rates: Optional[List[float]] = None,
    ) -> None:
        """
        For each Poisson arrival rate, sample *n_requests* (input, output) pairs
        and write a CSV trace:  arrive_time, input_tokens_length, output_tokens_length.

        Arrival times follow a Poisson process:
            inter-arrival ~ Exponential(1 / rate)  [seconds]
        """
        if rates is None:
            rates = self.POISSON_RATES

        pairs_arr = np.array(self.pairs, dtype=np.int32)  # (N, 2)
        rng = np.random.default_rng(seed)

        print(f"\n[{self.dataset_name}] Generating traces for rates {rates} req/s …")
        for rate in rates:
            trace = self._build_trace(pairs_arr, n_requests, rate, rng)
            self._save_trace(trace, rate)

            # Auto-select snapshot: 20 min, or full trace if shorter
            total_s    = float(trace["arrive_time"][-1] - trace["arrive_time"][0])
            snapshot_s = 1200.0 if total_s > 1200.0 else None
            self._plot_arrival_rate(
                trace["arrive_time"],
                trace["input_tokens_length"],
                trace["output_tokens_length"],
                label=f"rate{rate:.1f}",
                window_s=5.0,
                snapshot_s=snapshot_s,
            )

    # ── Statistics ────────────────────────────────────────────────────────────
    def _compute_stats(self) -> dict:
        stats: dict = {}
        for name, data in (("prompt", self.prompt_lengths),
                           ("output", self.output_lengths)):
            stats[name] = {
                "count":  int(len(data)),
                "mean":   float(data.mean()),
                "median": float(np.median(data)),
                "p75":    float(np.percentile(data, 75)),
                "p90":    float(np.percentile(data, 90)),
                "p95":    float(np.percentile(data, 95)),
                "p99":    float(np.percentile(data, 99)),
                "min":    int(data.min()),
                "max":    int(data.max()),
            }
        return stats

    def _save_stats(self, stats: dict) -> None:
        path = self.stats_dir / "distribution_stats.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[{self.dataset_name}] Stats saved → {path}")

    def _print_summary(self, stats: dict) -> None:
        for name in ("prompt", "output"):
            s = stats[name]
            print(
                f"[{self.dataset_name}] {name.capitalize():6s}  "
                f"count={s['count']:,}  mean={s['mean']:.1f}  "
                f"median={s['median']:.1f}  p95={s['p95']:.1f}  "
                f"p99={s['p99']:.1f}  max={s['max']}"
            )

    # ── Plotting ──────────────────────────────────────────────────────────────
    @staticmethod
    def _log_bins(data: np.ndarray, cap_pct: float = 99.0, n_bins: int = 40):
        lo = max(1, int(data.min()))
        hi = int(np.percentile(data, cap_pct)) + 1
        return np.logspace(np.log10(lo), np.log10(hi), num=n_bins), hi

    def _draw_hist(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        stats: dict,
        color: str,
        label_prefix: str,
    ) -> None:
        bins, cap = self._log_bins(data)
        clipped = np.clip(data, 1, cap)

        # Histogram bars (PDF)
        counts, edges = np.histogram(clipped, bins=bins)
        widths = np.diff(edges)
        density = counts / (counts.sum() * widths)
        ax.bar(edges[:-1], density, width=widths, align="edge",
               color=color, alpha=0.70, edgecolor="white", linewidth=0.3)

        # Annotation lines: mean, median, p99
        mean_v   = min(stats["mean"],   cap)
        median_v = min(stats["median"], cap)
        p99_v    = min(stats["p99"],    cap)

        ax.axvline(mean_v,   color=self._COLORS["mean"],   ls="--", lw=1.6,
                   label=f"Mean = {stats['mean']:.0f}")
        ax.axvline(median_v, color=self._COLORS["median"], ls="-.", lw=1.6,
                   label=f"Median = {stats['median']:.0f}")
        ax.axvline(p99_v,    color=self._COLORS["p99"],    ls=":",  lw=1.6,
                   label=f"P99 = {stats['p99']:.0f}")

        ax.set_xscale("log")
        ax.set_xlabel("Length (tokens)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{label_prefix} Length Distribution")
        ax.legend(loc="upper right", framealpha=0.8)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    def _draw_cdf(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        stats: dict,
        color: str,
        label_prefix: str,
    ) -> None:
        _, cap = self._log_bins(data)
        clipped = np.clip(data, 1, cap)
        sorted_data = np.sort(clipped)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(sorted_data, cdf, color=color, linewidth=1.8)

        # Annotate median (P50) and P99
        for pct, key, lcolor, ls in (
            (50,  "median", self._COLORS["median"], "-."),
            (99,  "p99",    self._COLORS["p99"],    ":"),
        ):
            val  = stats[key]
            valc = min(val, cap)
            frac = np.searchsorted(sorted_data, valc) / len(sorted_data)

            ax.axvline(valc, color=lcolor, ls=ls, lw=1.2, alpha=0.85)
            ax.axhline(frac, color=lcolor, ls=ls, lw=0.8, alpha=0.5)
            # Offset label upward to avoid crowding
            offset_y = max(0.05, frac - 0.10)
            ax.annotate(
                f"P{pct}={int(val):,}",
                xy=(valc, frac),
                xytext=(valc * 1.25, offset_y),
                fontsize=8, color=lcolor,
                arrowprops=dict(arrowstyle="-", color=lcolor, lw=0.7),
                va="center",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Length (tokens)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label_prefix} Length CDF")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    def _plot_distributions(self, stats: dict) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        (ax_ph, ax_pc), (ax_oh, ax_oc) = axes

        self._draw_hist(ax_ph, self.prompt_lengths, stats["prompt"],
                        self._COLORS["prompt"], "(a) Prompt")
        self._draw_cdf(ax_pc, self.prompt_lengths, stats["prompt"],
                       self._COLORS["prompt"], "(b) Prompt")
        self._draw_hist(ax_oh, self.output_lengths, stats["output"],
                        self._COLORS["output"], "(c) Output")
        self._draw_cdf(ax_oc, self.output_lengths, stats["output"],
                       self._COLORS["output"], "(d) Output")

        fig.suptitle(
            f"{self.dataset_name}: Prompt & Output Token-Length Distributions\n"
            f"Tokenizer: LLaMA-3-8B  ·  {len(self.pairs):,} request pairs",
            fontsize=12, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        stem = self.fig_dir / "length_distribution"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png")
        plt.close(fig)
        print(f"[{self.dataset_name}] Figures saved → {stem}.pdf / .png")

    # ── Sliding-window rate utilities ─────────────────────────────────────────
    @staticmethod
    def _sliding_rate(
        arrive_times: np.ndarray,
        weights: np.ndarray,
        window_s: float,
        n_points: int = 400,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window rate estimator.

        Parameters
        ----------
        arrive_times : sorted array of arrival times (seconds from t = 0)
        weights      : per-request weight (1 for req/s, token count for tok/s)
        window_s     : window width in seconds
        n_points     : number of evaluation points

        Returns
        -------
        t_centers : centre of each evaluation window (seconds)
        rates     : sum(weights in window) / window_s  at each centre
        """
        t0, t1 = arrive_times[0], arrive_times[-1]
        span   = t1 - t0
        # Guard: window must be <= span; use at most half span if too large
        w = min(window_s, span / 2) if span > 0 else window_s

        edges = np.linspace(t0, t1 - w, n_points)
        rates = np.empty(n_points)
        for i, t in enumerate(edges):
            lo = np.searchsorted(arrive_times, t,     side="left")
            hi = np.searchsorted(arrive_times, t + w, side="right")
            rates[i] = weights[lo:hi].sum() / w

        return edges + w / 2, rates

    def _plot_arrival_rate(
        self,
        arrive_times: np.ndarray,
        input_lengths: np.ndarray,
        output_lengths: np.ndarray,
        label: str = "",
        window_s: float = 5.0,
        snapshot_s: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        """
        Plot request arrival rate (req/s) and token arrival rate (tok/s)
        using a sliding window over a snapshot of the trace.

        Parameters
        ----------
        arrive_times   : sorted array of arrival times (seconds from t = 0)
        input_lengths  : input token counts per request
        output_lengths : output token counts per request
        label          : suffix for the output filename
        window_s       : sliding-window width in seconds (default: 5)
        snapshot_s     : length of the randomly-selected snapshot in seconds;
                         None → use the whole trace
        seed           : RNG seed for reproducible snapshot selection
        """
        # ── Select snapshot ───────────────────────────────────────────────────
        if snapshot_s is not None and snapshot_s < (arrive_times[-1] - arrive_times[0]):
            rng       = np.random.default_rng(seed)
            max_start = (arrive_times[-1] - arrive_times[0]) - snapshot_s
            t_start   = float(rng.uniform(0, max_start))
            t_end     = t_start + snapshot_s
            mask          = (arrive_times >= t_start) & (arrive_times < t_end)
            arrive_times  = arrive_times[mask]  - t_start
            input_lengths = input_lengths[mask]
            output_lengths= output_lengths[mask]

        if len(arrive_times) < 2:
            print(f"[{self.dataset_name}] Too few points in snapshot — skipping rate plot.")
            return

        total_tokens = (input_lengths + output_lengths).astype(np.float64)
        ones         = np.ones(len(arrive_times))

        t_req, req_rates = self._sliding_rate(arrive_times, ones,         window_s)
        t_tok, tok_rates = self._sliding_rate(arrive_times, total_tokens, window_s)

        # ── Choose x-axis unit ────────────────────────────────────────────────
        total_s = arrive_times[-1]
        if total_s >= 86_400:
            x_div, x_unit = 86_400, "days"
        elif total_s >= 3_600:
            x_div, x_unit = 3_600, "hours"
        elif total_s >= 60:
            x_div, x_unit = 60, "minutes"
        else:
            x_div, x_unit = 1, "seconds"

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        snap_note = (
            f",  {snapshot_s/60:.0f}-min snapshot" if snapshot_s is not None else ""
        )
        fig.suptitle(
            f"{self.dataset_name}  [{label}]: Arrival Rates  "
            f"(window = {window_s:.0f} s{snap_note})",
            fontsize=12, fontweight="bold",
        )

        # Request rate
        x_req = t_req / x_div
        ax1.fill_between(x_req, req_rates, alpha=0.18, color=self._COLORS["prompt"])
        ax1.plot(x_req, req_rates, color=self._COLORS["prompt"], lw=1.4)
        mean_req = float(req_rates.mean())
        peak_req = float(req_rates.max())
        ax1.axhline(mean_req, color=self._COLORS["mean"], ls="--", lw=1.5,
                    label=f"Mean = {mean_req:.3f} req/s")
        ax1.axhline(peak_req, color=self._COLORS["p99"],  ls=":",  lw=1.5,
                    label=f"Peak = {peak_req:.3f} req/s")
        ax1.set_ylabel("Request Rate (req/s)")
        ax1.legend(loc="upper right", framealpha=0.85)

        # Token rate
        x_tok = t_tok / x_div
        ax2.fill_between(x_tok, tok_rates, alpha=0.18, color=self._COLORS["output"])
        ax2.plot(x_tok, tok_rates, color=self._COLORS["output"], lw=1.4)
        mean_tok = float(tok_rates.mean())
        peak_tok = float(tok_rates.max())
        ax2.axhline(mean_tok, color=self._COLORS["mean"], ls="--", lw=1.5,
                    label=f"Mean = {mean_tok:.1f} tok/s")
        ax2.axhline(peak_tok, color=self._COLORS["p99"],  ls=":",  lw=1.5,
                    label=f"Peak = {peak_tok:.1f} tok/s")
        ax2.set_ylabel("Token Rate (tok/s)")
        ax2.set_xlabel(f"Time ({x_unit})")
        ax2.legend(loc="upper right", framealpha=0.85)

        plt.tight_layout()

        safe_label = label.replace(" ", "_").replace("/", "_")
        stem = self.fig_dir / f"arrival_rate_{safe_label}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png")
        plt.close(fig)
        print(f"[{self.dataset_name}] Arrival rate figure → {stem}.pdf / .png")

    # ── Trace generation ──────────────────────────────────────────────────────
    @staticmethod
    def _build_trace(
        pairs: np.ndarray,
        n_requests: int,
        rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample *n_requests* pairs and assign Poisson-process arrival times.

        Returns a structured NumPy array with fields:
          arrive_time          – float64, seconds since t=0
          input_tokens_length  – int32
          output_tokens_length – int32
        """
        idx = rng.integers(0, len(pairs), size=n_requests)
        sampled = pairs[idx]  # (n_requests, 2)

        # Poisson process: inter-arrivals ~ Exp(1/rate)
        inter = rng.exponential(scale=1.0 / rate, size=n_requests)
        arrive = np.cumsum(inter)

        dtype = np.dtype([
            ("arrive_time",          np.float64),
            ("input_tokens_length",  np.int32),
            ("output_tokens_length", np.int32),
        ])
        trace = np.empty(n_requests, dtype=dtype)
        trace["arrive_time"]          = arrive
        trace["input_tokens_length"]  = sampled[:, 0]
        trace["output_tokens_length"] = sampled[:, 1]
        return trace

    def _save_trace(self, trace: np.ndarray, rate: float) -> None:
        # e.g. sharegpt_gpt4_rate0p5.csv, sharegpt_gpt4_rate1p0.csv …
        rate_tag = f"{rate:.1f}".replace(".", "p")
        fname = f"{self.dataset_name}_rate{rate_tag}.csv"
        path  = self.trace_dir / fname

        np.savetxt(
            path,
            np.column_stack([
                trace["arrive_time"],
                trace["input_tokens_length"],
                trace["output_tokens_length"],
            ]),
            delimiter=",",
            header="arrive_time,input_tokens_length,output_tokens_length",
            comments="",
            fmt=["%.6f", "%d", "%d"],
        )
        print(f"  → {path}  (rate={rate} req/s, n={len(trace):,})")
