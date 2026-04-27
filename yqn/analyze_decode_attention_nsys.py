#!/usr/bin/env python
"""
Decode Attention NVTX Profiling Analyzer
=========================================
Reads Nsight Systems SQLite exports, extracts ``decode_attn`` NVTX ranges
produced by ``qwen2_moe.py``, and generates comprehensive matplotlib charts.

Usage:
    python analyze_decode_attention_nsys.py [inputs ...] [--output-root DIR]

Each input can be a ``.sqlite`` file or a directory containing them.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    REPO_ROOT / "yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "yqn/figures/decode_attention_nsys"
NVTX_TABLE_CANDIDATES = ("NVTX_EVENTS", "NVTX_PUSHPOP_EVENTS")
LABEL_PATTERN = re.compile(
    r"(step|layer|rank|local_rank|num_reqs|num_tokens|req_key_kind|layer_name|req_keys|qlens|slens)"
    r"=((?:\[[^\]]*\])|(?:\S+))",
)

# ---------------------------------------------------------------------------
# Matplotlib global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ===================================================================
# Data classes
# ===================================================================
@dataclass
class AttnEvent:
    """One NVTX ``decode_attn`` range from the profiler."""
    label: str
    start_ns: int
    end_ns: int
    step: int
    layer: int
    layer_name: str
    rank: str
    local_rank: str
    num_reqs: int
    num_tokens: int
    req_key_kind: str
    req_keys: list[str]
    qlens: list[int]
    slens: list[int]

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


@dataclass
class RequestLayerStat:
    total_shared_ns: float = 0.0
    samples: int = 0

    @property
    def avg_ms(self) -> float:
        return (self.total_shared_ns / self.samples / 1e6) if self.samples else 0.0


@dataclass
class RequestSummary:
    request_key: str
    first_start_ns: int
    total_shared_ns: float = 0.0
    samples: int = 0
    unique_steps: set[int] = field(default_factory=set)
    unique_layers: set[int] = field(default_factory=set)
    mean_qlen_acc: float = 0.0
    mean_slen_acc: float = 0.0
    lens_samples: int = 0

    def add(self, *, start_ns: int, step: int, layer: int,
            shared_ns: float, qlen: int | None, slen: int | None) -> None:
        self.total_shared_ns += shared_ns
        self.samples += 1
        self.unique_steps.add(step)
        self.unique_layers.add(layer)
        if start_ns < self.first_start_ns:
            self.first_start_ns = start_ns
        if qlen is not None:
            self.mean_qlen_acc += qlen
            self.lens_samples += 1
        if slen is not None:
            self.mean_slen_acc += slen

    @property
    def avg_attn_per_layer_ms(self) -> float:
        return (self.total_shared_ns / self.samples / 1e6) if self.samples else 0.0

    @property
    def total_attn_ms(self) -> float:
        return self.total_shared_ns / 1e6

    @property
    def mean_qlen(self) -> float | None:
        return (self.mean_qlen_acc / self.lens_samples) if self.lens_samples else None

    @property
    def mean_slen(self) -> float | None:
        return (self.mean_slen_acc / self.lens_samples) if self.lens_samples else None


# ===================================================================
# CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze decode attention NVTX profiling data from Nsight SQLite exports.",
    )
    p.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS),
                   help="Files/directories containing .nsys-rep or .sqlite files.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Directory for generated figures and CSVs.")
    p.add_argument("--nsys-bin", default="nsys",
                   help="Path to the nsys executable.")
    p.add_argument("--force-export", action="store_true",
                   help="Re-export .nsys-rep even if .sqlite already exists.")
    p.add_argument("--skip-export", action="store_true",
                   help="Only analyze existing .sqlite files.")
    p.add_argument("--max-requests", type=int, default=60,
                   help="Max requests to show in per-request plots.")
    p.add_argument("--sort-by", choices=("first_seen", "avg_attn", "total_attn"),
                   default="first_seen",
                   help="How to order requests in plots.")
    return p.parse_args()


# ===================================================================
# SQLite helpers — reading NVTX data
# ===================================================================
def discover_inputs(inputs: Iterable[Path]) -> list[Path]:
    chosen: dict[Path, Path] = {}
    for raw in inputs:
        path = raw.resolve()
        if not path.exists():
            print(f"Skipping missing: {raw}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix in {".sqlite", ".nsys-rep"}:
            key = path.with_suffix("")
            chosen[key] = _prefer(chosen.get(key), path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix in {".sqlite", ".nsys-rep"}:
                    key = child.with_suffix("")
                    chosen[key] = _prefer(chosen.get(key), child)
    return sorted(chosen.values())


def _prefer(current: Path | None, candidate: Path) -> Path:
    if current is None:
        return candidate
    return candidate if candidate.suffix == ".nsys-rep" else current


def ensure_sqlite(source: Path, nsys_bin: str, force: bool, skip: bool) -> Path:
    if source.suffix == ".sqlite":
        return source
    sqlite_path = source.with_suffix(".sqlite")
    if sqlite_path.exists() and not force:
        return sqlite_path
    if skip:
        raise FileNotFoundError(f"Missing SQLite for {source} (--skip-export)")
    cmd = [nsys_bin, "export", "--type", "sqlite", "--force-overwrite", "true",
           "--quiet", "true", "--output", str(sqlite_path), str(source)]
    print(f"[export] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return sqlite_path


def _first_table(conn: sqlite3.Connection, candidates: Iterable[str]) -> str | None:
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in candidates:
        if t in tables:
            return t
    return None


def _name_expr(conn: sqlite3.Connection, table: str, alias: str = "t") -> tuple[str, str]:
    cols = {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    # PRIORITY 1: Direct text columns — most reliable, avoids broken
    # StringIds joins when textId is present but always NULL.
    for c in ("text", "name"):
        if c in cols and "INT" not in str(cols[c][2]).upper():
            return f"{alias}.{c}", ""
    # PRIORITY 2: Integer text-id columns that need join to StringIds
    for c in ("textId", "nameId", "shortName", "shortNameId"):
        r = cols.get(c)
        if r and str(r[2]).upper() == "INTEGER":
            return (f"COALESCE(s.value, CAST({alias}.{c} AS TEXT))",
                    f"LEFT JOIN StringIds s ON {alias}.{c} = s.id")
    for c in ("text", "name"):
        if c in cols:
            return f"CAST({alias}.{c} AS TEXT)", ""
    return "'<unnamed>'", ""


def read_events(conn: sqlite3.Connection, nvtx_table: str) -> list[AttnEvent]:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({nvtx_table})").fetchall()}
    if "start" not in cols or "end" not in cols:
        return []
    expr, join = _name_expr(conn, nvtx_table)
    sql = f"""
        SELECT label, start, end FROM (
            SELECT {expr} AS label, t.start AS start, t.end AS end
            FROM {nvtx_table} t {join}
        ) WHERE label LIKE 'decode_attn %' ORDER BY start
    """
    events: list[AttnEvent] = []
    for label, s, e in conn.execute(sql).fetchall():
        ev = _parse_label(str(label), int(s), int(e))
        if ev:
            events.append(ev)
    return events


# ===================================================================
# Parsing
# ===================================================================
def _parse_label(label: str, start_ns: int, end_ns: int) -> AttnEvent | None:
    if not label.startswith("decode_attn "):
        return None
    fields = dict(LABEL_PATTERN.findall(label))
    req_keys = _list_str(fields.get("req_keys", "[]"))
    qlens = _list_int(fields.get("qlens", "[]"))
    slens = _list_int(fields.get("slens", "[]"))
    return AttnEvent(
        label=label, start_ns=start_ns, end_ns=end_ns,
        step=_int(fields.get("step"), -1),
        layer=_int(fields.get("layer"), -1),
        layer_name=fields.get("layer_name", "unknown"),
        rank=fields.get("rank", "na"),
        local_rank=fields.get("local_rank", "na"),
        num_reqs=_int(fields.get("num_reqs"), len(req_keys)),
        num_tokens=_int(fields.get("num_tokens"), 0),
        req_key_kind=fields.get("req_key_kind", "unknown"),
        req_keys=req_keys, qlens=qlens, slens=slens,
    )


def _list_str(v: str) -> list[str]:
    v = v.strip()
    if len(v) < 2 or v[0] != "[" or v[-1] != "]":
        return []
    body = v[1:-1]
    return [x for x in body.split(",") if x] if body else []


def _list_int(v: str) -> list[int]:
    out: list[int] = []
    for x in _list_str(v):
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def _int(v: str | None, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ===================================================================
# Statistics aggregation
# ===================================================================
def expand_stats(events: list[AttnEvent]) -> tuple[
    dict[str, RequestSummary],
    dict[tuple[str, int], RequestLayerStat],
]:
    reqs: dict[str, RequestSummary] = {}
    layer_stats: dict[tuple[str, int], RequestLayerStat] = defaultdict(RequestLayerStat)

    for ev in events:
        n = max(1, ev.num_reqs, len(ev.req_keys))
        share = ev.duration_ns / n
        keys = ev.req_keys or [f"slot_{i}" for i in range(n)]
        for i in range(n):
            rk = keys[i] if i < len(keys) else f"slot_{i}"
            ql = ev.qlens[i] if i < len(ev.qlens) else None
            sl = ev.slens[i] if i < len(ev.slens) else None
            if rk not in reqs:
                reqs[rk] = RequestSummary(request_key=rk, first_start_ns=ev.start_ns)
            reqs[rk].add(start_ns=ev.start_ns, step=ev.step, layer=ev.layer,
                         shared_ns=share, qlen=ql, slen=sl)
            layer_stats[(rk, ev.layer)].total_shared_ns += share
            layer_stats[(rk, ev.layer)].samples += 1
    return reqs, layer_stats


def sort_requests(reqs: list[RequestSummary], by: str) -> list[RequestSummary]:
    if by == "avg_attn":
        return sorted(reqs, key=lambda r: (-r.avg_attn_per_layer_ms, r.first_start_ns))
    if by == "total_attn":
        return sorted(reqs, key=lambda r: (-r.total_attn_ms, r.first_start_ns))
    return sorted(reqs, key=lambda r: (r.first_start_ns, r.request_key))


def _short_key(key: str, maxlen: int = 16) -> str:
    if len(key) <= maxlen:
        return key
    return key[:6] + ".." + key[-6:]


# ===================================================================
# CSV output
# ===================================================================
def write_events_csv(path: Path, events: list[AttnEvent]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_ns", "end_ns", "duration_ns", "duration_ms",
                     "step", "layer", "layer_name", "num_reqs", "num_tokens",
                     "req_key_kind", "req_keys", "qlens", "slens"])
        for ev in events:
            w.writerow([ev.start_ns, ev.end_ns, ev.duration_ns,
                        f"{ev.duration_ms:.6f}", ev.step, ev.layer,
                        ev.layer_name, ev.num_reqs, ev.num_tokens,
                        ev.req_key_kind, ",".join(ev.req_keys),
                        ",".join(str(x) for x in ev.qlens),
                        ",".join(str(x) for x in ev.slens)])


def write_request_csv(path: Path, reqs: list[RequestSummary]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_key", "total_attn_ms", "avg_attn_per_layer_ms",
                     "samples", "unique_steps", "unique_layers",
                     "mean_qlen", "mean_slen"])
        for r in reqs:
            w.writerow([r.request_key, f"{r.total_attn_ms:.6f}",
                        f"{r.avg_attn_per_layer_ms:.6f}", r.samples,
                        len(r.unique_steps), len(r.unique_layers),
                        "" if r.mean_qlen is None else f"{r.mean_qlen:.1f}",
                        "" if r.mean_slen is None else f"{r.mean_slen:.1f}"])


def write_request_layer_csv(path: Path,
                            layer_stats: dict[tuple[str, int], RequestLayerStat],
                            reqs: list[RequestSummary]) -> None:
    idx = {r.request_key: r for r in reqs}
    rows = []
    for (rk, ly), st in layer_stats.items():
        rows.append((idx[rk].first_start_ns, rk, ly, st.samples,
                      st.total_shared_ns / 1e6, st.avg_ms))
    rows.sort()
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_key", "layer", "samples", "total_attn_ms", "avg_attn_ms"])
        for _, rk, ly, sa, total, avg in rows:
            w.writerow([rk, ly, sa, f"{total:.6f}", f"{avg:.6f}"])


# ===================================================================
# Chart 1: Per-request average attention per layer (bar chart)
# ===================================================================
def plot_request_avg_attn_per_layer(fig_dir: Path,
                                    reqs: list[RequestSummary]) -> Path:
    """Bar chart — x=request, y=average attention time per layer invocation (ms)."""
    path = fig_dir / "01_request_avg_attn_per_layer.png"
    labels = [_short_key(r.request_key) for r in reqs]
    values = [r.avg_attn_per_layer_ms for r in reqs]

    fig, ax = plt.subplots(figsize=(max(10, len(reqs) * 0.35), 6))
    bars = ax.bar(range(len(reqs)), values, color=COLORS[0], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Avg Attention Time per Layer (ms)")
    ax.set_title("Per-Request Average Decode Attention Time per Layer")
    ax.set_xlabel("Request")

    # annotate max
    if values:
        max_idx = int(np.argmax(values))
        ax.annotate(f"{values[max_idx]:.3f}ms", xy=(max_idx, values[max_idx]),
                    fontsize=7, ha="center", va="bottom", color="red")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 2: Per-request total attention time (bar chart)
# ===================================================================
def plot_request_total_attn(fig_dir: Path,
                            reqs: list[RequestSummary]) -> Path:
    """Bar chart — x=request, y=total attributed attention time (ms)."""
    path = fig_dir / "02_request_total_attn.png"
    labels = [_short_key(r.request_key) for r in reqs]
    values = [r.total_attn_ms for r in reqs]

    fig, ax = plt.subplots(figsize=(max(10, len(reqs) * 0.35), 6))
    ax.bar(range(len(reqs)), values, color=COLORS[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Total Attention Time (ms)")
    ax.set_title("Per-Request Total Decode Attention Time")
    ax.set_xlabel("Request")

    if values:
        max_idx = int(np.argmax(values))
        ax.annotate(f"{values[max_idx]:.3f}ms", xy=(max_idx, values[max_idx]),
                    fontsize=7, ha="center", va="bottom", color="red")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 3: Heatmap — request × layer
# ===================================================================
def plot_request_layer_heatmap(fig_dir: Path,
                               reqs: list[RequestSummary],
                               layer_stats: dict[tuple[str, int], RequestLayerStat]) -> Path:
    """Heatmap — x=request, y=layer, color=avg attention time (ms)."""
    path = fig_dir / "03_request_layer_heatmap.png"
    layers = sorted({ly for _, ly in layer_stats})
    if not layers or not reqs:
        _empty_figure(path, "Request × Layer Heatmap (no data)")
        return path

    matrix = np.zeros((len(layers), len(reqs)))
    layer_idx = {ly: i for i, ly in enumerate(layers)}
    for ri, req in enumerate(reqs):
        for ly in layers:
            st = layer_stats.get((req.request_key, ly))
            if st:
                matrix[layer_idx[ly], ri] = st.avg_ms

    fig, ax = plt.subplots(figsize=(max(10, len(reqs) * 0.3), max(6, len(layers) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels([_short_key(r.request_key) for r in reqs],
                       rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{ly}" for ly in layers], fontsize=7)
    ax.set_xlabel("Request")
    ax.set_ylabel("Layer")
    ax.set_title("Decode Attention Heatmap (avg ms per layer invocation)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Avg Attention (ms)")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 4: Per-layer average attention across all requests (bar chart)
# ===================================================================
def plot_layer_avg_attn(fig_dir: Path,
                        layer_stats: dict[tuple[str, int], RequestLayerStat]) -> Path:
    """Bar chart — x=layer, y=average attention (ms) across all requests."""
    path = fig_dir / "04_layer_avg_attn.png"
    layer_agg: dict[int, list[float]] = defaultdict(list)
    for (_, ly), st in layer_stats.items():
        layer_agg[ly].append(st.avg_ms)
    layers = sorted(layer_agg)
    means = [float(np.mean(layer_agg[ly])) for ly in layers]
    stds = [float(np.std(layer_agg[ly])) for ly in layers]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.25), 6))
    ax.bar(range(len(layers)), means, yerr=stds, color=COLORS[2],
           alpha=0.85, edgecolor="white", capsize=2)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{ly}" for ly in layers], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Avg Attention Time (ms)")
    ax.set_title("Per-Layer Average Decode Attention (across all requests)")
    ax.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 5: Attention duration distribution (histogram)
# ===================================================================
def plot_attn_duration_distribution(fig_dir: Path,
                                    events: list[AttnEvent]) -> Path:
    """Histogram of raw decode_attn event durations."""
    path = fig_dir / "05_attn_duration_distribution.png"
    durations = [ev.duration_ms for ev in events]
    if not durations:
        _empty_figure(path, "Duration Distribution (no data)")
        return path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(durations, bins=min(100, max(10, len(durations) // 5)),
            color=COLORS[3], alpha=0.8, edgecolor="white")
    ax.axvline(float(np.median(durations)), color="red", linestyle="--", linewidth=1.5,
               label=f"median={np.median(durations):.3f}ms")
    ax.axvline(float(np.mean(durations)), color="blue", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(durations):.3f}ms")
    ax.set_xlabel("Attention Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Decode Attention Duration Distribution (all events)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 6: Attention time vs sequence length (scatter)
# ===================================================================
def plot_attn_vs_seqlen(fig_dir: Path,
                        reqs: list[RequestSummary]) -> Path:
    """Scatter — x=mean_slen, y=avg_attn_per_layer_ms, shows correlation."""
    path = fig_dir / "06_attn_vs_seqlen.png"
    xs, ys, labels = [], [], []
    for r in reqs:
        if r.mean_slen is not None:
            xs.append(r.mean_slen)
            ys.append(r.avg_attn_per_layer_ms)
            labels.append(r.request_key)
    if not xs:
        _empty_figure(path, "Attention vs SeqLen (no slen data)")
        return path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, c=COLORS[4], alpha=0.6, edgecolors="white", s=40)
    ax.set_xlabel("Mean Sequence Length (tokens)")
    ax.set_ylabel("Avg Attention per Layer (ms)")
    ax.set_title("Attention Time vs Sequence Length")

    # Trend line
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, p(x_line), "--", color="red", alpha=0.7,
                label=f"trend: y={z[0]:.4f}x+{z[1]:.4f}")
        corr = float(np.corrcoef(xs, ys)[0, 1])
        ax.legend(title=f"Pearson r={corr:.3f}")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 7: Per-step timeline — batch attention cost over time
# ===================================================================
def plot_step_timeline(fig_dir: Path, events: list[AttnEvent]) -> Path:
    """Line chart — x=step, y=total attention time for that step (ms)."""
    path = fig_dir / "07_step_timeline.png"
    step_agg: dict[int, float] = defaultdict(float)
    step_nreqs: dict[int, int] = {}
    for ev in events:
        step_agg[ev.step] += ev.duration_ms
        step_nreqs[ev.step] = ev.num_reqs

    steps = sorted(step_agg)
    if not steps:
        _empty_figure(path, "Step Timeline (no data)")
        return path

    totals = [step_agg[s] for s in steps]
    nreqs = [step_nreqs.get(s, 0) for s in steps]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(steps, totals, "-o", markersize=3, color=COLORS[0], alpha=0.8,
             label="Total attn/step (ms)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Attention Time (ms)", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    ax2 = ax1.twinx()
    ax2.fill_between(steps, nreqs, alpha=0.15, color=COLORS[1])
    ax2.plot(steps, nreqs, "-", linewidth=1, color=COLORS[1], alpha=0.6,
             label="num_reqs")
    ax2.set_ylabel("Number of Requests", color=COLORS[1])
    ax2.tick_params(axis="y", labelcolor=COLORS[1])

    ax1.set_title("Decode Attention Cost per Step (with batch size)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 8: Per-step per-layer attention stacked area
# ===================================================================
def plot_step_layer_stacked(fig_dir: Path, events: list[AttnEvent]) -> Path:
    """Stacked area — x=step, y=attention time, stacked by layer."""
    path = fig_dir / "08_step_layer_stacked.png"
    step_layer: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for ev in events:
        step_layer[ev.step][ev.layer] += ev.duration_ms

    steps = sorted(step_layer)
    layers = sorted({ly for d in step_layer.values() for ly in d})
    if not steps or not layers:
        _empty_figure(path, "Step-Layer Stacked (no data)")
        return path

    data = np.zeros((len(layers), len(steps)))
    for si, s in enumerate(steps):
        for li, ly in enumerate(layers):
            data[li, si] = step_layer[s].get(ly, 0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    # Too many layers → group them
    if len(layers) > 20:
        # Group into 10 bands
        n_groups = 10
        group_size = len(layers) // n_groups + 1
        grouped = np.zeros((n_groups, len(steps)))
        group_labels = []
        for gi in range(n_groups):
            start = gi * group_size
            end = min((gi + 1) * group_size, len(layers))
            if start >= len(layers):
                break
            grouped[gi] = data[start:end].sum(axis=0)
            group_labels.append(f"L{layers[start]}-L{layers[min(end-1, len(layers)-1)]}")
        cmap = plt.cm.get_cmap("tab10", n_groups)
        ax.stackplot(steps, grouped[:len(group_labels)],
                     labels=group_labels,
                     colors=[cmap(i) for i in range(len(group_labels))],
                     alpha=0.8)
    else:
        cmap = plt.cm.get_cmap("tab20", len(layers))
        ax.stackplot(steps, data, labels=[f"L{ly}" for ly in layers],
                     colors=[cmap(i) for i in range(len(layers))], alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Attention Time (ms)")
    ax.set_title("Per-Step Attention Time by Layer (stacked)")
    ax.legend(loc="upper left", fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 9: Request lifetime — how many steps each request lived
# ===================================================================
def plot_request_lifetime(fig_dir: Path, reqs: list[RequestSummary]) -> Path:
    """Bar chart — x=request, y=number of unique steps (lifetime)."""
    path = fig_dir / "09_request_lifetime.png"
    labels = [_short_key(r.request_key) for r in reqs]
    values = [len(r.unique_steps) for r in reqs]

    fig, ax = plt.subplots(figsize=(max(10, len(reqs) * 0.35), 6))
    ax.bar(range(len(reqs)), values, color=COLORS[5], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Number of Steps (lifetime)")
    ax.set_title("Request Lifetime (unique decode steps)")
    ax.set_xlabel("Request")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 10: Batch size distribution over time
# ===================================================================
def plot_batch_size_over_time(fig_dir: Path, events: list[AttnEvent]) -> Path:
    """Line chart showing how batch size (num_reqs) changes over steps."""
    path = fig_dir / "10_batch_size_over_time.png"
    # Take layer=0 events as proxy for batch size per step
    step_batch: dict[int, int] = {}
    for ev in events:
        if ev.layer == 0 or ev.step not in step_batch:
            step_batch[ev.step] = ev.num_reqs
    steps = sorted(step_batch)
    if not steps:
        _empty_figure(path, "Batch Size Over Time (no data)")
        return path

    sizes = [step_batch[s] for s in steps]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(steps, sizes, alpha=0.3, color=COLORS[6])
    ax.plot(steps, sizes, "-", linewidth=1.5, color=COLORS[6])
    ax.set_xlabel("Step")
    ax.set_ylabel("Batch Size (num_reqs)")
    ax.set_title("Decode Batch Size Over Time")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 11: Attention efficiency — time per token per layer
# ===================================================================
def plot_attn_per_token(fig_dir: Path, events: list[AttnEvent]) -> Path:
    """Scatter — x=num_tokens, y=duration/num_tokens (us/token)."""
    path = fig_dir / "11_attn_per_token.png"
    xs, ys = [], []
    for ev in events:
        if ev.num_tokens > 0:
            xs.append(ev.num_tokens)
            ys.append(ev.duration_ns / ev.num_tokens / 1000.0)  # us/token
    if not xs:
        _empty_figure(path, "Attention per Token (no data)")
        return path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, c=COLORS[7], alpha=0.4, s=15, edgecolors="none")
    ax.set_xlabel("Number of Tokens in Batch")
    ax.set_ylabel("Attention Time per Token (µs)")
    ax.set_title("Decode Attention Efficiency: Time per Token vs Batch Token Count")

    if len(xs) > 2:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.02, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 12: Summary statistics dashboard
# ===================================================================
def plot_summary_dashboard(fig_dir: Path,
                           events: list[AttnEvent],
                           reqs: list[RequestSummary],
                           layer_stats: dict[tuple[str, int], RequestLayerStat]) -> Path:
    """4-panel summary dashboard."""
    path = fig_dir / "00_summary_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Decode Attention Analysis — Summary Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Top-10 requests by avg attn
    ax = axes[0, 0]
    top10 = sorted(reqs, key=lambda r: -r.avg_attn_per_layer_ms)[:10]
    labels = [_short_key(r.request_key, 12) for r in top10]
    vals = [r.avg_attn_per_layer_ms for r in top10]
    ax.barh(range(len(top10)), vals, color=COLORS[0], alpha=0.85)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Avg Attn per Layer (ms)")
    ax.set_title("Top-10 Requests by Avg Attention")
    ax.invert_yaxis()

    # Panel 2: Duration distribution (box plot per layer, sampled)
    ax = axes[0, 1]
    layer_durations: dict[int, list[float]] = defaultdict(list)
    for ev in events:
        layer_durations[ev.layer].append(ev.duration_ms)
    layers_sorted = sorted(layer_durations)
    if layers_sorted:
        # Show at most 20 layers evenly sampled
        if len(layers_sorted) > 20:
            indices = np.linspace(0, len(layers_sorted) - 1, 20, dtype=int)
            sampled_layers = [layers_sorted[i] for i in indices]
        else:
            sampled_layers = layers_sorted
        box_data = [layer_durations[ly] for ly in sampled_layers]
        bp = ax.boxplot(box_data, vert=True, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[2])
            patch.set_alpha(0.7)
        ax.set_xticklabels([f"L{ly}" for ly in sampled_layers],
                           rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("Duration (ms)")
        ax.set_title("Attention Duration Distribution by Layer")
    else:
        ax.set_title("No layer data")

    # Panel 3: step total attn mini-timeline
    ax = axes[1, 0]
    step_total: dict[int, float] = defaultdict(float)
    for ev in events:
        step_total[ev.step] += ev.duration_ms
    steps = sorted(step_total)
    if steps:
        ax.plot(steps, [step_total[s] for s in steps], "-", linewidth=1,
                color=COLORS[0], alpha=0.8)
        ax.fill_between(steps, [step_total[s] for s in steps], alpha=0.15,
                        color=COLORS[0])
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Attn (ms)")
    ax.set_title("Total Attention per Step (timeline)")

    # Panel 4: Text summary
    ax = axes[1, 1]
    ax.axis("off")
    total_events = len(events)
    total_reqs = len(reqs)
    total_layers = len({ly for _, ly in layer_stats})
    total_steps = len({ev.step for ev in events})
    all_durations = [ev.duration_ms for ev in events]
    summary_text = (
        f"Total NVTX Events:  {total_events}\n"
        f"Unique Requests:    {total_reqs}\n"
        f"Unique Layers:      {total_layers}\n"
        f"Unique Steps:       {total_steps}\n"
        f"\n"
        f"Event Duration (ms):\n"
        f"  Mean:    {np.mean(all_durations):.4f}\n"
        f"  Median:  {np.median(all_durations):.4f}\n"
        f"  Std:     {np.std(all_durations):.4f}\n"
        f"  Min:     {np.min(all_durations):.4f}\n"
        f"  Max:     {np.max(all_durations):.4f}\n"
        f"  P95:     {np.percentile(all_durations, 95):.4f}\n"
        f"  P99:     {np.percentile(all_durations, 99):.4f}\n"
    ) if all_durations else "No events found."
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary Statistics")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Helpers
# ===================================================================
def _empty_figure(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center",
            fontsize=14, color="gray", transform=ax.transAxes)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", text).strip("_")


# ===================================================================
# Report-level analysis orchestrator
# ===================================================================
def analyze_one_report(source: Path, sqlite_path: Path, output_root: Path,
                       max_requests: int, sort_by: str) -> dict:
    slug = slugify(source.with_suffix("").relative_to(REPO_ROOT).as_posix())
    report_dir = output_root / slug
    fig_dir = report_dir / "figures"
    csv_dir = report_dir / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    info: dict = {"source": str(source), "sqlite": str(sqlite_path), "slug": slug,
                  "generated": []}
    conn = sqlite3.connect(sqlite_path)
    try:
        nvtx_table = _first_table(conn, NVTX_TABLE_CANDIDATES)
        if nvtx_table is None:
            print(f"  WARN: No NVTX table in {sqlite_path.name}")
            info["error"] = "no NVTX table"
            return info

        events = read_events(conn, nvtx_table)
        if not events:
            print(f"  WARN: No decode_attn events in {sqlite_path.name}")
            info["error"] = "no decode_attn events"
            return info

        print(f"  Found {len(events)} decode_attn events in table '{nvtx_table}'")

        # Aggregate
        req_map, layer_stats = expand_stats(events)
        reqs = sort_requests(list(req_map.values()), sort_by)
        plot_reqs = reqs[:max_requests]

        # CSVs
        write_events_csv(csv_dir / "decode_attention_events.csv", events)
        write_request_csv(csv_dir / "request_summary.csv", reqs)
        write_request_layer_csv(csv_dir / "request_layer.csv", layer_stats, reqs)

        # --- All charts ---
        generated = []
        generated.append(plot_summary_dashboard(fig_dir, events, reqs, layer_stats))
        generated.append(plot_request_avg_attn_per_layer(fig_dir, plot_reqs))
        generated.append(plot_request_total_attn(fig_dir, plot_reqs))
        generated.append(plot_request_layer_heatmap(fig_dir, plot_reqs, layer_stats))
        generated.append(plot_layer_avg_attn(fig_dir, layer_stats))
        generated.append(plot_attn_duration_distribution(fig_dir, events))
        generated.append(plot_attn_vs_seqlen(fig_dir, reqs))
        generated.append(plot_step_timeline(fig_dir, events))
        generated.append(plot_step_layer_stacked(fig_dir, events))
        generated.append(plot_request_lifetime(fig_dir, plot_reqs))
        generated.append(plot_batch_size_over_time(fig_dir, events))
        generated.append(plot_attn_per_token(fig_dir, events))

        info["generated"] = [str(p) for p in generated]
        info["n_events"] = len(events)
        info["n_requests"] = len(reqs)
        info["n_layers"] = len({ly for _, ly in layer_stats})
        info["n_steps"] = len({ev.step for ev in events})
        return info
    finally:
        conn.close()


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(args.inputs)
    if not inputs:
        print("ERROR: No .nsys-rep or .sqlite files found.", file=sys.stderr)
        return 1

    print(f"Found {len(inputs)} report(s) to analyze.")
    results = []
    for source in inputs:
        print(f"\n{'='*60}")
        print(f"Analyzing: {source.name}")
        print(f"{'='*60}")
        sqlite_path = ensure_sqlite(source, args.nsys_bin,
                                    args.force_export, args.skip_export)
        info = analyze_one_report(source, sqlite_path, output_root,
                                  args.max_requests, args.sort_by)
        results.append(info)

    # Global summary
    summary_path = output_root / "analysis_summary.txt"
    with summary_path.open("w") as f:
        f.write("Decode Attention NVTX Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        for info in results:
            f.write(f"Source:    {info['source']}\n")
            f.write(f"SQLite:   {info['sqlite']}\n")
            if "error" in info:
                f.write(f"Status:   SKIPPED — {info['error']}\n")
            else:
                f.write(f"Events:   {info.get('n_events', 0)}\n")
                f.write(f"Requests: {info.get('n_requests', 0)}\n")
                f.write(f"Layers:   {info.get('n_layers', 0)}\n")
                f.write(f"Steps:    {info.get('n_steps', 0)}\n")
                f.write(f"Charts:   {len(info.get('generated', []))}\n")
            f.write("\n")

    print(f"\nDone! All outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
