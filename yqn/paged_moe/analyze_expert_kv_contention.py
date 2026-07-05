#!/usr/bin/env python
"""
Focused analysis for the Expert/KV contention R x N experiment.

The goal is to answer the main question clearly: does increasing expert offload
R buy enough KV-cache headroom to beat the extra offload cost as request load N
increases?

The script also emits diagnostic plots so a run can be inspected from multiple
angles: throughput, TPOT/ITL/TTFT, preemption, avg/peak KV pressure, KV capacity,
offload cost, attention/MoE timing, active expert ratio, and R0-relative trends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "results_expert_kv_contention"
POINT_RE = re.compile(r"^R(?P<r>\d+)_N(?P<n>\d+(?:\.\d+)?)$")
KV_MEM_RE = re.compile(r"Available KV cache memory: ([0-9.]+) GiB")
KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
KV_USAGE_RE = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")

C_TPUT = "#0072B2"
C_OUTPUT = "#56B4E9"
C_TPOT = "#CC79A7"
C_TTFT = "#E69F00"
C_ITL = "#999999"
C_KV = "#009E73"
C_PREEMPT = "#D55E00"
C_OFFLOAD = "#D55E00"
C_ATTN = "#009E73"
C_MOE = "#0072B2"
C_WAIT = "#CC79A7"
C_MARK = "#F0E442"

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


@dataclass
class Point:
    point: str
    path: Path
    r_label: str
    r_value: int
    n_value: float
    status: str
    has_bench: bool
    completed: int | None = None
    failed: int | None = None
    duration_s: float | None = None
    request_throughput: float | None = None
    output_throughput: float | None = None
    total_token_throughput: float | None = None
    max_output_tokens_per_s: float | None = None
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    median_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    median_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    std_tpot_ms: float | None = None
    request_rate: float | None = None
    max_concurrency: int | None = None
    max_concurrent_requests: int | None = None
    num_prompts: int | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_scheduled_tokens: int | None = None
    total_attention_ms: float | None = None
    total_moe_ms: float | None = None
    total_h2d_copy_ms: float | None = None
    total_prefetch_wait_ms: float | None = None
    total_prefetch_copy_bytes: int | None = None
    total_preemptions: int | None = None
    avg_kv_cache_usage: float | None = None
    peak_kv_cache_usage: float | None = None
    latest_kv_cache_usage: float | None = None
    avg_active_expert_ratio: float | None = None
    prefix_cache_hit_rate: float | None = None
    prefix_cache_hits: int | None = None
    prefix_cache_queries: int | None = None
    kv_cache_memory_gib: float | None = None
    kv_cache_tokens: int | None = None
    forwards: int | None = None
    scheduler_records: int | None = None
    request_final_records: int | None = None

    @property
    def complete(self) -> bool:
        return self.has_bench and self.completed is not None

    @property
    def preemptions_per_request(self) -> float | None:
        if not self.completed:
            return None
        return safe_float(self.total_preemptions, default=0.0) / self.completed

    @property
    def preemptions_per_1k_scheduled_tokens(self) -> float | None:
        if not self.total_scheduled_tokens:
            return None
        return (safe_float(self.total_preemptions, default=0.0) * 1000.0
                / self.total_scheduled_tokens)

    @property
    def input_tokens_per_request(self) -> float | None:
        if not self.completed or self.total_input_tokens is None:
            return None
        return self.total_input_tokens / self.completed

    @property
    def output_tokens_per_request(self) -> float | None:
        if not self.completed or self.total_output_tokens is None:
            return None
        return self.total_output_tokens / self.completed

    @property
    def scheduled_tokens_per_request(self) -> float | None:
        if not self.completed or self.total_scheduled_tokens is None:
            return None
        return self.total_scheduled_tokens / self.completed

    @property
    def h2d_copy_ms_per_scheduled_token(self) -> float | None:
        return per_token(self.total_h2d_copy_ms, self.total_scheduled_tokens)

    @property
    def prefetch_wait_ms_per_scheduled_token(self) -> float | None:
        return per_token(self.total_prefetch_wait_ms, self.total_scheduled_tokens)

    @property
    def moe_ms_per_scheduled_token(self) -> float | None:
        return per_token(self.total_moe_ms, self.total_scheduled_tokens)

    @property
    def attention_ms_per_scheduled_token(self) -> float | None:
        return per_token(self.total_attention_ms, self.total_scheduled_tokens)

    @property
    def offload_ms_per_scheduled_token(self) -> float | None:
        copy_ms = safe_float(self.h2d_copy_ms_per_scheduled_token, default=0.0)
        wait_ms = safe_float(self.prefetch_wait_ms_per_scheduled_token, default=0.0)
        return copy_ms + wait_ms

    @property
    def h2d_copy_gib(self) -> float | None:
        if self.total_prefetch_copy_bytes is None:
            return None
        return self.total_prefetch_copy_bytes / (1024**3)

    @property
    def tokens_per_kv_cache_token(self) -> float | None:
        if not self.kv_cache_tokens or self.total_token_throughput is None:
            return None
        return self.total_token_throughput / self.kv_cache_tokens

    def as_row(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "r_label": self.r_label,
            "r_value": self.r_value,
            "n_value": self.n_value,
            "status": self.status,
            "completed": self.completed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "request_rate": self.request_rate,
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "total_token_throughput": self.total_token_throughput,
            "max_output_tokens_per_s": self.max_output_tokens_per_s,
            "mean_ttft_ms": self.mean_ttft_ms,
            "median_ttft_ms": self.median_ttft_ms,
            "p99_ttft_ms": self.p99_ttft_ms,
            "mean_tpot_ms": self.mean_tpot_ms,
            "median_tpot_ms": self.median_tpot_ms,
            "p99_tpot_ms": self.p99_tpot_ms,
            "mean_itl_ms": self.mean_itl_ms,
            "median_itl_ms": self.median_itl_ms,
            "p99_itl_ms": self.p99_itl_ms,
            "std_tpot_ms": self.std_tpot_ms,
            "max_concurrency": self.max_concurrency,
            "max_concurrent_requests": self.max_concurrent_requests,
            "num_prompts": self.num_prompts,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
            "scheduled_tokens_per_request": self.scheduled_tokens_per_request,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "total_preemptions": self.total_preemptions,
            "preemptions_per_request": self.preemptions_per_request,
            "preemptions_per_1k_scheduled_tokens": self.preemptions_per_1k_scheduled_tokens,
            "avg_kv_cache_usage": self.avg_kv_cache_usage,
            "peak_kv_cache_usage": self.peak_kv_cache_usage,
            "latest_kv_cache_usage": self.latest_kv_cache_usage,
            "avg_active_expert_ratio": self.avg_active_expert_ratio,
            "prefix_cache_hit_rate_aux": self.prefix_cache_hit_rate,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_queries": self.prefix_cache_queries,
            "kv_cache_memory_gib": self.kv_cache_memory_gib,
            "kv_cache_tokens": self.kv_cache_tokens,
            "tokens_per_kv_cache_token": self.tokens_per_kv_cache_token,
            "forwards": self.forwards,
            "scheduler_records": self.scheduler_records,
            "request_final_records": self.request_final_records,
            "total_attention_ms": self.total_attention_ms,
            "total_moe_ms": self.total_moe_ms,
            "total_h2d_copy_ms": self.total_h2d_copy_ms,
            "total_prefetch_wait_ms": self.total_prefetch_wait_ms,
            "h2d_copy_gib": self.h2d_copy_gib,
            "attention_ms_per_scheduled_token": self.attention_ms_per_scheduled_token,
            "moe_ms_per_scheduled_token": self.moe_ms_per_scheduled_token,
            "h2d_copy_ms_per_scheduled_token": self.h2d_copy_ms_per_scheduled_token,
            "prefetch_wait_ms_per_scheduled_token": self.prefetch_wait_ms_per_scheduled_token,
            "offload_ms_per_scheduled_token": self.offload_ms_per_scheduled_token,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focused Expert/KV contention analysis plots."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. If omitted, use the latest run under --result-root.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root containing timestamped run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run-dir>/analysis_focused.",
    )
    parser.add_argument(
        "--focus-n",
        type=float,
        default=None,
        help="Request rate for the tradeoff/cost plots. Default: highest measured N.",
    )
    parser.add_argument(
        "--focus-r",
        type=int,
        default=None,
        help="Accepted for compatibility with the old analyzer; unused.",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Accepted for compatibility; this analyzer does not parse full traces.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def per_token(total_ms: float | None, tokens: int | None) -> float | None:
    if total_ms is None or not tokens:
        return None
    return total_ms / tokens


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def fmt_n(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def latest_run(result_root: Path) -> Path:
    runs = [p for p in result_root.expanduser().resolve().iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no run directories found under {result_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_status(point_dir: Path) -> str:
    status_path = point_dir / "status.txt"
    if not status_path.is_file():
        return "missing_status"
    return status_path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"


def read_run_config(run_dir: Path) -> dict[str, str]:
    config_path = run_dir / "run_config.env"
    config: dict[str, str] = {}
    if not config_path.is_file():
        return config
    for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key] = value
    return config


def parse_server_stats(point_dir: Path) -> tuple[float | None, int | None, float | None, float | None]:
    log_path = point_dir / "server.log"
    if not log_path.is_file():
        return None, None, None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    mem_matches = KV_MEM_RE.findall(text)
    token_matches = KV_TOKENS_RE.findall(text)
    usage_matches = [safe_float(x) for x in KV_USAGE_RE.findall(text)]
    usage_values = [x for x in usage_matches if x is not None]
    mem_gib = safe_float(mem_matches[-1]) if mem_matches else None
    tokens = safe_int(token_matches[-1].replace(",", "")) if token_matches else None
    peak_usage = max(usage_values) / 100.0 if usage_values else None
    latest_usage = usage_values[-1] / 100.0 if usage_values else None
    return mem_gib, tokens, peak_usage, latest_usage


def load_point(point_dir: Path) -> Point | None:
    match = POINT_RE.match(point_dir.name)
    if match is None:
        return None

    r_value = int(match.group("r"))
    n_value = float(match.group("n"))
    bench_path = point_dir / "bench.json"
    status = read_status(point_dir)
    kv_mem_gib, kv_tokens, peak_kv, latest_kv = parse_server_stats(point_dir)

    point = Point(
        point=point_dir.name,
        path=point_dir,
        r_label=f"R{r_value}",
        r_value=r_value,
        n_value=n_value,
        status=status,
        has_bench=bench_path.is_file(),
        kv_cache_memory_gib=kv_mem_gib,
        kv_cache_tokens=kv_tokens,
        peak_kv_cache_usage=peak_kv,
        latest_kv_cache_usage=latest_kv,
    )
    if not bench_path.is_file() or bench_path.stat().st_size == 0:
        return point

    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return point
    ekv = bench.get("expert_kv_contention") or {}

    point.completed = safe_int(bench.get("completed"))
    point.failed = safe_int(bench.get("failed"))
    point.duration_s = safe_float(bench.get("duration"))
    point.request_rate = safe_float(bench.get("request_rate"), default=n_value)
    point.request_throughput = safe_float(bench.get("request_throughput"))
    point.output_throughput = safe_float(bench.get("output_throughput"))
    point.total_token_throughput = safe_float(bench.get("total_token_throughput"))
    point.max_output_tokens_per_s = safe_float(bench.get("max_output_tokens_per_s"))
    point.mean_ttft_ms = safe_float(bench.get("mean_ttft_ms"))
    point.median_ttft_ms = safe_float(bench.get("median_ttft_ms"))
    point.p99_ttft_ms = safe_float(bench.get("p99_ttft_ms"))
    point.mean_tpot_ms = safe_float(bench.get("mean_tpot_ms"))
    point.median_tpot_ms = safe_float(bench.get("median_tpot_ms"))
    point.p99_tpot_ms = safe_float(bench.get("p99_tpot_ms"))
    point.mean_itl_ms = safe_float(bench.get("mean_itl_ms"))
    point.median_itl_ms = safe_float(bench.get("median_itl_ms"))
    point.p99_itl_ms = safe_float(bench.get("p99_itl_ms"))
    point.std_tpot_ms = safe_float(bench.get("std_tpot_ms"))
    point.max_concurrency = safe_int(bench.get("max_concurrency"))
    point.max_concurrent_requests = safe_int(bench.get("max_concurrent_requests"))
    point.num_prompts = safe_int(bench.get("num_prompts"))
    point.total_input_tokens = safe_int(bench.get("total_input_tokens"))
    point.total_output_tokens = safe_int(bench.get("total_output_tokens"))

    point.forwards = safe_int(ekv.get("forwards"))
    point.scheduler_records = safe_int(ekv.get("scheduler_records"))
    point.request_final_records = safe_int(ekv.get("request_final_records"))
    point.total_scheduled_tokens = safe_int(ekv.get("total_scheduled_tokens"))
    point.total_attention_ms = safe_float(ekv.get("total_attention_ms"))
    point.total_moe_ms = safe_float(ekv.get("total_moe_ms"))
    point.total_h2d_copy_ms = safe_float(ekv.get("total_h2d_copy_ms"))
    point.total_prefetch_wait_ms = safe_float(ekv.get("total_prefetch_wait_ms"))
    point.total_prefetch_copy_bytes = safe_int(ekv.get("total_prefetch_copy_bytes"))
    point.total_preemptions = safe_int(ekv.get("total_preemptions"), default=0)
    point.avg_kv_cache_usage = safe_float(ekv.get("avg_kv_cache_usage"))
    point.avg_active_expert_ratio = safe_float(ekv.get("avg_active_expert_ratio"))
    point.prefix_cache_hit_rate = safe_float(ekv.get("prefix_cache_hit_rate"))
    point.prefix_cache_hits = safe_int(ekv.get("prefix_cache_hits"))
    point.prefix_cache_queries = safe_int(ekv.get("prefix_cache_queries"))
    return point


def load_points(run_dir: Path) -> list[Point]:
    points: list[Point] = []
    for point_dir in run_dir.expanduser().resolve().iterdir():
        if not point_dir.is_dir():
            continue
        point = load_point(point_dir)
        if point is not None:
            points.append(point)
    points.sort(key=lambda p: (p.n_value, p.r_value))
    if not points:
        raise FileNotFoundError(f"no R*_N* point directories found under {run_dir}")
    return points


def write_point_summary(points: list[Point], output_dir: Path) -> None:
    rows = [p.as_row() for p in points]
    fieldnames = list(rows[0].keys())
    with (output_dir / "point_summary.csv").open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for k, v in row.items()})


def best_by_n(points: list[Point]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_values = sorted({p.n_value for p in points})
    for n_value in n_values:
        candidates = [
            p for p in points
            if p.n_value == n_value
            and p.complete
            and p.total_token_throughput is not None
            and p.mean_tpot_ms is not None
        ]
        if not candidates:
            continue
        best_tput = max(candidates, key=lambda p: p.total_token_throughput or -math.inf)
        best_tpot = min(candidates, key=lambda p: p.mean_tpot_ms or math.inf)
        r0 = next((p for p in candidates if p.r_value == 0), None)
        rows.append({
            "n_value": n_value,
            "best_total_tput_point": best_tput.point,
            "best_total_tput_r": best_tput.r_value,
            "best_total_tput": best_tput.total_token_throughput,
            "best_tpot_point": best_tpot.point,
            "best_tpot_r": best_tpot.r_value,
            "best_tpot_ms": best_tpot.mean_tpot_ms,
            "r0_total_tput": r0.total_token_throughput if r0 else None,
            "r0_tpot_ms": r0.mean_tpot_ms if r0 else None,
            "best_total_tput_vs_r0": (
                best_tput.total_token_throughput / r0.total_token_throughput
                if r0 and r0.total_token_throughput else None
            ),
            "best_tpot_vs_r0": (
                best_tpot.mean_tpot_ms / r0.mean_tpot_ms
                if r0 and r0.mean_tpot_ms else None
            ),
        })
    return rows


def write_best_by_n(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "best_by_n.csv").open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for k, v in row.items()})


def matrix_for(points: list[Point], key: str) -> tuple[list[int], list[float], np.ndarray]:
    r_values = sorted({p.r_value for p in points})
    n_values = sorted({p.n_value for p in points})
    by_coord = {(p.r_value, p.n_value): p for p in points}
    values = np.full((len(r_values), len(n_values)), np.nan, dtype=float)
    for i, r_value in enumerate(r_values):
        for j, n_value in enumerate(n_values):
            point = by_coord.get((r_value, n_value))
            if point is None:
                continue
            value = getattr(point, key)
            if value is not None:
                values[i, j] = float(value)
    return r_values, n_values, values


def relative_matrix(points: list[Point], key: str) -> tuple[list[int], list[float], np.ndarray]:
    r_values = sorted({p.r_value for p in points})
    n_values = sorted({p.n_value for p in points})
    by_coord = {(p.r_value, p.n_value): p for p in points}
    values = np.full((len(r_values), len(n_values)), np.nan, dtype=float)
    for j, n_value in enumerate(n_values):
        base = by_coord.get((0, n_value))
        base_value = getattr(base, key) if base is not None else None
        if base_value in (None, 0):
            continue
        for i, r_value in enumerate(r_values):
            point = by_coord.get((r_value, n_value))
            if point is None:
                continue
            value = getattr(point, key)
            if value is not None:
                values[i, j] = float(value) / float(base_value)
    return r_values, n_values, values


def plot_matrix(
    ax: plt.Axes,
    r_values: list[int],
    n_values: list[float],
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cell_fmt: str,
    cmap: str,
    best: str | None = None,
    reverse_text_threshold: bool = False,
) -> None:
    masked = np.ma.masked_invalid(values)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#eeeeee")
    im = ax.imshow(masked, aspect="auto", cmap=colormap)

    ax.set_title(title)
    ax.set_xticks(range(len(n_values)), [f"N={fmt_n(n)}" for n in n_values])
    ax.set_yticks(range(len(r_values)), [f"R{r}" for r in r_values])
    ax.set_xlabel("request rate N")
    ax.set_ylabel("expert offload ratio R")

    finite_values = values[np.isfinite(values)]
    midpoint = float(np.nanmedian(finite_values)) if finite_values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            dark_cell = value > midpoint
            if reverse_text_threshold:
                dark_cell = not dark_cell
            color = "white" if dark_cell else "black"
            ax.text(j, i, cell_fmt.format(value), ha="center", va="center", color=color)

    if best in {"max", "min"}:
        for j in range(values.shape[1]):
            col = values[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                continue
            idx = int(np.nanargmax(col) if best == "max" else np.nanargmin(col))
            ax.add_patch(Rectangle(
                (j - 0.5, idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor=C_MARK,
                linewidth=2.4,
            ))

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_heatmap(
    ax: plt.Axes,
    points: list[Point],
    key: str,
    title: str,
    cbar_label: str,
    cell_fmt: str,
    cmap: str,
    scale: float = 1.0,
    best: str | None = None,
) -> None:
    r_values, n_values, raw_values = matrix_for(points, key)
    plot_matrix(
        ax,
        r_values,
        n_values,
        raw_values * scale,
        title,
        cbar_label,
        cell_fmt,
        cmap,
        best=best,
    )


def plot_main_summary(points: list[Point], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plot_heatmap(
        axes[0, 0],
        points,
        "total_token_throughput",
        "Total token throughput\n(higher is better; yellow = best R per N)",
        "tokens/s",
        "{:.0f}",
        "Blues",
        best="max",
    )
    plot_heatmap(
        axes[0, 1],
        points,
        "mean_tpot_ms",
        "Mean TPOT\n(lower is better; yellow = best R per N)",
        "ms/token",
        "{:.0f}",
        "RdPu",
        best="min",
    )
    plot_heatmap(
        axes[1, 0],
        points,
        "avg_kv_cache_usage",
        "Average KV cache usage\n(capacity pressure, not prefix hit rate)",
        "%",
        "{:.0f}%",
        "Greens",
        scale=100.0,
    )
    plot_heatmap(
        axes[1, 1],
        points,
        "offload_ms_per_scheduled_token",
        "Offload overhead\n(H2D copy + prefetch wait)",
        "ms/scheduled token",
        "{:.2f}",
        "Oranges",
    )
    fig.suptitle("Expert/KV contention matrix", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "00_main_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_r(best_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    labels = [fmt_n(float(row["n_value"])) for row in best_rows]
    xs = np.arange(len(labels))
    best_tput_r = [safe_float(row["best_total_tput_r"], default=np.nan) for row in best_rows]
    best_tpot_r = [safe_float(row["best_tpot_r"], default=np.nan) for row in best_rows]
    tput_vs_r0 = [safe_float(row["best_total_tput_vs_r0"], default=np.nan) for row in best_rows]
    tpot_vs_r0 = [safe_float(row["best_tpot_vs_r0"], default=np.nan) for row in best_rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(xs, best_tput_r, marker="o", color=C_TPUT, label="best throughput R")
    axes[0].plot(xs, best_tpot_r, marker="s", color=C_TPOT, label="best TPOT R")
    axes[0].set_ylabel("best R")
    axes[0].set_ylim(-5, 105)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(loc="best")
    axes[0].set_title("Does the optimal R shift as load N increases?")

    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].plot(xs, tput_vs_r0, marker="o", color=C_TPUT, label="best throughput / R0")
    axes[1].plot(xs, tpot_vs_r0, marker="s", color=C_TPOT, label="best TPOT / R0")
    axes[1].set_ylabel("relative to R0")
    axes[1].set_xlabel("request rate N")
    axes[1].set_xticks(xs, labels)
    axes[1].legend(loc="best")
    axes[1].set_title("Values near 1 mean offload did not beat all-HBM")

    fig.savefig(figure_dir / "01_best_r_by_n.png", bbox_inches="tight")
    plt.close(fig)


def pick_focus_n(points: list[Point], requested_n: float | None) -> float:
    n_values = sorted({p.n_value for p in points})
    if not n_values:
        raise ValueError("no N values available")
    if requested_n is None:
        return n_values[-1]
    return min(n_values, key=lambda n: abs(n - requested_n))


def plot_focus_tradeoff(points: list[Point], focus_n: float, figure_dir: Path) -> None:
    rows = sorted([p for p in points if p.n_value == focus_n], key=lambda p: p.r_value)
    if not rows:
        return

    rs = np.asarray([p.r_value for p in rows], dtype=float)
    tpot = np.asarray([safe_float(p.mean_tpot_ms, default=np.nan) for p in rows], dtype=float)
    offload = np.asarray([
        safe_float(p.offload_ms_per_scheduled_token, default=np.nan) for p in rows
    ], dtype=float)
    avg_kv_usage = np.asarray([
        safe_float(p.avg_kv_cache_usage, default=np.nan) * 100.0 for p in rows
    ], dtype=float)
    peak_kv_usage = np.asarray([
        safe_float(p.peak_kv_cache_usage, default=np.nan) * 100.0 for p in rows
    ], dtype=float)
    kv_mem = np.asarray([safe_float(p.kv_cache_memory_gib, default=np.nan) for p in rows])
    preempt = np.asarray([
        safe_float(p.total_preemptions, default=np.nan) for p in rows
    ], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(rs, tpot, marker="o", color=C_TPOT, label="mean TPOT")
    ax0.set_xlabel("expert offload ratio R")
    ax0.set_ylabel("mean TPOT (ms/token)", color=C_TPOT)
    ax0.tick_params(axis="y", labelcolor=C_TPOT)
    ax0.set_xticks(rs, [f"R{int(r)}" for r in rs])
    ax0.set_title(f"Latency/offload cost at N={fmt_n(focus_n)}")

    ax0b = ax0.twinx()
    ax0b.plot(rs, offload, marker="s", color=C_OFFLOAD, label="H2D + wait")
    ax0b.set_ylabel("offload overhead (ms/scheduled token)", color=C_OFFLOAD)
    ax0b.tick_params(axis="y", labelcolor=C_OFFLOAD)

    lines = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lines, [line.get_label() for line in lines], loc="upper left")

    ax1 = axes[1]
    ax1.plot(rs, avg_kv_usage, marker="o", color=C_KV, label="avg KV usage")
    ax1.plot(rs, peak_kv_usage, marker="v", color="#006400", label="peak KV usage")
    ax1.set_xlabel("expert offload ratio R")
    ax1.set_ylabel("KV cache usage (%)", color=C_KV)
    ax1.tick_params(axis="y", labelcolor=C_KV)
    ax1.set_xticks(rs, [f"R{int(r)}" for r in rs])
    ax1.set_title("KV pressure vs capacity")

    ax1b = ax1.twinx()
    ax1b.plot(rs, kv_mem, marker="s", color=C_TPUT, label="KV capacity GiB")
    ax1b.plot(rs, preempt, marker="^", color="#333333", label="total preemptions")
    ax1b.set_ylabel("KV GiB / total preemptions")

    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")

    if finite_max(preempt) == 0:
        ax1.text(
            0.5,
            0.08,
            "No preemption observed",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#999999"},
        )

    fig.savefig(figure_dir / "02_focus_n_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_n_tradeoffs(points: list[Point], figure_dir: Path) -> None:
    n_values = sorted({p.n_value for p in points})
    if not n_values:
        return

    fig, axes = plt.subplots(
        len(n_values),
        2,
        figsize=(13, max(3.2 * len(n_values), 4.8)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, n_value in enumerate(n_values):
        rows = sorted([p for p in points if p.n_value == n_value], key=lambda p: p.r_value)
        if not rows:
            continue

        rs = np.asarray([p.r_value for p in rows], dtype=float)
        labels = [f"R{int(r)}" for r in rs]
        tpot = np.asarray([safe_float(p.mean_tpot_ms, default=np.nan) for p in rows], dtype=float)
        offload = np.asarray([
            safe_float(p.offload_ms_per_scheduled_token, default=np.nan) for p in rows
        ], dtype=float)
        avg_kv_usage = np.asarray([
            safe_float(p.avg_kv_cache_usage, default=np.nan) * 100.0 for p in rows
        ], dtype=float)
        peak_kv_usage = np.asarray([
            safe_float(p.peak_kv_cache_usage, default=np.nan) * 100.0 for p in rows
        ], dtype=float)
        preempt = np.asarray([
            safe_float(p.total_preemptions, default=np.nan) for p in rows
        ], dtype=float)
        throughput = np.asarray([
            safe_float(p.total_token_throughput, default=np.nan) for p in rows
        ], dtype=float)

        ax0 = axes[row_idx, 0]
        ax0.plot(rs, tpot, marker="o", color=C_TPOT, label="mean TPOT")
        ax0.set_ylabel("TPOT (ms/token)", color=C_TPOT)
        ax0.tick_params(axis="y", labelcolor=C_TPOT)
        ax0.set_xticks(rs, labels)
        ax0.set_title(f"N={fmt_n(n_value)} latency/offload")
        ax0b = ax0.twinx()
        ax0b.plot(rs, offload, marker="s", color=C_OFFLOAD, label="H2D + wait")
        ax0b.plot(rs, throughput, marker="^", color=C_TPUT, label="throughput")
        ax0b.set_ylabel("offload ms/token or tok/s")
        lines = ax0.get_lines() + ax0b.get_lines()
        ax0.legend(lines, [line.get_label() for line in lines], loc="best")

        ax1 = axes[row_idx, 1]
        ax1.plot(rs, avg_kv_usage, marker="o", color=C_KV, label="avg KV usage")
        ax1.plot(rs, peak_kv_usage, marker="v", color="#006400", label="peak KV usage")
        ax1.set_ylabel("KV usage (%)", color=C_KV)
        ax1.tick_params(axis="y", labelcolor=C_KV)
        ax1.set_xticks(rs, labels)
        ax1.set_title(f"N={fmt_n(n_value)} KV pressure/preemption")
        ax1b = ax1.twinx()
        ax1b.plot(rs, preempt, marker="^", color="#333333", label="total preemptions")
        ax1b.set_ylabel("total preemptions")
        lines = ax1.get_lines() + ax1b.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")

    axes[-1, 0].set_xlabel("expert offload ratio R")
    axes[-1, 1].set_xlabel("expert offload ratio R")
    fig.suptitle("Focus tradeoff expanded across all request rates", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "02b_focus_n_tradeoff_all_n.png", bbox_inches="tight")
    plt.close(fig)


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def plot_pressure_summary(points: list[Point], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    specs = [
        ("total_preemptions", "Total preemptions", "count", "{:.0f}", "Reds", 1.0),
        ("preemptions_per_request", "Preemptions/request", "count/request", "{:.2f}", "Reds", 1.0),
        ("preemptions_per_1k_scheduled_tokens", "Preemptions / 1K scheduled tokens", "count/1K tokens", "{:.2f}", "Reds", 1.0),
        ("avg_kv_cache_usage", "Average KV usage", "%", "{:.0f}%", "Greens", 100.0),
        ("peak_kv_cache_usage", "Peak logged KV usage", "%", "{:.0f}%", "Greens", 100.0),
        ("kv_cache_tokens", "KV cache capacity", "tokens", "{:.0f}", "Blues", 1.0),
    ]
    for ax, (key, title, label, fmt, cmap, scale) in zip(axes.flat, specs):
        plot_heatmap(ax, points, key, title, label, fmt, cmap, scale=scale)
    fig.suptitle("KV pressure and capacity diagnostics", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "03_pressure_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_curves_by_n(points: list[Point], figure_dir: Path) -> None:
    n_values = sorted({p.n_value for p in points})
    by_n = {n: sorted([p for p in points if p.n_value == n], key=lambda p: p.r_value)
            for n in n_values}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    curve_specs: list[tuple[str, str, str, str, Callable[[Point], float | None]]] = [
        ("total_token_throughput", "Total token throughput vs R", "tokens/s", C_TPUT,
         lambda p: p.total_token_throughput),
        ("mean_tpot_ms", "Mean TPOT vs R", "ms/token", C_TPOT,
         lambda p: p.mean_tpot_ms),
        ("total_preemptions", "Total preemptions vs R", "count", C_PREEMPT,
         lambda p: p.total_preemptions),
        ("offload_ms_per_scheduled_token", "Offload overhead vs R", "ms/scheduled token", C_OFFLOAD,
         lambda p: p.offload_ms_per_scheduled_token),
    ]
    for ax, (_, title, ylabel, _, getter) in zip(axes.flat, curve_specs):
        for n_value, rows in by_n.items():
            xs = [p.r_value for p in rows]
            ys = [safe_float(getter(p), default=np.nan) for p in rows]
            ax.plot(xs, ys, marker="o", label=f"N={fmt_n(n_value)}")
        ax.set_title(title)
        ax.set_xlabel("expert offload ratio R")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({p.r_value for p in points}))
        ax.legend(loc="best", ncols=2)
    fig.suptitle("Curves across request load", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "04_curves_by_n.png", bbox_inches="tight")
    plt.close(fig)


def plot_relative_to_r0(points: list[Point], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = [
        ("total_token_throughput", "Throughput / R0\n(>1 means offload faster)", "ratio", "{:.2f}", "RdYlGn", "max"),
        ("mean_tpot_ms", "TPOT / R0\n(<1 means offload lower latency)", "ratio", "{:.2f}", "RdYlGn_r", "min"),
        ("preemptions_per_request", "Preemptions/request / R0", "ratio", "{:.2f}", "Reds", None),
        ("avg_kv_cache_usage", "Average KV usage / R0", "ratio", "{:.2f}", "Greens", None),
    ]
    for ax, (key, title, label, fmt, cmap, best) in zip(axes.flat, specs):
        r_values, n_values, values = relative_matrix(points, key)
        plot_matrix(ax, r_values, n_values, values, title, label, fmt, cmap, best=best)
    fig.suptitle("R0-relative diagnostics", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "05_relative_to_r0.png", bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(points: list[Point], focus_n: float, figure_dir: Path) -> None:
    rows = sorted([p for p in points if p.n_value == focus_n], key=lambda p: p.r_value)
    if not rows:
        return
    labels = [f"R{p.r_value}" for p in rows]
    xs = np.arange(len(labels))
    attention = np.asarray([
        safe_float(p.attention_ms_per_scheduled_token, default=0.0) for p in rows
    ])
    moe = np.asarray([safe_float(p.moe_ms_per_scheduled_token, default=0.0) for p in rows])
    h2d = np.asarray([
        safe_float(p.h2d_copy_ms_per_scheduled_token, default=0.0) for p in rows
    ])
    wait = np.asarray([
        safe_float(p.prefetch_wait_ms_per_scheduled_token, default=0.0) for p in rows
    ])
    tpot = np.asarray([safe_float(p.mean_tpot_ms, default=np.nan) for p in rows])
    preempt = np.asarray([safe_float(p.total_preemptions, default=np.nan) for p in rows])
    throughput = np.asarray([
        safe_float(p.total_token_throughput, default=np.nan) for p in rows
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    bottom = np.zeros_like(xs, dtype=float)
    ax.bar(xs, attention, label="attention", color=C_ATTN)
    bottom += attention
    ax.bar(xs, moe, bottom=bottom, label="MoE", color=C_MOE)
    bottom += moe
    ax.bar(xs, h2d, bottom=bottom, label="H2D copy", color=C_OFFLOAD)
    bottom += h2d
    ax.bar(xs, wait, bottom=bottom, label="prefetch wait", color=C_WAIT)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("ms / scheduled token")
    ax.set_title(f"Compute/offload cost breakdown at N={fmt_n(focus_n)}")
    ax.legend(loc="best")

    ax2 = axes[1]
    ax2.plot(xs, tpot, marker="o", color=C_TPOT, label="TPOT")
    ax2.set_xticks(xs, labels)
    ax2.set_ylabel("TPOT (ms/token)", color=C_TPOT)
    ax2.tick_params(axis="y", labelcolor=C_TPOT)
    ax2.set_title("End-to-end symptom vs pressure")
    ax2b = ax2.twinx()
    ax2b.plot(xs, preempt, marker="s", color=C_PREEMPT, label="total preemptions")
    ax2b.plot(xs, throughput, marker="^", color=C_TPUT, label="throughput")
    ax2b.set_ylabel("total preemptions or tokens/s")
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [line.get_label() for line in lines], loc="best")

    fig.savefig(figure_dir / "06_cost_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency_tail(points: list[Point], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = [
        ("mean_ttft_ms", "Mean TTFT", "ms", "{:.0f}", "YlOrBr"),
        ("p99_ttft_ms", "P99 TTFT", "ms", "{:.0f}", "YlOrBr"),
        ("mean_itl_ms", "Mean ITL", "ms", "{:.0f}", "Purples"),
        ("p99_itl_ms", "P99 ITL", "ms", "{:.0f}", "Purples"),
    ]
    for ax, (key, title, label, fmt, cmap) in zip(axes.flat, specs):
        plot_heatmap(ax, points, key, title, label, fmt, cmap, best="min")
    fig.suptitle("Latency distribution diagnostics", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "07_latency_tail.png", bbox_inches="tight")
    plt.close(fig)


def plot_scatter_diagnostics(points: list[Point], figure_dir: Path) -> None:
    rows = [p for p in points if p.complete]
    if not rows:
        return
    r_values = sorted({p.r_value for p in rows})
    color_map = {r: plt.get_cmap("viridis")(i / max(1, len(r_values) - 1))
                 for i, r in enumerate(r_values)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    scatter_specs = [
        (
            axes[0, 0],
            "KV capacity vs preemption",
            "KV cache tokens",
            "preemptions/request",
            lambda p: p.kv_cache_tokens,
            lambda p: p.preemptions_per_request,
        ),
        (
            axes[0, 1],
            "Offload overhead vs TPOT",
            "offload ms/scheduled token",
            "mean TPOT (ms/token)",
            lambda p: p.offload_ms_per_scheduled_token,
            lambda p: p.mean_tpot_ms,
        ),
        (
            axes[1, 0],
            "Peak KV pressure vs throughput",
            "peak KV usage (%)",
            "total token throughput",
            lambda p: (p.peak_kv_cache_usage * 100.0 if p.peak_kv_cache_usage is not None else None),
            lambda p: p.total_token_throughput,
        ),
        (
            axes[1, 1],
            "Active expert ratio vs MoE cost",
            "avg active expert ratio",
            "MoE ms/scheduled token",
            lambda p: p.avg_active_expert_ratio,
            lambda p: p.moe_ms_per_scheduled_token,
        ),
    ]

    for ax, title, xlabel, ylabel, x_getter, y_getter in scatter_specs:
        for point in rows:
            x = safe_float(x_getter(point))
            y = safe_float(y_getter(point))
            if x is None or y is None:
                continue
            size = 28 + 80 * safe_float(point.n_value, default=1.0) / max(p.n_value for p in rows)
            ax.scatter(x, y, s=size, color=color_map[point.r_value], alpha=0.8,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(point.point, (x, y), fontsize=6, alpha=0.75, xytext=(2, 2),
                        textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    handles = [plt.Line2D([0], [0], marker="o", color="w", label=f"R{r}",
                          markerfacecolor=color_map[r], markersize=7)
               for r in r_values]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles),
               title="Offload ratio")
    fig.savefig(figure_dir / "08_scatter_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_auxiliary_heatmaps(points: list[Point], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = [
        ("avg_active_expert_ratio", "Avg active expert ratio", "ratio", "{:.2f}", "GnBu", 1.0),
        ("prefix_cache_hit_rate", "Prefix cache hit rate\n(auxiliary, not KV capacity hit rate)", "ratio", "{:.2f}", "Greys", 1.0),
        ("h2d_copy_gib", "Total H2D copy volume", "GiB", "{:.1f}", "Oranges", 1.0),
        ("tokens_per_kv_cache_token", "Throughput per KV capacity token", "tok/s per KV token", "{:.3f}", "Blues", 1.0),
    ]
    for ax, (key, title, label, fmt, cmap, scale) in zip(axes.flat, specs):
        plot_heatmap(ax, points, key, title, label, fmt, cmap, scale=scale)
    fig.suptitle("Auxiliary diagnostics", fontsize=14, fontweight="bold")
    fig.savefig(figure_dir / "09_auxiliary_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    run_dir: Path,
    points: list[Point],
    best_rows: list[dict[str, Any]],
    focus_n: float,
    output_dir: Path,
) -> None:
    completed = [p for p in points if p.complete]
    total_preemptions = sum(safe_int(p.total_preemptions, default=0) or 0 for p in completed)
    max_avg_kv = max((safe_float(p.avg_kv_cache_usage, default=0.0) or 0.0) for p in completed)
    max_kv_point = max(completed, key=lambda p: safe_float(p.avg_kv_cache_usage, default=0.0) or 0.0)
    max_peak_kv = max((safe_float(p.peak_kv_cache_usage, default=0.0) or 0.0) for p in completed)
    max_peak_point = max(completed, key=lambda p: safe_float(p.peak_kv_cache_usage, default=0.0) or 0.0)
    input_per_req = np.nanmean([
        safe_float(p.input_tokens_per_request, default=np.nan) for p in completed
    ])
    output_per_req = np.nanmean([
        safe_float(p.output_tokens_per_request, default=np.nan) for p in completed
    ])

    config = read_run_config(run_dir)
    capacity_by_r: dict[int, Point] = {}
    for point in completed:
        if point.kv_cache_memory_gib is None and point.kv_cache_tokens is None:
            continue
        capacity_by_r.setdefault(point.r_value, point)

    lines = [
        "Expert/KV contention focused analysis",
        "=======================================",
        f"Run directory: {run_dir}",
        f"Output directory: {output_dir}",
        f"Completed points: {len(completed)} / {len(points)}",
        f"R values: {', '.join('R' + str(r) for r in sorted({p.r_value for p in points}))}",
        f"N values: {', '.join(fmt_n(n) for n in sorted({p.n_value for p in points}))}",
    ]
    if config:
        lines.extend([
            "",
            "Run config",
            "----------",
            f"GPU_ID={config.get('GPU_ID', '')}, NUMA_NODE={config.get('NUMA_NODE', '')}, PORT={config.get('PORT', '')}",
            f"GPU_MEMORY_UTILIZATION={config.get('GPU_MEMORY_UTILIZATION', '')}",
            f"BENCH_NUM_PROMPTS={config.get('BENCH_NUM_PROMPTS', '')}, SHAREGPT_OUTPUT_LEN={config.get('SHAREGPT_OUTPUT_LEN', '')}",
            f"BENCH_MAX_CONCURRENCY={config.get('BENCH_MAX_CONCURRENCY', '')}, MAX_NUM_SEQS={config.get('MAX_NUM_SEQS', '')}",
        ])

    lines.extend([
        "",
        "Key diagnosis",
        "-------------",
        f"Total preemptions: {total_preemptions}",
        f"Max average KV usage: {max_avg_kv * 100:.1f}% at {max_kv_point.point}",
        f"Max logged peak KV usage: {max_peak_kv * 100:.1f}% at {max_peak_point.point}",
        f"Average input tokens/request: {input_per_req:.1f}",
        f"Average output tokens/request: {output_per_req:.1f}",
    ])

    if total_preemptions == 0 and max_avg_kv < 0.8:
        lines.extend([
            "",
            "Conclusion: this run did not reach KV-cache contention.",
            "Offload therefore cannot show its intended benefit. It only adds H2D",
            "copy and prefetch-wait cost, so R0/all-HBM wins for every measured N.",
        ])
    elif total_preemptions == 0:
        lines.extend([
            "",
            "Conclusion: KV usage became high, but no preemption was observed.",
            "This is still not strong evidence for the intended contention effect.",
        ])
    else:
        lines.extend([
            "",
            "Conclusion: preemption was observed. Check whether best R shifts with N",
            "and whether the shift coincides with lower KV pressure/recompute cost.",
        ])

    lines.extend([
        "",
        "Best R by request rate",
        "----------------------",
        "N | best throughput | best TPOT | best/R0 throughput | best/R0 TPOT",
    ])
    for row in best_rows:
        lines.append(
            f"{fmt_n(float(row['n_value']))} | "
            f"{row['best_total_tput_point']} ({row['best_total_tput']:.1f} tok/s) | "
            f"{row['best_tpot_point']} ({row['best_tpot_ms']:.2f} ms) | "
            f"{row['best_total_tput_vs_r0']:.3f} | "
            f"{row['best_tpot_vs_r0']:.3f}"
        )

    if capacity_by_r:
        lines.extend([
            "",
            "Observed KV capacity by R",
            "-------------------------",
            "R | KV memory GiB | KV tokens",
        ])
        for r_value, point in sorted(capacity_by_r.items()):
            mem = "" if point.kv_cache_memory_gib is None else f"{point.kv_cache_memory_gib:.2f}"
            tokens = "" if point.kv_cache_tokens is None else f"{point.kv_cache_tokens:,}"
            lines.append(f"R{r_value} | {mem} | {tokens}")

    focus_rows = sorted([p for p in completed if p.n_value == focus_n], key=lambda p: p.r_value)
    if focus_rows:
        lines.extend([
            "",
            f"Focus N={fmt_n(focus_n)} tradeoff",
            "---------------------",
            "R | throughput | TPOT ms | avg KV % | peak KV % | preempt/req | offload ms/scheduled-token",
        ])
        for point in focus_rows:
            lines.append(
                f"R{point.r_value} | "
                f"{point.total_token_throughput or 0.0:.1f} | "
                f"{point.mean_tpot_ms or 0.0:.2f} | "
                f"{(point.avg_kv_cache_usage or 0.0) * 100:.1f} | "
                f"{(point.peak_kv_cache_usage or 0.0) * 100:.1f} | "
                f"{point.preemptions_per_request or 0.0:.4f} | "
                f"{point.offload_ms_per_scheduled_token or 0.0:.4f}"
            )

    lines.extend([
        "",
        "Generated figures",
        "-----------------",
        "00_main_summary.png: throughput, TPOT, avg KV, offload overhead.",
        "01_best_r_by_n.png: best R and best/R0 ratios by request rate.",
        "02_focus_n_tradeoff.png: focused R tradeoff at selected N; preemption is raw total count.",
        "02b_focus_n_tradeoff_all_n.png: the same tradeoff expanded across all completed N values.",
        "03_pressure_summary.png: preemption, avg/peak KV, KV capacity.",
        "04_curves_by_n.png: metric curves across R for each N.",
        "05_relative_to_r0.png: R0-relative throughput/TPOT/preemption/KV.",
        "06_cost_breakdown.png: attention/MoE/H2D/wait cost breakdown.",
        "07_latency_tail.png: mean/P99 TTFT and ITL heatmaps.",
        "08_scatter_diagnostics.png: scatter views for cross-metric relationships.",
        "09_auxiliary_diagnostics.png: active expert, prefix cache, copy volume, capacity efficiency.",
        "",
        "Note: prefix-cache hit rate is an auxiliary shared-prefix metric. Do not read",
        "it as decode KV-cache capacity hit rate.",
    ])

    (output_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else latest_run(args.result_root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "analysis_focused"
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(run_dir)
    completed = [p for p in points if p.complete]
    if not completed:
        raise RuntimeError(f"no completed points with bench.json found under {run_dir}")

    best_rows = best_by_n(points)
    focus_n = pick_focus_n(completed, args.focus_n)

    write_point_summary(points, output_dir)
    write_best_by_n(best_rows, output_dir)
    write_summary(run_dir, points, best_rows, focus_n, output_dir)
    plot_main_summary(completed, figure_dir)
    plot_best_r(best_rows, figure_dir)
    plot_focus_tradeoff(completed, focus_n, figure_dir)
    plot_all_n_tradeoffs(completed, figure_dir)
    plot_pressure_summary(completed, figure_dir)
    plot_curves_by_n(completed, figure_dir)
    plot_relative_to_r0(completed, figure_dir)
    plot_cost_breakdown(completed, focus_n, figure_dir)
    plot_latency_tail(completed, figure_dir)
    plot_scatter_diagnostics(completed, figure_dir)
    plot_auxiliary_heatmaps(completed, figure_dir)

    total_preemptions = sum(safe_int(p.total_preemptions, default=0) or 0 for p in completed)
    max_avg_kv = max((safe_float(p.avg_kv_cache_usage, default=0.0) or 0.0) for p in completed)
    max_peak_kv = max((safe_float(p.peak_kv_cache_usage, default=0.0) or 0.0) for p in completed)
    print(f"wrote focused analysis to {output_dir}")
    print(
        f"total_preemptions={total_preemptions}, "
        f"max_avg_kv={max_avg_kv * 100:.1f}%, "
        f"max_peak_kv={max_peak_kv * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
