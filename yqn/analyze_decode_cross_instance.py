#!/usr/bin/env python
"""
Cross-Instance Decode Attention Comparison
===========================================
Reads two Nsight Systems SQLite exports (one per decode instance), aligns
``decode_attn`` NVTX events by step number, and generates three publication-
quality charts showing the attention-time imbalance between instances.

**Why this matters for MoE models:**
After attention, every decode step hits an all2all collective before expert
routing.  The faster instance must *wait* at this barrier for the slower one.
The per-step |attn1 - attn2| delta is the "bubble time" wasted in each step.

Usage:
    python analyze_decode_cross_instance.py <decode1.sqlite> <decode2.sqlite> \
        [--output-root DIR] [--labels LABEL1 LABEL2]

    # Or use defaults (reads from logs_dp_2p2d_nixl/decode_attn_nsys/):
    python analyze_decode_cross_instance.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NSYS_DIR = (
    REPO_ROOT / "yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "yqn/figures/decode_cross_instance"

LABEL_PATTERN = re.compile(
    r"(step|layer|rank|local_rank|num_reqs|num_tokens"
    r"|req_key_kind|layer_name|req_keys|qlens|slens)"
    r"=((?:\[[^\]]*\])|(?:\S+))",
)

# ---------------------------------------------------------------------------
# Matplotlib — publication-quality defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "lines.linewidth": 1.2,
})

# Color palette — colorblind-friendly
C_INST1 = "#0072B2"   # blue
C_INST2 = "#D55E00"   # vermillion
C_BUBBLE = "#CC79A7"  # reddish-pink for bubble bars
C_CUM = "#009E73"     # teal for cumulative line


# ===================================================================
# Data classes
# ===================================================================
@dataclass
class StepSummary:
    """Aggregated attention info for one step in one instance."""
    step: int
    total_attn_ms: float
    num_layers: int
    avg_attn_per_layer_ms: float
    num_reqs: int
    num_tokens: int
    mean_slen: float | None
    mean_qlen: float | None


# ===================================================================
# CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare decode attention between two instances "
            "(MoE all2all bubble analysis)."
        ),
    )
    p.add_argument(
        "inputs", nargs="*", type=Path,
        help="Two .sqlite files (decode1 and decode2). "
             "If omitted, auto-discovers from default nsys dir.",
    )
    p.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated figures and CSVs.",
    )
    p.add_argument(
        "--labels", nargs=2, default=["Decode-1", "Decode-2"],
        help="Display labels for the two instances.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Dump SQLite schema + sample rows for debugging, then exit.",
    )
    p.add_argument(
        "--ylim-pct", type=float, default=98.0,
        help="Percentile for y-axis upper limit (default 98, clips outliers).",
    )
    return p.parse_args()


def auto_discover_inputs() -> list[Path]:
    """Find decode1.sqlite and decode2.sqlite in the default directory."""
    if not DEFAULT_NSYS_DIR.exists():
        return []
    found = sorted(DEFAULT_NSYS_DIR.glob("decode*.sqlite"))
    return found[:2]


# ===================================================================
# Debug helper
# ===================================================================
def dump_sqlite_debug(sqlite_path: Path) -> None:
    """Print every table's schema + first 5 rows for debugging."""
    conn = sqlite3.connect(sqlite_path)
    print(f"\n{'=' * 70}")
    print(f"DEBUG: {sqlite_path}")
    print(f"{'=' * 70}")

    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    print(f"Tables: {tables}")

    for table in tables:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols]
        print(f"\n--- {table} ---")
        print(f"  Columns: {col_names}")
        print(f"  Schema:  {cols}")
        try:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  Row count: {count}")
        except Exception:
            print("  Row count: <error>")

        try:
            rows = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchall()
            for i, row in enumerate(rows):
                print(f"  Row {i}: {row}")
        except Exception as e:
            print(f"  Sample rows error: {e}")

        if "NVTX" in table.upper():
            for text_col in (
                "textId", "nameId", "text", "name", "shortName", "shortNameId",
            ):
                if text_col in col_names:
                    print(f"\n  Checking {text_col} for decode_attn markers...")
                    try:
                        sample = conn.execute(
                            f"SELECT {text_col} FROM {table} LIMIT 5"
                        ).fetchall()
                        print(f"    Raw sample values: {sample}")
                        col_info = next(
                            (c for c in cols if c[1] == text_col), None
                        )
                        if col_info and "INT" in str(col_info[2]).upper():
                            str_tables = [
                                r[0]
                                for r in conn.execute(
                                    "SELECT name FROM sqlite_master "
                                    "WHERE type='table' AND name LIKE '%tring%'"
                                ).fetchall()
                            ]
                            print(f"    String tables: {str_tables}")
                            for st in str_tables:
                                st_cols = [
                                    c[1]
                                    for c in conn.execute(
                                        f"PRAGMA table_info({st})"
                                    ).fetchall()
                                ]
                                print(f"    {st} columns: {st_cols}")
                                for row in sample:
                                    tid = row[0]
                                    if tid is not None:
                                        try:
                                            resolved = conn.execute(
                                                f"SELECT * FROM {st} WHERE id = ?",
                                                (tid,),
                                            ).fetchone()
                                            print(f"      id={tid} -> {resolved}")
                                        except Exception as e2:
                                            print(f"      id={tid} -> error: {e2}")
                    except Exception as e:
                        print(f"    Error: {e}")

            for text_col in ("text", "name"):
                if text_col in col_names:
                    try:
                        hits = conn.execute(
                            f"SELECT count(*) FROM {table} "
                            f"WHERE {text_col} LIKE '%decode_attn%'"
                        ).fetchone()[0]
                        print(f"\n  '{text_col} LIKE decode_attn': {hits} matches")
                    except Exception:
                        pass

    conn.close()


# ===================================================================
# SQLite helpers
# ===================================================================
NVTX_TABLE_NAMES = (
    "NVTX_EVENTS",
    "NVTX_PUSHPOP_EVENTS",
    "NVTX_RANGES",
    "ANALYSIS_NVTX_RANGES",
    "NVTX_GPU_PROJ_TRACE",
)

_START_COL_CANDIDATES = ("start", "startTimestamp", "Start", "start_time", "startNs")
_END_COL_CANDIDATES = ("end", "endTimestamp", "End", "end_time", "endNs")


def _first_table(
    conn: sqlite3.Connection,
    candidates: tuple[str, ...] | None = None,
) -> str | None:
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if candidates is None:
        candidates = NVTX_TABLE_NAMES
    for t in candidates:
        if t in tables:
            return t
    for t in sorted(tables):
        if "NVTX" in t.upper():
            print(f"  [fallback] Using NVTX table: {t}")
            return t
    return None


def _find_col(col_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in col_names:
            return c
    return None


def _name_expr(
    conn: sqlite3.Connection, table: str, alias: str = "t",
) -> tuple[str, str]:
    cols = {
        r[1]: r
        for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }

    str_table = None
    str_id_col = "id"
    str_val_col = "value"
    all_tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for candidate_table in ("StringIds", "StringTable", "Strings"):
        if candidate_table in all_tables:
            st_cols = {
                r[1]
                for r in conn.execute(
                    f"PRAGMA table_info({candidate_table})"
                ).fetchall()
            }
            if "id" in st_cols and "value" in st_cols:
                str_table = candidate_table
                break
            if "Id" in st_cols:
                str_id_col = "Id"
            if "Value" in st_cols:
                str_val_col = "Value"
            if "string" in st_cols:
                str_val_col = "string"
            str_table = candidate_table
            break

    # PRIORITY 1: Direct text columns
    for c in ("text", "name", "Text", "Name"):
        if c in cols:
            col_type = str(cols[c][2]).upper()
            if "INT" not in col_type:
                return f"{alias}.{c}", ""

    # PRIORITY 2: Integer text-id columns needing StringIds JOIN
    for c in ("textId", "nameId", "shortName", "shortNameId"):
        r = cols.get(c)
        if r and "INT" in str(r[2]).upper():
            if str_table:
                return (
                    f"COALESCE(s.{str_val_col}, CAST({alias}.{c} AS TEXT))",
                    f"LEFT JOIN {str_table} s ON {alias}.{c} = s.{str_id_col}",
                )
            else:
                return f"CAST({alias}.{c} AS TEXT)", ""

    return "'<unnamed>'", ""


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
# Read & aggregate per step
# ===================================================================
def read_step_summaries(sqlite_path: Path) -> dict[int, StepSummary]:
    """Read NVTX decode_attn events and aggregate per step."""
    conn = sqlite3.connect(sqlite_path)
    try:
        nvtx_table = _first_table(conn)
        if nvtx_table is None:
            print(
                f"  WARN: No NVTX table in {sqlite_path.name}", file=sys.stderr,
            )
            return {}

        col_names = {
            r[1]
            for r in conn.execute(
                f"PRAGMA table_info({nvtx_table})"
            ).fetchall()
        }
        start_col = _find_col(col_names, _START_COL_CANDIDATES)
        end_col = _find_col(col_names, _END_COL_CANDIDATES)
        if not start_col or not end_col:
            print(
                f"  WARN: Cannot find start/end columns in {nvtx_table}. "
                f"Available columns: {sorted(col_names)}",
                file=sys.stderr,
            )
            return {}

        expr, join = _name_expr(conn, nvtx_table)
        sql = f"""
            SELECT label, ts_start, ts_end FROM (
                SELECT {expr} AS label,
                       t.{start_col} AS ts_start,
                       t.{end_col}   AS ts_end
                FROM {nvtx_table} t {join}
            ) WHERE label LIKE 'decode_attn %' ORDER BY ts_start
        """
        step_total_ns: dict[int, float] = defaultdict(float)
        step_layers: dict[int, set[int]] = defaultdict(set)
        step_nreqs: dict[int, int] = {}
        step_ntokens: dict[int, int] = {}
        step_slens: dict[int, list[int]] = defaultdict(list)
        step_qlens: dict[int, list[int]] = defaultdict(list)

        n_events = 0
        for label, s, e in conn.execute(sql).fetchall():
            label = str(label)
            if not label.startswith("decode_attn "):
                continue
            fields = dict(LABEL_PATTERN.findall(label))
            step = _int(fields.get("step"), -1)
            layer = _int(fields.get("layer"), -1)
            duration_ns = max(0, int(e) - int(s))
            num_reqs = _int(fields.get("num_reqs"), 0)
            num_tokens = _int(fields.get("num_tokens"), 0)
            slens = _list_int(fields.get("slens", "[]"))
            qlens = _list_int(fields.get("qlens", "[]"))

            step_total_ns[step] += duration_ns
            step_layers[step].add(layer)
            step_nreqs[step] = num_reqs
            step_ntokens[step] = num_tokens
            step_slens[step].extend(slens)
            step_qlens[step].extend(qlens)
            n_events += 1

        print(
            f"  {sqlite_path.name}: {n_events} decode_attn events, "
            f"{len(step_total_ns)} steps"
        )

        summaries: dict[int, StepSummary] = {}
        for step in sorted(step_total_ns):
            total_ms = step_total_ns[step] / 1e6
            n_layers = len(step_layers[step])
            sl = step_slens[step]
            ql = step_qlens[step]
            summaries[step] = StepSummary(
                step=step,
                total_attn_ms=total_ms,
                num_layers=n_layers,
                avg_attn_per_layer_ms=total_ms / n_layers if n_layers else 0,
                num_reqs=step_nreqs.get(step, 0),
                num_tokens=step_ntokens.get(step, 0),
                mean_slen=float(np.mean(sl)) if sl else None,
                mean_qlen=float(np.mean(ql)) if ql else None,
            )
        return summaries
    finally:
        conn.close()


def align_steps(
    s1: dict[int, StepSummary], s2: dict[int, StepSummary],
) -> list[int]:
    """Return sorted list of steps that exist in both instances."""
    common = sorted(set(s1.keys()) & set(s2.keys()))
    only1 = set(s1.keys()) - set(s2.keys())
    only2 = set(s2.keys()) - set(s1.keys())
    if only1:
        print(f"  Steps only in instance 1: {len(only1)} (skipped)")
    if only2:
        print(f"  Steps only in instance 2: {len(only2)} (skipped)")
    print(f"  Common steps: {len(common)}")
    return common


# ===================================================================
# Y-axis helper: percentile-based limits to clip outliers
# ===================================================================
def _robust_ylim(
    values: np.ndarray, pct: float = 98.0, symmetric: bool = False,
) -> tuple[float, float]:
    """Return (ymin, ymax) clipped at the given percentile + 10% padding."""
    if len(values) == 0:
        return (0.0, 1.0)
    if symmetric:
        abs_vals = np.abs(values)
        cap = float(np.percentile(abs_vals, pct))
        pad = cap * 0.12
        return (-(cap + pad), cap + pad)
    lo = float(np.percentile(values, 100 - pct))
    hi = float(np.percentile(values, pct))
    rng = hi - lo if hi > lo else abs(hi) * 0.1 + 1e-6
    pad = rng * 0.10
    return (max(0, lo - pad), hi + pad)


def _stat_text(values: np.ndarray, label: str) -> str:
    """Format a compact statistics block for annotation."""
    return (
        f"{label}\n"
        f"  mean = {np.mean(values):.3f} ms\n"
        f"  med  = {np.median(values):.3f} ms\n"
        f"  std  = {np.std(values):.3f} ms\n"
        f"  P5   = {np.percentile(values, 5):.3f} ms\n"
        f"  P95  = {np.percentile(values, 95):.3f} ms"
    )


# ===================================================================
# CSV output
# ===================================================================
def write_comparison_csv(
    path: Path,
    steps: list[int],
    s1: dict[int, StepSummary],
    s2: dict[int, StepSummary],
    labels: list[str],
) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step",
            f"{labels[0]}_total_attn_ms",
            f"{labels[1]}_total_attn_ms",
            "delta_ms",
            "abs_delta_ms",
            f"{labels[0]}_num_reqs",
            f"{labels[1]}_num_reqs",
            f"{labels[0]}_num_tokens",
            f"{labels[1]}_num_tokens",
            f"{labels[0]}_mean_slen",
            f"{labels[1]}_mean_slen",
        ])
        for step in steps:
            a, b = s1[step], s2[step]
            delta = a.total_attn_ms - b.total_attn_ms
            w.writerow([
                step,
                f"{a.total_attn_ms:.6f}",
                f"{b.total_attn_ms:.6f}",
                f"{delta:.6f}",
                f"{abs(delta):.6f}",
                a.num_reqs,
                b.num_reqs,
                a.num_tokens,
                b.num_tokens,
                "" if a.mean_slen is None else f"{a.mean_slen:.1f}",
                "" if b.mean_slen is None else f"{b.mean_slen:.1f}",
            ])


# ===================================================================
# Fig 1: Per-Step Attention Overlay (sampled windows of 200 steps)
# ===================================================================
_OVERLAY_WINDOW = 200


def _pick_overlay_windows(n_steps: int, window: int) -> list[int]:
    """Return start indices for 4 evenly-spaced windows (or fewer if short)."""
    if n_steps <= window:
        return [0]
    n_windows = min(4, max(1, n_steps // window))
    last_start = n_steps - window
    if n_windows == 1:
        return [0]
    stride = last_start / (n_windows - 1)
    return [int(round(i * stride)) for i in range(n_windows)]


def plot_step_overlays(
    fig_dir: Path,
    steps: list[int],
    s1: dict[int, StepSummary],
    s2: dict[int, StepSummary],
    labels: list[str],
    ylim_pct: float,
) -> list[Path]:
    """Generate one overlay chart per sampled window."""
    starts = _pick_overlay_windows(len(steps), _OVERLAY_WINDOW)
    paths: list[Path] = []

    for idx, start in enumerate(starts):
        end = min(start + _OVERLAY_WINDOW, len(steps))
        window_steps = steps[start:end]

        suffix = chr(ord("a") + idx)  # a, b, c, d, ...
        path = fig_dir / f"01{suffix}_step_attn_overlay.png"

        y1 = np.array([s1[s].total_attn_ms for s in window_steps])
        y2 = np.array([s2[s].total_attn_ms for s in window_steps])
        xs = np.array(window_steps)

        fig, ax = plt.subplots(figsize=(14, 5))

        # Shaded gap
        ax.fill_between(
            xs, y1, y2, alpha=0.15, color="#888888",
            label="Attention gap (bubble)",
        )

        # Real data lines
        ax.plot(xs, y1, "-", linewidth=1.0, color=C_INST1, alpha=0.85,
                label=labels[0])
        ax.plot(xs, y2, "-", linewidth=1.0, color=C_INST2, alpha=0.85,
                label=labels[1])

        # Y-axis
        all_vals = np.concatenate([y1, y2])
        ymin, ymax = _robust_ylim(all_vals, pct=ylim_pct)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Decode Step")
        ax.set_ylabel("Total Attention Time (ms)")
        ax.set_title(
            f"Per-Step Decode Attention: Steps {window_steps[0]}"
            f"-{window_steps[-1]}  ({idx + 1}/{len(starts)})"
        )
        ax.legend(loc="upper right", frameon=True)

        # Stats box
        stats = (
            f"{labels[0]}:  mean={np.mean(y1):.3f},  "
            f"med={np.median(y1):.3f} ms\n"
            f"{labels[1]}:  mean={np.mean(y2):.3f},  "
            f"med={np.median(y2):.3f} ms"
        )
        ax.text(
            0.01, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="minor", length=3)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [chart] {path.name}")
        paths.append(path)

    return paths


# ===================================================================
# Fig 2: Per-Step Bubble Time (signed delta, raw data)
# ===================================================================
def plot_step_bubble(
    fig_dir: Path,
    steps: list[int],
    s1: dict[int, StepSummary],
    s2: dict[int, StepSummary],
    labels: list[str],
    ylim_pct: float,
) -> Path:
    path = fig_dir / "02_step_bubble_time.png"

    deltas = np.array([
        s1[s].total_attn_ms - s2[s].total_attn_ms for s in steps
    ])
    abs_deltas = np.abs(deltas)
    xs = np.array(steps)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Raw bars colored by direction
    colors = np.where(deltas > 0, C_INST1, C_INST2)
    ax.bar(xs, deltas, color=colors, alpha=0.70, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)

    # Y-axis: symmetric, clipped
    ymin, ymax = _robust_ylim(deltas, pct=ylim_pct, symmetric=True)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Decode Step")
    ax.set_ylabel(
        f"$\\Delta t_{{\\mathrm{{attn}}}}$ = "
        f"{labels[0]} $-$ {labels[1]}  (ms)"
    )
    ax.set_title(
        "Per-Step Attention Imbalance (All2All Bubble Time)"
    )

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_INST1, alpha=0.7,
              label=f"$\\Delta > 0$: {labels[0]} slower"),
        Patch(facecolor=C_INST2, alpha=0.7,
              label=f"$\\Delta < 0$: {labels[1]} slower"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    # Stats box
    n_total = len(deltas)
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    stats = (
        f"Total steps:  {n_total}\n"
        f"{labels[0]} slower:  {n_pos}  ({n_pos / n_total * 100:.1f}%)\n"
        f"{labels[1]} slower:  {n_neg}  ({n_neg / n_total * 100:.1f}%)\n"
        f"\n"
        f"|$\\Delta$|  mean = {np.mean(abs_deltas):.3f} ms\n"
        f"|$\\Delta$|  med  = {np.median(abs_deltas):.3f} ms\n"
        f"|$\\Delta$|  P95  = {np.percentile(abs_deltas, 95):.3f} ms\n"
        f"|$\\Delta$|  max  = {np.max(abs_deltas):.3f} ms"
    )
    ax.text(
        0.01, 0.97, stats, transform=ax.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="minor", length=3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Fig 3: Cumulative Bubble Time
# ===================================================================
def plot_cumulative_bubble(
    fig_dir: Path,
    steps: list[int],
    s1: dict[int, StepSummary],
    s2: dict[int, StepSummary],
    labels: list[str],
    ylim_pct: float,
) -> Path:
    path = fig_dir / "03_cumulative_bubble_time.png"

    abs_deltas = np.array([
        abs(s1[s].total_attn_ms - s2[s].total_attn_ms) for s in steps
    ])
    cum = np.cumsum(abs_deltas)
    xs = np.array(steps)

    total_bubble = float(cum[-1]) if len(cum) > 0 else 0.0
    total_max_attn = sum(
        max(s1[s].total_attn_ms, s2[s].total_attn_ms) for s in steps
    )
    overhead_pct = (
        (total_bubble / total_max_attn * 100) if total_max_attn > 0 else 0.0
    )

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Left axis: per-step |delta| bars
    _, bar_ymax = _robust_ylim(abs_deltas, pct=ylim_pct)
    ax1.bar(
        xs, abs_deltas, color=C_BUBBLE, alpha=0.50, width=1.0,
        edgecolor="none", label="$|\\Delta t_{\\mathrm{attn}}|$ per step",
    )
    ax1.set_ylim(0, bar_ymax)
    ax1.set_xlabel("Decode Step")
    ax1.set_ylabel("$|\\Delta t_{\\mathrm{attn}}|$ per Step (ms)", color=C_BUBBLE)
    ax1.tick_params(axis="y", labelcolor=C_BUBBLE)

    # Right axis: cumulative line
    ax2 = ax1.twinx()
    ax2.plot(
        xs, cum, "-", linewidth=2.0, color=C_CUM,
        label="Cumulative bubble time",
    )
    ax2.fill_between(xs, cum, alpha=0.08, color=C_CUM)
    ax2.set_ylabel("Cumulative Bubble Time (ms)", color=C_CUM)
    ax2.tick_params(axis="y", labelcolor=C_CUM)

    ax1.set_title(
        f"Cumulative All2All Bubble Time:  "
        f"{total_bubble:.1f} ms total  "
        f"({overhead_pct:.1f}% overhead)"
    )

    # Merged legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labs1 + labs2,
        loc="upper left", frameon=True,
    )

    # Stats box (right side)
    stats = (
        f"Total bubble:     {total_bubble:.2f} ms\n"
        f"Overhead:         {overhead_pct:.2f}%\n"
        f"Per-step mean:    {np.mean(abs_deltas):.3f} ms\n"
        f"Per-step median:  {np.median(abs_deltas):.3f} ms\n"
        f"Per-step P95:     {np.percentile(abs_deltas, 95):.3f} ms\n"
        f"Per-step max:     {np.max(abs_deltas):.3f} ms"
    )
    ax1.text(
        0.99, 0.50, stats, transform=ax1.transAxes,
        fontsize=9, va="center", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    args = parse_args()

    inputs = args.inputs
    if not inputs:
        inputs = auto_discover_inputs()

    # --debug mode
    if args.debug:
        if not inputs:
            print("ERROR: No .sqlite files found for --debug.", file=sys.stderr)
            return 1
        for f in inputs:
            fp = Path(f).resolve()
            if fp.exists():
                dump_sqlite_debug(fp)
            else:
                print(f"File not found: {fp}", file=sys.stderr)
        return 0

    if len(inputs) < 2:
        print(
            "ERROR: Need exactly 2 .sqlite files (decode instance 1 and 2).",
            file=sys.stderr,
        )
        print(
            "  Pass them as arguments or place decode1.sqlite & "
            "decode2.sqlite in:",
            file=sys.stderr,
        )
        print(f"  {DEFAULT_NSYS_DIR}", file=sys.stderr)
        return 1

    file1, file2 = Path(inputs[0]).resolve(), Path(inputs[1]).resolve()
    for f in (file1, file2):
        if not f.exists():
            print(f"ERROR: File not found: {f}", file=sys.stderr)
            return 1

    labels = args.labels
    output_root = args.output_root.resolve()
    fig_dir = output_root / "figures"
    csv_dir = output_root / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Instance 1 ({labels[0]}): {file1.name}")
    print(f"Instance 2 ({labels[1]}): {file2.name}")

    s1 = read_step_summaries(file1)
    s2 = read_step_summaries(file2)

    if not s1:
        print(
            f"ERROR: No decode_attn events in {file1.name}", file=sys.stderr,
        )
        return 1
    if not s2:
        print(
            f"ERROR: No decode_attn events in {file2.name}", file=sys.stderr,
        )
        return 1

    steps = align_steps(s1, s2)
    if not steps:
        print(
            "ERROR: No common steps between the two instances.",
            file=sys.stderr,
        )
        return 1

    # CSV
    write_comparison_csv(
        csv_dir / "cross_instance_comparison.csv", steps, s1, s2, labels,
    )
    print("  [csv] cross_instance_comparison.csv")

    # Charts
    generated = []
    generated.extend(
        plot_step_overlays(fig_dir, steps, s1, s2, labels, args.ylim_pct)
    )
    generated.append(
        plot_step_bubble(fig_dir, steps, s1, s2, labels, args.ylim_pct)
    )
    generated.append(
        plot_cumulative_bubble(fig_dir, steps, s1, s2, labels, args.ylim_pct)
    )

    print(f"\nDone! {len(generated)} charts saved to: {fig_dir}")
    print(f"CSV saved to: {csv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
