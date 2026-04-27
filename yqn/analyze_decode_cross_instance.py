#!/usr/bin/env python
"""
Cross-Instance Decode Attention Comparison
===========================================
Reads two Nsight Systems SQLite exports (one per decode instance), aligns
``decode_attn`` NVTX events by step number, and generates comparative charts
that highlight the attention-time imbalance between instances.

**Why this matters for MoE models:**
After attention, every decode step hits an all2all collective before expert
routing.  The faster instance must *wait* at this barrier for the slower one.
The per-step |attn1 − attn2| delta is therefore the "bubble time" wasted in
each step.

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
DEFAULT_NSYS_DIR = REPO_ROOT / "yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "yqn/figures/decode_cross_instance"


LABEL_PATTERN = re.compile(
    r"(step|layer|rank|local_rank|num_reqs|num_tokens|req_key_kind|layer_name|req_keys|qlens|slens)"
    r"=((?:\[[^\]]*\])|(?:\S+))",
)

# ---------------------------------------------------------------------------
# Matplotlib style
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

C_INST1 = "#1f77b4"  # blue
C_INST2 = "#ff7f0e"  # orange
C_DELTA = "#d62728"  # red
C_FILL = "#2ca02c"   # green


# ===================================================================
# Data classes
# ===================================================================
@dataclass
class StepSummary:
    """Aggregated attention info for one step in one instance."""
    step: int
    total_attn_ms: float       # sum of all layer durations in this step
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
        description="Compare decode attention between two instances (MoE all2all bubble analysis).",
    )
    p.add_argument("inputs", nargs="*", type=Path,
                   help="Two .sqlite files (decode1 and decode2). "
                        "If omitted, auto-discovers from default nsys dir.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Directory for generated figures and CSVs.")
    p.add_argument("--labels", nargs=2, default=["Decode-1", "Decode-2"],
                   help="Display labels for the two instances.")
    p.add_argument("--debug", action="store_true",
                   help="Dump SQLite schema + sample rows for debugging, then exit.")
    return p.parse_args()


def auto_discover_inputs() -> list[Path]:
    """Find decode1.sqlite and decode2.sqlite in the default directory."""
    if not DEFAULT_NSYS_DIR.exists():
        return []
    found = sorted(DEFAULT_NSYS_DIR.glob("decode*.sqlite"))
    return found[:2]


# ===================================================================
# Debug: dump schema + sample rows
# ===================================================================
def dump_sqlite_debug(sqlite_path: Path) -> None:
    """Print every table's schema + first 5 rows for debugging."""
    conn = sqlite3.connect(sqlite_path)
    print(f"\n{'='*70}")
    print(f"DEBUG: {sqlite_path}")
    print(f"{'='*70}")

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
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
            print(f"  Row count: <error>")

        # Show sample rows
        try:
            rows = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchall()
            for i, row in enumerate(rows):
                print(f"  Row {i}: {row}")
        except Exception as e:
            print(f"  Sample rows error: {e}")

        # If this looks like an NVTX table, try to show decoded labels
        col_names_lower = {c.lower() for c in col_names}
        if any(kw in table.upper() for kw in ("NVTX",)):
            # Try to find text-like columns
            for text_col in ("textId", "nameId", "text", "name", "shortName", "shortNameId"):
                if text_col in col_names:
                    print(f"\n  Checking {text_col} for decode_attn markers...")
                    try:
                        # Try direct text match
                        sample = conn.execute(
                            f"SELECT {text_col} FROM {table} LIMIT 5"
                        ).fetchall()
                        print(f"    Raw sample values: {sample}")

                        # If integer, try StringIds join
                        col_info = next((c for c in cols if c[1] == text_col), None)
                        if col_info and "INT" in str(col_info[2]).upper():
                            # Check if StringIds exists
                            str_tables = [r[0] for r in conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%tring%'"
                            ).fetchall()]
                            print(f"    String tables: {str_tables}")
                            for st in str_tables:
                                st_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({st})").fetchall()]
                                print(f"    {st} columns: {st_cols}")
                                # Try to resolve a few IDs
                                for row in sample:
                                    tid = row[0]
                                    if tid is not None:
                                        try:
                                            resolved = conn.execute(
                                                f"SELECT * FROM {st} WHERE id = ?", (tid,)
                                            ).fetchone()
                                            print(f"      id={tid} -> {resolved}")
                                        except Exception as e2:
                                            print(f"      id={tid} -> error: {e2}")
                    except Exception as e:
                        print(f"    Error: {e}")

            # Also search for any row that might contain 'decode_attn' as text
            for text_col in ("text", "name"):
                if text_col in col_names:
                    try:
                        hits = conn.execute(
                            f"SELECT count(*) FROM {table} WHERE {text_col} LIKE '%decode_attn%'"
                        ).fetchone()[0]
                        print(f"\n  '{text_col} LIKE decode_attn': {hits} matches")
                    except Exception:
                        pass

    conn.close()


# ===================================================================
# SQLite helpers (reused logic from analyze_decode_attention_nsys.py)
# ===================================================================
NVTX_TABLE_NAMES = (
    "NVTX_EVENTS",
    "NVTX_PUSHPOP_EVENTS",
    # Older nsys versions:
    "NVTX_RANGES",
    "ANALYSIS_NVTX_RANGES",
    "NVTX_GPU_PROJ_TRACE",
)


def _first_table(conn: sqlite3.Connection, candidates: tuple[str, ...] | None = None) -> str | None:
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if candidates is None:
        candidates = NVTX_TABLE_NAMES
    for t in candidates:
        if t in tables:
            return t
    # Fallback: any table with NVTX in the name
    for t in sorted(tables):
        if "NVTX" in t.upper():
            print(f"  [fallback] Using NVTX table: {t}")
            return t
    return None


# Map of possible column name variants across nsys versions
_START_COL_CANDIDATES = ("start", "startTimestamp", "Start", "start_time", "startNs")
_END_COL_CANDIDATES = ("end", "endTimestamp", "End", "end_time", "endNs")


def _find_col(col_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in col_names:
            return c
    return None


def _name_expr(conn: sqlite3.Connection, table: str, alias: str = "t") -> tuple[str, str]:
    cols = {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    # Check for StringIds / StringTable
    str_table = None
    str_id_col = "id"
    str_val_col = "value"
    all_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for candidate_table in ("StringIds", "StringTable", "Strings"):
        if candidate_table in all_tables:
            st_cols = {r[1] for r in conn.execute(
                f"PRAGMA table_info({candidate_table})").fetchall()}
            if "id" in st_cols and "value" in st_cols:
                str_table = candidate_table
                break
            # Try other column name combinations
            if "Id" in st_cols:
                str_id_col = "Id"
            if "Value" in st_cols:
                str_val_col = "Value"
            if "string" in st_cols:
                str_val_col = "string"
            str_table = candidate_table
            break

    # PRIORITY 1: Direct text columns (no join needed, most reliable).
    # Check these FIRST because some schemas have both `text` (TEXT) and
    # `textId` (INTEGER/NULL) — using the direct column avoids a broken join.
    for c in ("text", "name", "Text", "Name"):
        if c in cols:
            col_type = str(cols[c][2]).upper()
            if "INT" not in col_type:  # genuine text column
                return f"{alias}.{c}", ""

    # PRIORITY 2: Integer text-id columns that need join to StringIds
    for c in ("textId", "nameId", "shortName", "shortNameId"):
        r = cols.get(c)
        if r and "INT" in str(r[2]).upper():
            if str_table:
                return (f"COALESCE(s.{str_val_col}, CAST({alias}.{c} AS TEXT))",
                        f"LEFT JOIN {str_table} s ON {alias}.{c} = s.{str_id_col}")
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
            print(f"  WARN: No NVTX table in {sqlite_path.name}", file=sys.stderr)
            return {}

        col_names = {r[1] for r in conn.execute(f"PRAGMA table_info({nvtx_table})").fetchall()}
        start_col = _find_col(col_names, _START_COL_CANDIDATES)
        end_col = _find_col(col_names, _END_COL_CANDIDATES)
        if not start_col or not end_col:
            print(f"  WARN: Cannot find start/end columns in {nvtx_table}. "
                  f"Available columns: {sorted(col_names)}", file=sys.stderr)
            return {}

        expr, join = _name_expr(conn, nvtx_table)
        sql = f"""
            SELECT label, ts_start, ts_end FROM (
                SELECT {expr} AS label,
                       t.{start_col} AS ts_start,
                       t.{end_col} AS ts_end
                FROM {nvtx_table} t {join}
            ) WHERE label LIKE 'decode_attn %' ORDER BY ts_start
        """
        # Per-step accumulators
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

        print(f"  {sqlite_path.name}: {n_events} decode_attn events, {len(step_total_ns)} steps")

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


def align_steps(s1: dict[int, StepSummary],
                s2: dict[int, StepSummary]) -> list[int]:
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
# CSV output
# ===================================================================
def write_comparison_csv(path: Path, steps: list[int],
                         s1: dict[int, StepSummary],
                         s2: dict[int, StepSummary],
                         labels: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step",
            f"{labels[0]}_total_attn_ms", f"{labels[1]}_total_attn_ms",
            "delta_ms", "abs_delta_ms",
            f"{labels[0]}_num_reqs", f"{labels[1]}_num_reqs",
            f"{labels[0]}_num_tokens", f"{labels[1]}_num_tokens",
            f"{labels[0]}_mean_slen", f"{labels[1]}_mean_slen",
        ])
        for step in steps:
            a, b = s1[step], s2[step]
            delta = a.total_attn_ms - b.total_attn_ms
            w.writerow([
                step,
                f"{a.total_attn_ms:.6f}", f"{b.total_attn_ms:.6f}",
                f"{delta:.6f}", f"{abs(delta):.6f}",
                a.num_reqs, b.num_reqs,
                a.num_tokens, b.num_tokens,
                "" if a.mean_slen is None else f"{a.mean_slen:.1f}",
                "" if b.mean_slen is None else f"{b.mean_slen:.1f}",
            ])


# ===================================================================
# Chart 1: Per-step overlay — two lines
# ===================================================================
def plot_step_overlay(fig_dir: Path, steps: list[int],
                      s1: dict[int, StepSummary],
                      s2: dict[int, StepSummary],
                      labels: list[str]) -> Path:
    path = fig_dir / "01_step_attn_overlay.png"
    y1 = [s1[s].total_attn_ms for s in steps]
    y2 = [s2[s].total_attn_ms for s in steps]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(steps, y1, "-", linewidth=1.0, color=C_INST1, alpha=0.8, label=labels[0])
    ax.plot(steps, y2, "-", linewidth=1.0, color=C_INST2, alpha=0.8, label=labels[1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Attention Time per Step (ms)")
    ax.set_title("Per-Step Decode Attention: Instance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 2: Per-step delta (bubble time)
# ===================================================================
def plot_step_delta(fig_dir: Path, steps: list[int],
                    s1: dict[int, StepSummary],
                    s2: dict[int, StepSummary],
                    labels: list[str]) -> Path:
    path = fig_dir / "02_step_bubble_time.png"
    deltas = [s1[s].total_attn_ms - s2[s].total_attn_ms for s in steps]

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [C_INST1 if d > 0 else C_INST2 for d in deltas]
    ax.bar(steps, deltas, color=colors, alpha=0.7, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Delta: {labels[0]} − {labels[1]} (ms)")
    ax.set_title("Per-Step Attention Imbalance (All2All Bubble Time)\n"
                 f"Positive = {labels[0]} slower, Negative = {labels[1]} slower")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 3: Absolute bubble time + cumulative
# ===================================================================
def plot_cumulative_bubble(fig_dir: Path, steps: list[int],
                           s1: dict[int, StepSummary],
                           s2: dict[int, StepSummary],
                           labels: list[str]) -> Path:
    path = fig_dir / "03_cumulative_bubble_time.png"
    abs_deltas = [abs(s1[s].total_attn_ms - s2[s].total_attn_ms) for s in steps]
    cum = np.cumsum(abs_deltas)

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(steps, abs_deltas, color=C_DELTA, alpha=0.4, width=1.0, label="|delta| per step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("|Delta| per Step (ms)", color=C_DELTA)
    ax1.tick_params(axis="y", labelcolor=C_DELTA)

    ax2 = ax1.twinx()
    ax2.plot(steps, cum, "-", linewidth=2, color=C_FILL, label="Cumulative bubble")
    ax2.set_ylabel("Cumulative Bubble Time (ms)", color=C_FILL)
    ax2.tick_params(axis="y", labelcolor=C_FILL)

    total_bubble = cum[-1] if len(cum) > 0 else 0
    total_attn = sum(max(s1[s].total_attn_ms, s2[s].total_attn_ms) for s in steps)
    overhead_pct = (total_bubble / total_attn * 100) if total_attn > 0 else 0

    ax1.set_title(
        f"Cumulative All2All Bubble Time: {total_bubble:.1f}ms total "
        f"({overhead_pct:.1f}% of max-instance attention)"
    )

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 4: Scatter — instance1 vs instance2
# ===================================================================
def plot_scatter_comparison(fig_dir: Path, steps: list[int],
                            s1: dict[int, StepSummary],
                            s2: dict[int, StepSummary],
                            labels: list[str]) -> Path:
    path = fig_dir / "04_scatter_inst1_vs_inst2.png"
    x = [s1[s].total_attn_ms for s in steps]
    y = [s2[s].total_attn_ms for s in steps]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, c=C_DELTA, alpha=0.3, s=15, edgecolors="none")

    # Diagonal
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="y=x (balanced)")

    ax.set_xlabel(f"{labels[0]} Attention per Step (ms)")
    ax.set_ylabel(f"{labels[1]} Attention per Step (ms)")
    ax.set_title("Cross-Instance Attention Correlation\n"
                 "Points far from diagonal = imbalanced steps")
    ax.set_aspect("equal")

    if len(x) > 2:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.legend(title=f"Pearson r={corr:.3f}")
    else:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 5: Batch composition diff
# ===================================================================
def plot_batch_composition_diff(fig_dir: Path, steps: list[int],
                                s1: dict[int, StepSummary],
                                s2: dict[int, StepSummary],
                                labels: list[str]) -> Path:
    path = fig_dir / "05_batch_composition_diff.png"
    nreqs1 = [s1[s].num_reqs for s in steps]
    nreqs2 = [s2[s].num_reqs for s in steps]

    slen1 = [s1[s].mean_slen if s1[s].mean_slen is not None else 0 for s in steps]
    slen2 = [s2[s].mean_slen if s2[s].mean_slen is not None else 0 for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Panel 1: num_reqs
    ax1.plot(steps, nreqs1, "-", linewidth=1, color=C_INST1, alpha=0.8, label=labels[0])
    ax1.plot(steps, nreqs2, "-", linewidth=1, color=C_INST2, alpha=0.8, label=labels[1])
    ax1.set_ylabel("Number of Requests")
    ax1.set_title("Batch Composition: Number of Requests per Step")
    ax1.legend()
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Panel 2: mean slen
    ax2.plot(steps, slen1, "-", linewidth=1, color=C_INST1, alpha=0.8, label=labels[0])
    ax2.plot(steps, slen2, "-", linewidth=1, color=C_INST2, alpha=0.8, label=labels[1])
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Sequence Length")
    ax2.set_title("Batch Composition: Mean Sequence Length per Step")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 6: CDF comparison
# ===================================================================
def plot_cdf_comparison(fig_dir: Path,
                        s1: dict[int, StepSummary],
                        s2: dict[int, StepSummary],
                        labels: list[str]) -> Path:
    path = fig_dir / "06_cdf_comparison.png"
    v1 = sorted(s.total_attn_ms for s in s1.values())
    v2 = sorted(s.total_attn_ms for s in s2.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    y1 = np.arange(1, len(v1) + 1) / len(v1)
    y2 = np.arange(1, len(v2) + 1) / len(v2)
    ax.step(v1, y1, where="post", linewidth=1.5, color=C_INST1, label=labels[0])
    ax.step(v2, y2, where="post", linewidth=1.5, color=C_INST2, label=labels[1])

    # Mark medians
    med1 = float(np.median(v1))
    med2 = float(np.median(v2))
    ax.axvline(med1, color=C_INST1, linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(med2, color=C_INST2, linestyle="--", alpha=0.5, linewidth=1)
    ax.text(med1, 0.5, f" med={med1:.2f}", color=C_INST1, fontsize=8, va="center")
    ax.text(med2, 0.55, f" med={med2:.2f}", color=C_INST2, fontsize=8, va="center")

    ax.set_xlabel("Total Attention Time per Step (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution: Per-Step Attention Time")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 7: Per-step delta vs batch diff — correlation
# ===================================================================
def plot_delta_vs_batch_diff(fig_dir: Path, steps: list[int],
                             s1: dict[int, StepSummary],
                             s2: dict[int, StepSummary],
                             labels: list[str]) -> Path:
    """Scatter of attention delta vs num_reqs delta — do different batch sizes
    explain the attention imbalance?"""
    path = fig_dir / "07_delta_vs_batch_diff.png"
    attn_delta = [s1[s].total_attn_ms - s2[s].total_attn_ms for s in steps]
    req_delta = [s1[s].num_reqs - s2[s].num_reqs for s in steps]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(req_delta, attn_delta, c=C_DELTA, alpha=0.3, s=15, edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"num_reqs Delta ({labels[0]} − {labels[1]})")
    ax.set_ylabel(f"Attention Delta ({labels[0]} − {labels[1]}, ms)")
    ax.set_title("Attention Imbalance vs Batch Size Difference\n"
                 "If correlated, batch imbalance explains the bubble")

    if len(attn_delta) > 2:
        corr = float(np.corrcoef(req_delta, attn_delta)[0, 1])
        ax.text(0.02, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path.name}")
    return path


# ===================================================================
# Chart 8: Summary dashboard
# ===================================================================
def plot_summary_dashboard(fig_dir: Path, steps: list[int],
                           s1: dict[int, StepSummary],
                           s2: dict[int, StepSummary],
                           labels: list[str]) -> Path:
    path = fig_dir / "00_summary_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Cross-Instance Decode Attention — MoE All2All Bubble Analysis",
                 fontsize=16, fontweight="bold")

    y1 = np.array([s1[s].total_attn_ms for s in steps])
    y2 = np.array([s2[s].total_attn_ms for s in steps])
    abs_deltas = np.abs(y1 - y2)
    cum_bubble = np.cumsum(abs_deltas)

    # Panel 1: Overlay (compact)
    ax = axes[0, 0]
    ax.plot(steps, y1, "-", linewidth=0.8, color=C_INST1, alpha=0.8, label=labels[0])
    ax.plot(steps, y2, "-", linewidth=0.8, color=C_INST2, alpha=0.8, label=labels[1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Attn Time (ms)")
    ax.set_title("Per-Step Attention Overlay")
    ax.legend(fontsize=8)

    # Panel 2: Scatter
    ax = axes[0, 1]
    ax.scatter(y1, y2, c=C_DELTA, alpha=0.3, s=10, edgecolors="none")
    lo = min(y1.min(), y2.min())
    hi = max(y1.max(), y2.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_xlabel(f"{labels[0]} (ms)")
    ax.set_ylabel(f"{labels[1]} (ms)")
    ax.set_title("Instance Correlation")
    ax.set_aspect("equal")
    if len(y1) > 2:
        corr = float(np.corrcoef(y1, y2)[0, 1])
        ax.text(0.02, 0.95, f"r={corr:.3f}", transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 3: Cumulative bubble
    ax = axes[1, 0]
    ax.fill_between(steps, cum_bubble, alpha=0.3, color=C_FILL)
    ax.plot(steps, cum_bubble, "-", linewidth=1.5, color=C_FILL)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Bubble (ms)")
    ax.set_title("Cumulative All2All Bubble Time")

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")

    total_bubble = float(cum_bubble[-1]) if len(cum_bubble) > 0 else 0
    total_max_attn = float(np.sum(np.maximum(y1, y2)))
    overhead_pct = (total_bubble / total_max_attn * 100) if total_max_attn > 0 else 0

    # Which instance was slower more often
    inst1_slower = int(np.sum(y1 > y2))
    inst2_slower = int(np.sum(y2 > y1))
    equal = len(steps) - inst1_slower - inst2_slower

    summary = (
        f"Common Steps:            {len(steps)}\n"
        f"\n"
        f"─── Bubble Time (All2All Wait) ───\n"
        f"  Total:                 {total_bubble:.2f} ms\n"
        f"  Per-Step Mean:         {float(np.mean(abs_deltas)):.4f} ms\n"
        f"  Per-Step Median:       {float(np.median(abs_deltas)):.4f} ms\n"
        f"  Per-Step P95:          {float(np.percentile(abs_deltas, 95)):.4f} ms\n"
        f"  Per-Step Max:          {float(np.max(abs_deltas)):.4f} ms\n"
        f"  Overhead %%:            {overhead_pct:.2f}%\n"
        f"\n"
        f"─── {labels[0]} Attention ───\n"
        f"  Mean:                  {float(np.mean(y1)):.4f} ms\n"
        f"  Median:                {float(np.median(y1)):.4f} ms\n"
        f"  P95:                   {float(np.percentile(y1, 95)):.4f} ms\n"
        f"\n"
        f"─── {labels[1]} Attention ───\n"
        f"  Mean:                  {float(np.mean(y2)):.4f} ms\n"
        f"  Median:                {float(np.median(y2)):.4f} ms\n"
        f"  P95:                   {float(np.percentile(y2, 95)):.4f} ms\n"
        f"\n"
        f"─── Imbalance Direction ───\n"
        f"  {labels[0]} slower:     {inst1_slower} steps ({inst1_slower/len(steps)*100:.1f}%)\n"
        f"  {labels[1]} slower:     {inst2_slower} steps ({inst2_slower/len(steps)*100:.1f}%)\n"
        f"  Equal:                 {equal} steps\n"
    )
    ax.text(0.03, 0.97, summary, transform=ax.transAxes,
            fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary Statistics")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
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

    # --debug mode: dump schema and exit
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
        print("ERROR: Need exactly 2 .sqlite files (decode instance 1 and 2).",
              file=sys.stderr)
        print("  Pass them as arguments or place decode1.sqlite & decode2.sqlite in:",
              file=sys.stderr)
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
        print(f"ERROR: No decode_attn events in {file1.name}", file=sys.stderr)
        return 1
    if not s2:
        print(f"ERROR: No decode_attn events in {file2.name}", file=sys.stderr)
        return 1

    steps = align_steps(s1, s2)
    if not steps:
        print("ERROR: No common steps between the two instances.", file=sys.stderr)
        return 1

    # CSV
    write_comparison_csv(csv_dir / "cross_instance_comparison.csv", steps, s1, s2, labels)
    print(f"  [csv] cross_instance_comparison.csv")

    # Charts
    generated = []
    generated.append(plot_summary_dashboard(fig_dir, steps, s1, s2, labels))
    generated.append(plot_step_overlay(fig_dir, steps, s1, s2, labels))
    generated.append(plot_step_delta(fig_dir, steps, s1, s2, labels))
    generated.append(plot_cumulative_bubble(fig_dir, steps, s1, s2, labels))
    generated.append(plot_scatter_comparison(fig_dir, steps, s1, s2, labels))
    generated.append(plot_batch_composition_diff(fig_dir, steps, s1, s2, labels))
    generated.append(plot_cdf_comparison(fig_dir, s1, s2, labels))
    generated.append(plot_delta_vs_batch_diff(fig_dir, steps, s1, s2, labels))

    print(f"\nDone! {len(generated)} charts saved to: {fig_dir}")
    print(f"CSV saved to: {csv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
