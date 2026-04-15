# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Extract per-batch and per-attention-layer timings from nsys SQLite export.

Expected workflow:
1) Collect profile with NVTX enabled, e.g. nsys profile --trace=cuda,nvtx ...
2) Export .nsys-rep to SQLite, e.g. nsys export --type sqlite --output run.sqlite run.nsys-rep
3) Run this script on run.sqlite.

This script parses NVTX ranges and outputs:
- <stem>.batch_summary.csv
- <stem>.batch_layer_attention.csv
- <stem>.attention_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BATCH_NAME_RE = re.compile(
    r"^execute_context_(?P<ctx_reqs>\d+)\((?P<ctx_tokens>\d+)\)"
    r"_generation_(?P<gen_reqs>\d+)\((?P<gen_tokens>\d+)\)$"
)
MODULE_NAME_RE = re.compile(r"""['"]Module['"]\s*:\s*['"]([^'"]+)['"]""")


@dataclass
class NvtxRange:
    name: str
    start: int
    end: int
    tid: int | None

    @property
    def dur(self) -> int:
        return self.end - self.start


@dataclass
class BatchMarker:
    batch_idx: int
    start: int
    end: int
    tid: int | None
    ctx_reqs: int | None
    ctx_tokens: int | None
    gen_reqs: int | None
    gen_tokens: int | None


@dataclass
class AttnEvent:
    start: int
    dur: int
    tid: int | None
    layer_name: str


@dataclass
class ForwardEvent:
    start: int
    dur: int
    tid: int | None


def _sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    # cid, name, type, notnull, dflt_value, pk
    return [r[1] for r in cur.fetchall()]


def _find_col(cols: list[str], aliases: list[str]) -> str | None:
    cols_map = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in cols_map:
            return cols_map[a.lower()]
    return None


def _pick_nvtx_table(conn: sqlite3.Connection) -> tuple[str, dict[str, str | None]]:
    best: tuple[int, str, dict[str, str | None]] | None = None
    for table in _sqlite_tables(conn):
        if "sqlite_" in table:
            continue
        cols = _table_columns(conn, table)
        if not cols:
            continue

        start_col = _find_col(cols, ["start", "start_ns", "ts"])
        end_col = _find_col(cols, ["end", "end_ns", "stop"])
        if not start_col or not end_col:
            continue

        text_col = _find_col(cols, ["text", "name", "message"])
        textid_col = _find_col(cols, ["textid", "text_id", "stringid", "nameid"])
        tid_col = _find_col(cols, ["globalTid", "global_tid", "tid", "threadid"])

        score = 0
        if "nvtx" in table.lower():
            score += 4
        if text_col:
            score += 2
        if textid_col:
            score += 1
        if tid_col:
            score += 1

        payload = {
            "start": start_col,
            "end": end_col,
            "text": text_col,
            "textid": textid_col,
            "tid": tid_col,
        }
        if best is None or score > best[0]:
            best = (score, table, payload)

    if best is None:
        raise RuntimeError("Could not find NVTX-like table with start/end columns.")
    return best[1], best[2]


def _pick_string_table(conn: sqlite3.Connection) -> tuple[str, str, str] | None:
    for table in _sqlite_tables(conn):
        cols = _table_columns(conn, table)
        if not cols:
            continue
        id_col = _find_col(cols, ["id", "stringid", "sid"])
        val_col = _find_col(cols, ["value", "string", "text", "name"])
        if id_col and val_col and ("string" in table.lower() or "str" in table.lower()):
            return table, id_col, val_col
    return None


def _build_string_map(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    val_col: str,
    ids: set[int],
) -> dict[int, str]:
    if not ids:
        return {}
    out: dict[int, str] = {}
    ids_list = sorted(ids)
    # SQLite has variable binding limits; chunk conservatively.
    chunk_size = 500
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        q = f"SELECT {id_col}, {val_col} FROM {table} WHERE {id_col} IN ({placeholders})"
        for rid, rval in conn.execute(q, chunk):
            if isinstance(rid, int) and rval is not None:
                out[rid] = str(rval)
    return out


def _extract_nvtx_ranges(conn: sqlite3.Connection) -> list[NvtxRange]:
    table, cols = _pick_nvtx_table(conn)

    start_col = cols["start"]
    end_col = cols["end"]
    text_col = cols["text"]
    textid_col = cols["textid"]
    tid_col = cols["tid"]

    select_cols = [start_col, end_col]
    if text_col:
        select_cols.append(text_col)
    elif textid_col:
        select_cols.append(textid_col)
    else:
        raise RuntimeError(
            f"NVTX table '{table}' has no text/textid column to identify ranges."
        )

    if tid_col:
        select_cols.append(tid_col)

    q = f"SELECT {', '.join(select_cols)} FROM {table} WHERE {end_col} > {start_col}"
    rows = conn.execute(q).fetchall()

    # Resolve textid -> text if needed
    text_map: dict[int, str] = {}
    if not text_col and textid_col:
        ids: set[int] = set()
        for row in rows:
            textid = row[2]
            if isinstance(textid, int):
                ids.add(textid)
        st = _pick_string_table(conn)
        if st is not None:
            text_map = _build_string_map(conn, st[0], st[1], st[2], ids)

    out: list[NvtxRange] = []
    for row in rows:
        start = int(row[0])
        end = int(row[1])
        raw_text = row[2]
        tid = int(row[3]) if tid_col and row[3] is not None else None

        if isinstance(raw_text, str):
            name = raw_text
        elif isinstance(raw_text, (int,)):
            name = text_map.get(raw_text, str(raw_text))
        else:
            name = str(raw_text)

        out.append(NvtxRange(name=name, start=start, end=end, tid=tid))

    out.sort(key=lambda r: r.start)
    return out


def _parse_batch_marker(name: str) -> tuple[int, int, int, int] | None:
    m = BATCH_NAME_RE.match(name)
    if m is None:
        return None
    return (
        int(m.group("ctx_reqs")),
        int(m.group("ctx_tokens")),
        int(m.group("gen_reqs")),
        int(m.group("gen_tokens")),
    )


def _extract_module_name(event_name: str) -> str | None:
    m = MODULE_NAME_RE.search(event_name)
    if m:
        return m.group(1)
    return None


def _find_batches(ranges: list[NvtxRange]) -> list[BatchMarker]:
    batches: list[BatchMarker] = []
    for r in ranges:
        parsed = _parse_batch_marker(r.name)
        if parsed is None:
            continue
        ctx_reqs, ctx_tokens, gen_reqs, gen_tokens = parsed
        batches.append(
            BatchMarker(
                batch_idx=len(batches),
                start=r.start,
                end=r.end,
                tid=r.tid,
                ctx_reqs=ctx_reqs,
                ctx_tokens=ctx_tokens,
                gen_reqs=gen_reqs,
                gen_tokens=gen_tokens,
            )
        )

    if batches:
        return batches

    # Fallback: each forward range as one batch.
    for r in ranges:
        if r.name == "gpu_model_runner: forward":
            batches.append(
                BatchMarker(
                    batch_idx=len(batches),
                    start=r.start,
                    end=r.end,
                    tid=r.tid,
                    ctx_reqs=None,
                    ctx_tokens=None,
                    gen_reqs=None,
                    gen_tokens=None,
                )
            )
    return batches


def _collect_events(
    ranges: list[NvtxRange],
    attention_re: re.Pattern[str],
) -> tuple[list[AttnEvent], list[ForwardEvent]]:
    attn_events: list[AttnEvent] = []
    fwd_events: list[ForwardEvent] = []
    for r in ranges:
        if r.name == "gpu_model_runner: forward":
            fwd_events.append(ForwardEvent(start=r.start, dur=r.dur, tid=r.tid))

        if _parse_batch_marker(r.name) is not None:
            continue

        module_name = _extract_module_name(r.name)
        candidate = module_name or r.name
        if attention_re.search(candidate) is None:
            continue

        attn_events.append(
            AttnEvent(start=r.start, dur=r.dur, tid=r.tid, layer_name=candidate)
        )

    attn_events.sort(key=lambda e: e.start)
    fwd_events.sort(key=lambda e: e.start)
    return attn_events, fwd_events


def _aggregate(
    batches: list[BatchMarker],
    attn_events: list[AttnEvent],
    fwd_events: list[ForwardEvent],
    ns_to_ms: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attn_by_tid: dict[int | None, list[AttnEvent]] = defaultdict(list)
    fwd_by_tid: dict[int | None, list[ForwardEvent]] = defaultdict(list)

    for e in attn_events:
        attn_by_tid[e.tid].append(e)
    for e in fwd_events:
        fwd_by_tid[e.tid].append(e)

    attn_starts_by_tid = {
        tid: [e.start for e in events] for tid, events in attn_by_tid.items()
    }
    fwd_starts_by_tid = {
        tid: [e.start for e in events] for tid, events in fwd_by_tid.items()
    }

    batch_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for b in batches:
        tid = b.tid
        attn = attn_by_tid.get(tid, [])
        attn_starts = attn_starts_by_tid.get(tid, [])
        fwds = fwd_by_tid.get(tid, [])
        fwd_starts = fwd_starts_by_tid.get(tid, [])

        layer_dur_ns: dict[str, int] = defaultdict(int)
        layer_calls: dict[str, int] = defaultdict(int)

        i = bisect_left(attn_starts, b.start)
        while i < len(attn):
            e = attn[i]
            if e.start >= b.end:
                break
            layer_dur_ns[e.layer_name] += e.dur
            layer_calls[e.layer_name] += 1
            i += 1

        fwd_dur_ns = 0
        j = bisect_left(fwd_starts, b.start)
        while j < len(fwds):
            e = fwds[j]
            if e.start >= b.end:
                break
            fwd_dur_ns += e.dur
            j += 1

        total_attn_ns = sum(layer_dur_ns.values())
        batch_rows.append(
            {
                "batch_idx": b.batch_idx,
                "tid": b.tid,
                "ctx_reqs": b.ctx_reqs,
                "ctx_tokens": b.ctx_tokens,
                "gen_reqs": b.gen_reqs,
                "gen_tokens": b.gen_tokens,
                "batch_dur_ms": b.end - b.start,
                "forward_dur_ms": fwd_dur_ns,
                "attention_total_ms": total_attn_ns,
                "attention_event_calls": int(sum(layer_calls.values())),
                "attention_unique_layers": int(len(layer_calls)),
            }
        )
        batch_rows[-1]["batch_dur_ms"] *= ns_to_ms
        batch_rows[-1]["forward_dur_ms"] *= ns_to_ms
        batch_rows[-1]["attention_total_ms"] *= ns_to_ms

        for layer_name, dur_ns in sorted(
            layer_dur_ns.items(), key=lambda x: x[1], reverse=True
        ):
            calls = layer_calls[layer_name]
            total_ms = dur_ns * ns_to_ms
            avg_ms = (dur_ns / calls) * ns_to_ms if calls > 0 else 0.0
            layer_rows.append(
                {
                    "batch_idx": b.batch_idx,
                    "tid": b.tid,
                    "layer_name": layer_name,
                    "calls": calls,
                    "total_ms": total_ms,
                    "avg_ms": avg_ms,
                }
            )

    return batch_rows, layer_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stem(path: Path) -> str:
    name = path.name
    if name.endswith(".sqlite"):
        return name[: -len(".sqlite")]
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract per-batch and per-attention-layer timings from nsys SQLite NVTX ranges."
        )
    )
    p.add_argument(
        "--sqlite",
        type=Path,
        required=True,
        help="Path to SQLite exported from nsys report.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nsys_attention_reports"),
        help="Directory for output CSV/JSON files.",
    )
    p.add_argument(
        "--attention-pattern",
        type=str,
        default=r"(attn|attention)",
        help="Case-insensitive regex to match attention layer names.",
    )
    p.add_argument(
        "--time-unit",
        choices=["ns", "us"],
        default="ns",
        help="Timestamp unit in SQLite start/end columns. nsys export typically uses ns.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    sqlite_path: Path = args.sqlite
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ns_to_ms = 1e-6 if args.time_unit == "ns" else 1e-3
    attention_re = re.compile(args.attention_pattern, re.IGNORECASE)

    conn = sqlite3.connect(str(sqlite_path))
    try:
        ranges = _extract_nvtx_ranges(conn)
    finally:
        conn.close()

    batches = _find_batches(ranges)
    attn_events, fwd_events = _collect_events(ranges, attention_re)
    batch_rows, layer_rows = _aggregate(batches, attn_events, fwd_events, ns_to_ms)

    stem = _stem(sqlite_path)
    batch_csv = args.output_dir / f"{stem}.batch_summary.csv"
    layer_csv = args.output_dir / f"{stem}.batch_layer_attention.csv"
    report_json = args.output_dir / f"{stem}.attention_report.json"

    _write_csv(
        batch_csv,
        batch_rows,
        fields=[
            "batch_idx",
            "tid",
            "ctx_reqs",
            "ctx_tokens",
            "gen_reqs",
            "gen_tokens",
            "batch_dur_ms",
            "forward_dur_ms",
            "attention_total_ms",
            "attention_event_calls",
            "attention_unique_layers",
        ],
    )
    _write_csv(
        layer_csv,
        layer_rows,
        fields=[
            "batch_idx",
            "tid",
            "layer_name",
            "calls",
            "total_ms",
            "avg_ms",
        ],
    )

    report = {
        "sqlite": str(sqlite_path),
        "num_nvtx_ranges": len(ranges),
        "num_batches": len(batch_rows),
        "num_attention_events": len(attn_events),
        "outputs": {
            "batch_summary_csv": str(batch_csv),
            "batch_layer_attention_csv": str(layer_csv),
            "report_json": str(report_json),
        },
        "notes": [
            "Batch markers parsed from execute_context_x(...)_generation_y(...) NVTX ranges.",
            "If batch markers are absent, gpu_model_runner: forward ranges are used as fallback batches.",
            "Attention layers are matched using --attention-pattern on module/event names.",
        ],
    }
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[DONE] batches={report['num_batches']} attention_events={report['num_attention_events']}\n"
        f"  - {batch_csv}\n"
        f"  - {layer_csv}\n"
        f"  - {report_json}"
    )


if __name__ == "__main__":
    main()
