#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import html
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    # REPO_ROOT / "yqn/disaggregated_serving/logs_2p2d_nixl/decode_attn_nsys",
    REPO_ROOT / "yqn/disaggregated_serving_dp/logs_dp_2p2d_nixl/decode_attn_nsys",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "yqn/figures/decode_attention_nsys"
NVTX_TABLE_CANDIDATES = ("NVTX_EVENTS", "NVTX_PUSHPOP_EVENTS")
LABEL_PATTERN = re.compile(
    r"(step|layer|rank|local_rank|num_reqs|num_tokens|req_key_kind|layer_name|req_keys|qlens|slens)=((?:\[[^\]]*\])|(?:\S+))",
)


@dataclass
class AttnEvent:
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


@dataclass
class RequestLayerStat:
    total_shared_ns: float = 0.0
    samples: int = 0


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

    def add(self, *, start_ns: int, step: int, layer: int, shared_ns: float, qlen: int | None, slen: int | None) -> None:
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
    def avg_shared_attn_per_layer_ms(self) -> float:
        if self.samples == 0:
            return 0.0
        return ns_to_ms(self.total_shared_ns / self.samples)

    @property
    def total_shared_attn_ms(self) -> float:
        return ns_to_ms(self.total_shared_ns)

    @property
    def mean_qlen(self) -> float | None:
        if self.lens_samples == 0:
            return None
        return self.mean_qlen_acc / self.lens_samples

    @property
    def mean_slen(self) -> float | None:
        if self.lens_samples == 0:
            return None
        return self.mean_slen_acc / self.lens_samples


@dataclass
class ReportAnalysis:
    report_path: Path
    sqlite_path: Path
    slug: str
    nvtx_table: str | None = None
    attn_events: list[AttnEvent] = field(default_factory=list)
    request_summaries: list[RequestSummary] = field(default_factory=list)
    request_layer_stats: dict[tuple[str, int], RequestLayerStat] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generated_files: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Nsight .nsys-rep files to SQLite and analyze decode attention "
            "NVTX ranges by request."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Files or directories containing .nsys-rep/.sqlite files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated CSV, SVG, and summary files.",
    )
    parser.add_argument(
        "--nsys-bin",
        default="nsys",
        help="Path to the nsys executable.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export .nsys-rep files even if sibling .sqlite already exists.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Only analyze existing .sqlite files.",
    )
    parser.add_argument(
        "--max-requests-plot",
        type=int,
        default=60,
        help="Maximum number of requests to include in request-level SVG plots.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("first_seen", "avg_attn", "total_attn"),
        default="first_seen",
        help="How to order requests in request-level plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(args.inputs)
    if not inputs:
        print("No .nsys-rep or .sqlite files found.", file=sys.stderr)
        return 1

    analyses: list[ReportAnalysis] = []
    for source_path in inputs:
        sqlite_path = ensure_sqlite(
            source_path=source_path,
            nsys_bin=args.nsys_bin,
            force_export=args.force_export,
            skip_export=args.skip_export,
        )
        analyses.append(
            analyze_report(
                report_path=source_path,
                sqlite_path=sqlite_path,
                output_root=output_root,
                max_requests_plot=args.max_requests_plot,
                sort_by=args.sort_by,
            ),
        )

    write_global_readme(output_root, analyses)
    print(f"Wrote attention analysis outputs to: {output_root}")
    return 0


def discover_inputs(inputs: Iterable[Path]) -> list[Path]:
    chosen: dict[Path, Path] = {}
    for raw_path in inputs:
        path = raw_path.resolve()
        if not path.exists():
            print(f"Skipping missing path: {raw_path}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix in {".sqlite", ".nsys-rep"}:
            chosen[logical_report_id(path)] = choose_preferred_source(
                chosen.get(logical_report_id(path)),
                path,
            )
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file() or child.suffix not in {".sqlite", ".nsys-rep"}:
                    continue
                chosen[logical_report_id(child)] = choose_preferred_source(
                    chosen.get(logical_report_id(child)),
                    child,
                )
    return sorted(chosen.values())


def choose_preferred_source(current: Path | None, candidate: Path) -> Path:
    if current is None:
        return candidate
    if candidate.suffix == ".nsys-rep":
        return candidate
    return current


def logical_report_id(path: Path) -> Path:
    if path.suffix in {".sqlite", ".nsys-rep"}:
        return path.with_suffix("")
    return path


def ensure_sqlite(
    source_path: Path,
    nsys_bin: str,
    force_export: bool,
    skip_export: bool,
) -> Path:
    if source_path.suffix == ".sqlite":
        return source_path
    sqlite_path = source_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force_export:
        return sqlite_path
    if skip_export:
        raise FileNotFoundError(
            f"Missing SQLite export for {source_path} while --skip-export is set.",
        )
    cmd = [
        nsys_bin,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "--quiet",
        "true",
        "--output",
        str(sqlite_path),
        str(source_path),
    ]
    print(f"[export] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return sqlite_path


def analyze_report(
    report_path: Path,
    sqlite_path: Path,
    output_root: Path,
    max_requests_plot: int,
    sort_by: str,
) -> ReportAnalysis:
    slug = slugify(logical_report_id(report_path).relative_to(REPO_ROOT).as_posix())
    report_dir = output_root / slug
    figures_dir = report_dir / "figures"
    csv_dir = report_dir / "csv"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    analysis = ReportAnalysis(
        report_path=report_path,
        sqlite_path=sqlite_path,
        slug=slug,
    )

    conn = sqlite3.connect(sqlite_path)
    try:
        nvtx_table = first_existing_table(conn, NVTX_TABLE_CANDIDATES)
        analysis.nvtx_table = nvtx_table
        if nvtx_table is None:
            analysis.notes.append(
                "This SQLite export has no NVTX table, so decode attention time "
                "cannot be recovered from this report.",
            )
            write_report_summary(report_dir / "summary.md", analysis)
            analysis.generated_files.append(report_dir / "summary.md")
            return analysis

        events = read_decode_attention_events(conn, nvtx_table)
        analysis.attn_events = events
        if not events:
            analysis.notes.append(
                "NVTX table exists, but no 'decode_attn ...' ranges were found. "
                "The current report cannot answer request-level attention timing.",
            )
            write_report_summary(report_dir / "summary.md", analysis)
            analysis.generated_files.append(report_dir / "summary.md")
            return analysis

        request_map, layer_stats = expand_request_stats(events)
        requests = list(request_map.values())
        requests = sort_request_summaries(requests, sort_by)
        analysis.request_summaries = requests
        analysis.request_layer_stats = layer_stats

        write_attention_events_csv(csv_dir / "decode_attention_events.csv", events)
        write_request_summary_csv(csv_dir / "request_attention_summary.csv", requests)
        write_request_layer_csv(csv_dir / "request_layer_attention.csv", layer_stats, requests)
        analysis.generated_files.extend(
            [
                csv_dir / "decode_attention_events.csv",
                csv_dir / "request_attention_summary.csv",
                csv_dir / "request_layer_attention.csv",
            ],
        )

        plot_requests = requests[:max_requests_plot]
        render_vertical_bar_chart(
            path=figures_dir / "request_avg_attn_per_layer.svg",
            title="Average Decode Attention Time Per Layer Invocation",
            subtitle=(
                "x=request, y=average shared decode_attn NVTX duration per "
                "layer invocation (batched events split equally across requests)"
            ),
            items=[(req.request_key, req.avg_shared_attn_per_layer_ms) for req in plot_requests],
            y_label="Average attention time per layer (ms)",
            color="#c84c24",
        )
        render_vertical_bar_chart(
            path=figures_dir / "request_total_attn.svg",
            title="Total Decode Attention Time Attributed Per Request",
            subtitle="Equal split across requests inside each batched decode_attn NVTX event",
            items=[(req.request_key, req.total_shared_attn_ms) for req in plot_requests],
            y_label="Total attributed attention time (ms)",
            color="#1f77b4",
        )
        render_request_layer_heatmap(
            path=figures_dir / "request_layer_avg_attn_heatmap.svg",
            title="Decode Attention Heatmap",
            subtitle="x=request, y=layer, color=average attributed attention time per layer invocation (ms)",
            requests=plot_requests,
            layer_stats=layer_stats,
        )
        analysis.generated_files.extend(
            [
                figures_dir / "request_avg_attn_per_layer.svg",
                figures_dir / "request_total_attn.svg",
                figures_dir / "request_layer_avg_attn_heatmap.svg",
            ],
        )

        write_report_summary(report_dir / "summary.md", analysis)
        analysis.generated_files.append(report_dir / "summary.md")
        return analysis
    finally:
        conn.close()


def first_existing_table(conn: sqlite3.Connection, candidates: Iterable[str]) -> str | None:
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        ).fetchall()
    }
    for table in candidates:
        if table in tables:
            return table
    return None


def get_column_info(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return conn.execute(f"PRAGMA table_info({table})").fetchall()


def choose_name_expression(conn: sqlite3.Connection, table: str, alias: str = "t") -> tuple[str, str]:
    columns = {row[1]: row for row in get_column_info(conn, table)}
    for candidate in ("textId", "nameId", "shortName", "shortNameId"):
        row = columns.get(candidate)
        if row and str(row[2]).upper() == "INTEGER":
            return (
                f"COALESCE(s.value, CAST({alias}.{candidate} AS TEXT))",
                f"LEFT JOIN StringIds s ON {alias}.{candidate} = s.id",
            )
    for candidate in ("text", "name"):
        if candidate in columns:
            return f"CAST({alias}.{candidate} AS TEXT)", ""
    return "'<unnamed>'", ""


def read_decode_attention_events(conn: sqlite3.Connection, nvtx_table: str) -> list[AttnEvent]:
    columns = {row[1] for row in get_column_info(conn, nvtx_table)}
    if "start" not in columns or "end" not in columns:
        return []
    label_expr, join_clause = choose_name_expression(conn, nvtx_table)
    sql = f"""
        SELECT label, start, end
        FROM (
            SELECT
                {label_expr} AS label,
                t.start AS start,
                t.end AS end
            FROM {nvtx_table} t
            {join_clause}
        )
        WHERE label LIKE 'decode_attn %'
        ORDER BY start
    """
    rows = conn.execute(sql).fetchall()
    events: list[AttnEvent] = []
    for label, start_ns, end_ns in rows:
        event = parse_decode_attn_label(
            label=str(label),
            start_ns=int(start_ns),
            end_ns=int(end_ns),
        )
        if event is not None:
            events.append(event)
    return events


def parse_decode_attn_label(label: str, start_ns: int, end_ns: int) -> AttnEvent | None:
    if not label.startswith("decode_attn "):
        return None
    fields = {key: value for key, value in LABEL_PATTERN.findall(label)}
    req_keys = parse_list_str(fields.get("req_keys", "[]"))
    qlens = parse_list_int(fields.get("qlens", "[]"))
    slens = parse_list_int(fields.get("slens", "[]"))
    num_reqs = parse_int(fields.get("num_reqs"), default=len(req_keys))
    return AttnEvent(
        label=label,
        start_ns=start_ns,
        end_ns=end_ns,
        step=parse_int(fields.get("step"), default=-1),
        layer=parse_int(fields.get("layer"), default=-1),
        layer_name=fields.get("layer_name", "unknown"),
        rank=fields.get("rank", "na"),
        local_rank=fields.get("local_rank", "na"),
        num_reqs=num_reqs,
        num_tokens=parse_int(fields.get("num_tokens"), default=0),
        req_key_kind=fields.get("req_key_kind", "unknown"),
        req_keys=req_keys,
        qlens=qlens,
        slens=slens,
    )


def parse_list_str(value: str) -> list[str]:
    value = value.strip()
    if len(value) < 2 or not value.startswith("[") or not value.endswith("]"):
        return []
    body = value[1:-1]
    if not body:
        return []
    return [item for item in body.split(",") if item]


def parse_list_int(value: str) -> list[int]:
    results: list[int] = []
    for item in parse_list_str(value):
        try:
            results.append(int(item))
        except ValueError:
            continue
    return results


def parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def expand_request_stats(
    events: list[AttnEvent],
) -> tuple[dict[str, RequestSummary], dict[tuple[str, int], RequestLayerStat]]:
    requests: dict[str, RequestSummary] = {}
    layer_stats: dict[tuple[str, int], RequestLayerStat] = defaultdict(RequestLayerStat)

    for event in events:
        request_count = max(1, event.num_reqs, len(event.req_keys))
        share_ns = event.duration_ns / request_count
        request_keys = event.req_keys or [f"slot_{idx}" for idx in range(request_count)]

        for idx in range(request_count):
            request_key = request_keys[idx] if idx < len(request_keys) else f"slot_{idx}"
            qlen = event.qlens[idx] if idx < len(event.qlens) else None
            slen = event.slens[idx] if idx < len(event.slens) else None
            if request_key not in requests:
                requests[request_key] = RequestSummary(
                    request_key=request_key,
                    first_start_ns=event.start_ns,
                )
            requests[request_key].add(
                start_ns=event.start_ns,
                step=event.step,
                layer=event.layer,
                shared_ns=share_ns,
                qlen=qlen,
                slen=slen,
            )
            layer_stats[(request_key, event.layer)].total_shared_ns += share_ns
            layer_stats[(request_key, event.layer)].samples += 1

    return requests, layer_stats


def sort_request_summaries(requests: list[RequestSummary], sort_by: str) -> list[RequestSummary]:
    if sort_by == "avg_attn":
        return sorted(
            requests,
            key=lambda item: (-item.avg_shared_attn_per_layer_ms, item.first_start_ns, item.request_key),
        )
    if sort_by == "total_attn":
        return sorted(
            requests,
            key=lambda item: (-item.total_shared_attn_ms, item.first_start_ns, item.request_key),
        )
    return sorted(requests, key=lambda item: (item.first_start_ns, item.request_key))


def write_attention_events_csv(path: Path, events: list[AttnEvent]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_ns",
                "end_ns",
                "duration_ns",
                "duration_ms",
                "step",
                "layer",
                "layer_name",
                "num_reqs",
                "num_tokens",
                "req_key_kind",
                "req_keys",
                "qlens",
                "slens",
                "label",
            ],
        )
        for event in events:
            writer.writerow(
                [
                    event.start_ns,
                    event.end_ns,
                    event.duration_ns,
                    f"{ns_to_ms(event.duration_ns):.6f}",
                    event.step,
                    event.layer,
                    event.layer_name,
                    event.num_reqs,
                    event.num_tokens,
                    event.req_key_kind,
                    ",".join(event.req_keys),
                    ",".join(str(item) for item in event.qlens),
                    ",".join(str(item) for item in event.slens),
                    event.label,
                ],
            )


def write_request_summary_csv(path: Path, requests: list[RequestSummary]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_key",
                "first_start_ns",
                "total_shared_attn_ns",
                "total_shared_attn_ms",
                "samples",
                "unique_steps",
                "unique_layers",
                "avg_shared_attn_per_layer_ms",
                "mean_qlen",
                "mean_slen",
            ],
        )
        for req in requests:
            writer.writerow(
                [
                    req.request_key,
                    req.first_start_ns,
                    f"{req.total_shared_ns:.6f}",
                    f"{req.total_shared_attn_ms:.6f}",
                    req.samples,
                    len(req.unique_steps),
                    len(req.unique_layers),
                    f"{req.avg_shared_attn_per_layer_ms:.6f}",
                    "" if req.mean_qlen is None else f"{req.mean_qlen:.6f}",
                    "" if req.mean_slen is None else f"{req.mean_slen:.6f}",
                ],
            )


def write_request_layer_csv(
    path: Path,
    layer_stats: dict[tuple[str, int], RequestLayerStat],
    requests: list[RequestSummary],
) -> None:
    request_index = {req.request_key: req for req in requests}
    rows = []
    for (request_key, layer), stat in layer_stats.items():
        avg_ms = ns_to_ms(stat.total_shared_ns / stat.samples) if stat.samples else 0.0
        rows.append(
            (
                request_index[request_key].first_start_ns,
                request_key,
                layer,
                stat.samples,
                ns_to_ms(stat.total_shared_ns),
                avg_ms,
            ),
        )
    rows.sort()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_key",
                "layer",
                "samples",
                "total_shared_attn_ms",
                "avg_shared_attn_per_layer_ms",
            ],
        )
        for _, request_key, layer, samples, total_ms, avg_ms in rows:
            writer.writerow([request_key, layer, samples, f"{total_ms:.6f}", f"{avg_ms:.6f}"])


def render_vertical_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    items: list[tuple[str, float]],
    y_label: str,
    color: str,
) -> None:
    width = max(1200, 140 + len(items) * 24)
    height = 700
    left_margin = 100
    right_margin = 50
    top_margin = 110
    bottom_margin = 210
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    max_value = max((value for _, value in items), default=1.0)
    bar_width = chart_width / max(1, len(items)) * 0.75

    lines = [
        svg_header(width, height),
        svg_text(40, 40, title, size=26, weight="700"),
        svg_text(40, 72, subtitle, size=14, fill="#666666"),
        svg_text(24, top_margin + chart_height / 2, y_label, size=13, fill="#444444", rotate=-90),
    ]

    tick_count = 5
    for idx in range(tick_count + 1):
        y = top_margin + chart_height - chart_height * idx / tick_count
        value = max_value * idx / tick_count
        lines.append(
            f'<line x1="{left_margin}" y1="{y:.2f}" x2="{width - right_margin}" y2="{y:.2f}" '
            'stroke="#e6e6e6" stroke-width="1" />',
        )
        lines.append(
            svg_text(left_margin - 10, y + 4, format_value(value), size=12, anchor="end", fill="#666666"),
        )

    for idx, (label, value) in enumerate(items):
        slot_left = left_margin + chart_width * idx / max(1, len(items))
        x = slot_left + (chart_width / max(1, len(items)) - bar_width) / 2
        bar_height = 0 if max_value == 0 else chart_height * value / max_value
        y = top_margin + chart_height - bar_height
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'rx="3" fill="{color}" opacity="0.85" />',
        )
        label_x = x + bar_width / 2
        label_y = top_margin + chart_height + 18
        lines.append(
            svg_text(label_x, label_y, trim_label(label, 22), size=11, anchor="end", fill="#333333", rotate=60),
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def render_request_layer_heatmap(
    path: Path,
    title: str,
    subtitle: str,
    requests: list[RequestSummary],
    layer_stats: dict[tuple[str, int], RequestLayerStat],
) -> None:
    layers = sorted({layer for _, layer in layer_stats})
    width = max(1200, 300 + len(requests) * 20)
    height = max(700, 220 + len(layers) * 22)
    left_margin = 120
    right_margin = 40
    top_margin = 120
    bottom_margin = 220
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    cell_width = chart_width / max(1, len(requests))
    cell_height = chart_height / max(1, len(layers))
    max_ms = 0.0
    avg_ms_map: dict[tuple[str, int], float] = {}
    for key, stat in layer_stats.items():
        avg_ms = ns_to_ms(stat.total_shared_ns / stat.samples) if stat.samples else 0.0
        avg_ms_map[key] = avg_ms
        max_ms = max(max_ms, avg_ms)

    lines = [
        svg_header(width, height),
        svg_text(40, 40, title, size=26, weight="700"),
        svg_text(40, 72, subtitle, size=14, fill="#666666"),
        svg_text(width - 160, 96, f"max={format_value(max_ms)} ms", size=12, fill="#444444"),
    ]

    for layer_idx, layer in enumerate(layers):
        y = top_margin + layer_idx * cell_height
        lines.append(svg_text(left_margin - 10, y + cell_height * 0.68, f"L{layer}", size=12, anchor="end"))

    for req_idx, req in enumerate(requests):
        x = left_margin + req_idx * cell_width
        label_x = x + cell_width * 0.5
        label_y = top_margin + chart_height + 18
        lines.append(
            svg_text(label_x, label_y, trim_label(req.request_key, 18), size=11, anchor="end", fill="#333333", rotate=60),
        )
        for layer_idx, layer in enumerate(layers):
            y = top_margin + layer_idx * cell_height
            avg_ms = avg_ms_map.get((req.request_key, layer), 0.0)
            fill = heat_color(avg_ms, max_ms)
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="1" />',
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def heat_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        return "#f3f3f3"
    ratio = max(0.0, min(1.0, value / max_value))
    r = int(245 - 80 * ratio)
    g = int(245 - 170 * ratio)
    b = int(245 - 210 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_report_summary(path: Path, analysis: ReportAnalysis) -> None:
    lines = [
        f"# Decode Attention Summary: `{analysis.report_path.relative_to(REPO_ROOT).as_posix()}`",
        "",
        f"- SQLite: `{analysis.sqlite_path.relative_to(REPO_ROOT).as_posix()}`",
        f"- NVTX table: `{analysis.nvtx_table or 'missing'}`",
        f"- decode_attn events found: `{len(analysis.attn_events)}`",
        "",
        "## Interpretation",
        "",
        "- Each `decode_attn ...` NVTX range is a batched decode attention region for one layer.",
        "- This script attributes each batched range equally across all requests listed in `req_keys`.",
        "- `request_avg_attn_per_layer.svg` uses x=request and y=average attributed attention time per layer invocation.",
        "",
    ]
    if analysis.request_summaries:
        lines.extend(
            [
                "## Request Summary",
                "",
            ],
        )
        for req in analysis.request_summaries[:20]:
            lines.append(
                f"- `{req.request_key}`: avg/layer `{req.avg_shared_attn_per_layer_ms:.6f} ms`, "
                f"total `{req.total_shared_attn_ms:.6f} ms`, samples `{req.samples}`, "
                f"steps `{len(req.unique_steps)}`, layers `{len(req.unique_layers)}`",
            )
        lines.append("")
    if analysis.notes:
        lines.extend(["## Notes", ""])
        for note in analysis.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend(["## Generated Files", ""])
    for file_path in sorted(analysis.generated_files):
        lines.append(f"- `{file_path.relative_to(REPO_ROOT).as_posix()}`")
    path.write_text("\n".join(lines) + "\n")


def write_global_readme(output_root: Path, analyses: list[ReportAnalysis]) -> None:
    readme_path = output_root / "README.md"
    lines = [
        "# Decode Attention Nsight Analysis",
        "",
        "- This tool only analyzes NVTX ranges whose label starts with `decode_attn`.",
        "- If a report has no NVTX table or no `decode_attn` ranges, request-level attention timing cannot be reconstructed from that report.",
        "",
        "## Reports",
        "",
    ]
    for analysis in analyses:
        rel_report = analysis.report_path.relative_to(REPO_ROOT).as_posix()
        rel_summary = (output_root / analysis.slug / "summary.md").relative_to(REPO_ROOT).as_posix()
        lines.append(f"- `{rel_report}` -> `{rel_summary}`")
    readme_path.write_text("\n".join(lines) + "\n")


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", text).strip("_")


def ns_to_ms(value: float) -> float:
    return value / 1_000_000.0


def trim_label(label: str, limit: int) -> str:
    if len(label) <= limit:
        return label
    return label[: limit - 3] + "..."


def format_value(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:,.1f}"
    if value >= 1:
        return f"{value:,.2f}"
    return f"{value:.3f}"


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white" />'
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#111111",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}"{transform}>{html.escape(text)}</text>'
    )


if __name__ == "__main__":
    raise SystemExit(main())
