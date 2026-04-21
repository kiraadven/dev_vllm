#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Replay one trace CSV online in async mode against disaggregated P/D proxies.

Input CSV columns:
- required: input_tokens/output_tokens (or *_length variants)
- optional: arrive_time

Output files under --output-dir:
- dispatch_requests.csv
- request_assignments.jsonl (or user-specified --request-assignment-log-path)
- online_results.csv
- summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass(frozen=True)
class TraceRequest:
    trace_name: str
    request_id: int
    arrive_time: float
    input_tokens: int
    output_tokens: int
    in_str: str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online async trace replay")
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument(
        "--proxy-urls",
        type=str,
        required=True,
        help="comma-separated, e.g. http://127.0.0.1:8000,http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--prefill-ranks",
        type=str,
        default="0,1",
        help="comma-separated prefill ranks/gpu ids matched to --proxy-urls order",
    )
    parser.add_argument(
        "--request-assignment-log-path",
        type=str,
        default="",
        help=(
            "Optional JSONL file path. One line per request assignment: "
            "request_id -> prefill_rank/decode_tokens."
        ),
    )
    parser.add_argument("--model", type=str, default="/data/yqn/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def parse_int_csv(raw: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected non-empty integer CSV list")
    return [int(x) for x in vals]


def _get_int_field(row: dict[str, str], candidates: list[str]) -> int:
    for key in candidates:
        if key in row and row[key] != "":
            return int(row[key])
    raise KeyError(f"missing any of fields: {candidates}")


def load_trace_requests(trace_path: Path) -> list[TraceRequest]:
    requests: list[TraceRequest] = []
    with open(trace_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for request_id, row in enumerate(reader):
            in_tok = _get_int_field(row, ["input_tokens_length", "input_tokens"])
            out_tok = _get_int_field(row, ["output_tokens_length", "output_tokens"])
            arrive_time = float(row.get("arrive_time", "0") or 0.0)
            requests.append(
                TraceRequest(
                    trace_name=trace_path.stem,
                    request_id=request_id,
                    arrive_time=arrive_time,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    in_str="hi" * max(1, in_tok),
                )
            )
    if not requests:
        raise ValueError(f"Trace is empty: {trace_path}")
    return requests


def dispatch_round_robin(
    requests: list[TraceRequest], prefill_ranks: list[int]
) -> dict[int, list[TraceRequest]]:
    if not prefill_ranks:
        raise ValueError("prefill_ranks is empty")
    assigned: dict[int, list[TraceRequest]] = {rank: [] for rank in prefill_ranks}
    for idx, req in enumerate(requests):
        rank = prefill_ranks[idx % len(prefill_ranks)]
        assigned[rank].append(req)
    return assigned


def write_dispatch_csv(
    output_dir: Path,
    trace_name: str,
    assignments: dict[int, list[TraceRequest]],
) -> None:
    out_csv = output_dir / "dispatch_requests.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_name",
                "prefill_rank",
                "request_id",
                "input_tokens",
                "output_tokens",
                "arrive_time",
            ],
        )
        writer.writeheader()
        for prefill_rank, reqs in assignments.items():
            for req in reqs:
                writer.writerow(
                    {
                        "trace_name": trace_name,
                        "prefill_rank": prefill_rank,
                        "request_id": req.request_id,
                        "input_tokens": req.input_tokens,
                        "output_tokens": req.output_tokens,
                        "arrive_time": f"{req.arrive_time:.6f}",
                    }
                )


def write_request_assignment_jsonl(
    request_assignment_log_path: Path,
    assignments: dict[int, list[TraceRequest]],
) -> None:
    rows: list[dict[str, int | str]] = []
    for prefill_rank, reqs in assignments.items():
        for req in reqs:
            rows.append(
                {
                    "request_id": str(req.request_id),
                    "prefill_rank": int(prefill_rank),
                    "decode_tokens": int(req.output_tokens),
                    "output_tokens": int(req.output_tokens),
                }
            )

    rows.sort(key=lambda item: int(str(item["request_id"])))
    with open(request_assignment_log_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    vals = sorted(values)
    idx = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    ratio = idx - lo
    return vals[lo] * (1.0 - ratio) + vals[hi] * ratio


async def _submit_one(
    session: aiohttp.ClientSession,
    proxy_url: str,
    req: TraceRequest,
    model: str,
    base_arrive_time: float,
    run_start_perf: float,
) -> dict[str, Any]:
    request_url = f"{proxy_url.rstrip('/')}/v1/completions"

    scheduled_offset = req.arrive_time - base_arrive_time
    scheduled_at = run_start_perf + max(0.0, scheduled_offset)
    now = time.perf_counter()
    if scheduled_at > now:
        await asyncio.sleep(scheduled_at - now)

    submit_perf = time.perf_counter()
    payload = {
        "model": model,
        "prompt": req.in_str,
        "max_tokens": req.output_tokens,
        "temperature": 0,
    }
    headers = {"X-Trace-Request-Id": str(req.request_id)}

    status_code = -1
    error = ""
    body_bytes = 0
    ok = False
    try:
        async with session.post(request_url, json=payload, headers=headers) as resp:
            body = await resp.read()
            body_bytes = len(body)
            status_code = resp.status
            ok = resp.status == 200
            if not ok:
                error = body.decode("utf-8", errors="ignore")[:2000]
    except Exception as exc:
        error = str(exc)

    finish_perf = time.perf_counter()
    return {
        "request_id": req.request_id,
        "arrive_time": req.arrive_time,
        "scheduled_offset_s": scheduled_offset,
        "submit_offset_s": submit_perf - run_start_perf,
        "finish_offset_s": finish_perf - run_start_perf,
        "latency_s": finish_perf - submit_perf,
        "status_code": status_code,
        "ok": ok,
        "error": error,
        "response_bytes": body_bytes,
        "input_tokens": req.input_tokens,
        "output_tokens": req.output_tokens,
    }


async def replay_trace(
    requests: list[TraceRequest],
    assignments: dict[int, list[TraceRequest]],
    rank_to_proxy: dict[int, str],
    model: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    run_start_perf = time.perf_counter()
    base_arrive_time = requests[0].arrive_time

    req_to_rank: dict[int, int] = {}
    for rank, reqs in assignments.items():
        for req in reqs:
            req_to_rank[req.request_id] = rank

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for req in requests:
            prefill_rank = req_to_rank[req.request_id]
            proxy_url = rank_to_proxy[prefill_rank]
            task = asyncio.create_task(
                _submit_one(
                    session=session,
                    proxy_url=proxy_url,
                    req=req,
                    model=model,
                    base_arrive_time=base_arrive_time,
                    run_start_perf=run_start_perf,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    for row in results:
        row["prefill_rank"] = req_to_rank[int(row["request_id"])]

    return sorted(results, key=lambda x: int(x["request_id"]))


def write_results_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_csv = output_dir / "online_results.csv"
    fieldnames = [
        "request_id",
        "prefill_rank",
        "arrive_time",
        "scheduled_offset_s",
        "submit_offset_s",
        "finish_offset_s",
        "latency_s",
        "status_code",
        "ok",
        "error",
        "response_bytes",
        "input_tokens",
        "output_tokens",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(
    output_dir: Path,
    trace_name: str,
    trace_requests: list[TraceRequest],
    rows: list[dict[str, Any]],
    rank_to_proxy: dict[int, str],
) -> None:
    ok_rows = [r for r in rows if bool(r["ok"])]
    latencies = [float(r["latency_s"]) for r in ok_rows]

    summary = {
        "trace_name": trace_name,
        "dispatch_mode": "round_robin_prefill_ranks",
        "num_requests": len(rows),
        "num_success": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "rank_to_proxy": rank_to_proxy,
        "trace_start": float(trace_requests[0].arrive_time),
        "trace_end": float(trace_requests[-1].arrive_time),
        "trace_duration_s": float(
            trace_requests[-1].arrive_time - trace_requests[0].arrive_time
        ),
        "latency_s": {
            "p50": _pctl(latencies, 50),
            "p95": _pctl(latencies, 95),
            "p99": _pctl(latencies, 99),
            "mean": (sum(latencies) / len(latencies)) if latencies else math.nan,
        },
    }

    out_json = output_dir / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    args = create_parser().parse_args()

    trace_path = Path(args.trace_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_urls = [u.strip() for u in args.proxy_urls.split(",") if u.strip()]
    prefill_ranks = parse_int_csv(args.prefill_ranks)
    if not proxy_urls:
        raise ValueError("--proxy-urls is empty")
    if not prefill_ranks:
        raise ValueError("--prefill-ranks is empty")
    if len(proxy_urls) != len(prefill_ranks):
        raise ValueError(
            "proxy_urls and prefill_ranks must have same length. "
            f"Got proxy_urls={len(proxy_urls)} prefill_ranks={len(prefill_ranks)}"
        )

    rank_to_proxy = {rank: proxy_urls[idx] for idx, rank in enumerate(prefill_ranks)}

    requests = load_trace_requests(trace_path)
    assignments = dispatch_round_robin(requests, prefill_ranks)
    write_dispatch_csv(output_dir, requests[0].trace_name, assignments)

    request_assignment_log_path = (
        Path(args.request_assignment_log_path)
        if args.request_assignment_log_path
        else output_dir / "request_assignments.jsonl"
    )
    request_assignment_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_request_assignment_jsonl(request_assignment_log_path, assignments)

    rows = asyncio.run(
        replay_trace(
            requests=requests,
            assignments=assignments,
            rank_to_proxy=rank_to_proxy,
            model=args.model,
            timeout_s=float(args.timeout),
        )
    )

    write_results_csv(output_dir, rows)
    write_summary_json(
        output_dir,
        requests[0].trace_name,
        requests,
        rows,
        rank_to_proxy,
    )

    failed = sum(1 for row in rows if not row["ok"])
    print(
        f"trace={requests[0].trace_name} requests={len(rows)} "
        f"success={len(rows) - failed} failed={failed} dispatch=round_robin_prefill"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
