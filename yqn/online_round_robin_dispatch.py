# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Replay one trace CSV online against disaggregated prefill/decode proxies.

Input CSV columns:
- arrive_time,input_tokens_length,output_tokens_length

Output files under --output-dir:
- dispatch_requests.csv
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
    """One request row from trace csv."""

    trace_name: str
    request_id: int
    arrive_time: float
    input_tokens: int
    output_tokens: int
    in_str: str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online round-robin trace replay")
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument("--proxy-urls", type=str, required=True, help="comma-separated, e.g. http://127.0.0.1:8000,http://127.0.0.1:8001")
    parser.add_argument("--model", type=str, default="/data/yqn/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def load_trace_requests(trace_path: Path) -> list[TraceRequest]:
    """Load trace csv and keep original order."""
    requests: list[TraceRequest] = []
    with open(trace_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for request_id, row in enumerate(reader):
            in_tok = int(row["input_tokens_length"])
            out_tok = int(row["output_tokens_length"])
            arrive_time = float(row["arrive_time"])
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


def dispatch_round_robin(requests: list[TraceRequest], target_count: int) -> dict[int, list[TraceRequest]]:
    """Round-robin assign requests to target index list [0..target_count)."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    assigned: dict[int, list[TraceRequest]] = {idx: [] for idx in range(target_count)}
    for idx, req in enumerate(requests):
        assigned[idx % target_count].append(req)
    return assigned


def write_dispatch_csv(output_dir: Path, trace_name: str, assignments: dict[int, list[TraceRequest]]) -> None:
    """Keep a dispatch record compatible with offline result analysis."""
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
        for target_idx, reqs in assignments.items():
            for req in reqs:
                writer.writerow(
                    {
                        "trace_name": trace_name,
                        "prefill_rank": target_idx,
                        "request_id": req.request_id,
                        "input_tokens": req.input_tokens,
                        "output_tokens": req.output_tokens,
                        "arrive_time": f"{req.arrive_time:.6f}",
                    }
                )


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

    status_code = -1
    error = ""
    body_bytes = 0
    ok = False
    try:
        async with session.post(request_url, json=payload) as resp:
            body = await resp.read()
            body_bytes = len(body)
            status_code = resp.status
            ok = resp.status == 200
            if not ok:
                error = body.decode("utf-8", errors="ignore")[:2000]
    except Exception as exc:  # pragma: no cover - network/runtime path
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
    proxy_urls: list[str],
    model: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Replay all requests concurrently, preserving trace-based wall clock pacing."""
    run_start_perf = time.perf_counter()
    base_arrive_time = requests[0].arrive_time

    req_to_target: dict[int, int] = {}
    for target_idx, reqs in assignments.items():
        for req in reqs:
            req_to_target[req.request_id] = target_idx

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for req in requests:
            target_idx = req_to_target[req.request_id]
            task = asyncio.create_task(
                _submit_one(
                    session=session,
                    proxy_url=proxy_urls[target_idx],
                    req=req,
                    model=model,
                    base_arrive_time=base_arrive_time,
                    run_start_perf=run_start_perf,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    for row in results:
        row["prefill_rank"] = req_to_target[row["request_id"]]

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
    proxy_urls: list[str],
) -> None:
    ok_rows = [r for r in rows if bool(r["ok"])]
    latencies = [float(r["latency_s"]) for r in ok_rows]

    summary = {
        "trace_name": trace_name,
        "num_requests": len(rows),
        "num_success": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "proxy_urls": proxy_urls,
        "trace_start": float(trace_requests[0].arrive_time),
        "trace_end": float(trace_requests[-1].arrive_time),
        "trace_duration_s": float(trace_requests[-1].arrive_time - trace_requests[0].arrive_time),
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
    if not proxy_urls:
        raise ValueError("--proxy-urls is empty")

    requests = load_trace_requests(trace_path)
    assignments = dispatch_round_robin(requests, len(proxy_urls))
    write_dispatch_csv(output_dir, requests[0].trace_name, assignments)

    rows = asyncio.run(
        replay_trace(
            requests=requests,
            assignments=assignments,
            proxy_urls=proxy_urls,
            model=args.model,
            timeout_s=float(args.timeout),
        )
    )

    write_results_csv(output_dir, rows)
    write_summary_json(output_dir, requests[0].trace_name, requests, rows, proxy_urls)

    failed = sum(1 for row in rows if not row["ok"])
    print(
        f"trace={requests[0].trace_name} requests={len(rows)} "
        f"success={len(rows) - failed} failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
