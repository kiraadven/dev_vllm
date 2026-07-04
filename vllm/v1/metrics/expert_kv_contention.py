# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import torch

TRACE_ENV = "VLLM_EXPERT_KV_CONTENTION_TRACE"
SUMMARY_ENV = "VLLM_EXPERT_KV_CONTENTION_SUMMARY"


def get_trace_path() -> str | None:
    path = os.environ.get(TRACE_ENV)
    return path if path else None


def is_enabled() -> bool:
    return get_trace_path() is not None


def is_summary_enabled() -> bool:
    value = os.environ.get(SUMMARY_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class _TimelineEvent:
    event: str
    lane: str
    layer_name: str | None
    layer_id: int | None
    start_event: Any
    end_event: Any
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ForwardState:
    forward_id: int
    timestamp_ns: int
    request_ids: list[str]
    scheduled_tokens_by_request: dict[str, int]
    total_scheduled_tokens: int
    num_tokens_padded: int
    cudagraph_mode: str
    offload_config: dict[str, Any] | None
    forward_start_event: Any | None
    timeline_events: list[_TimelineEvent] = field(default_factory=list)


_current_forward: contextvars.ContextVar[_ForwardState | None] = contextvars.ContextVar(
    "expert_kv_contention_forward", default=None
)


def has_active_forward() -> bool:
    return is_enabled() and _current_forward.get() is not None


class ExpertKVContentionProfiler:
    def __init__(self) -> None:
        self._forward_id = 0
        self._writer_path: str | None = None
        self._writer = None
        self._lock = Lock()
        self._summary: dict[str, float | int] = _empty_summary()

    def begin_forward(
        self,
        *,
        scheduled_tokens_by_request: dict[str, int],
        total_scheduled_tokens: int,
        num_tokens_padded: int,
        cudagraph_mode: str,
        offload_config: dict[str, Any] | None = None,
    ) -> contextvars.Token[_ForwardState | None] | None:
        if not is_enabled():
            return None
        self._forward_id += 1
        state = _ForwardState(
            forward_id=self._forward_id,
            timestamp_ns=time.time_ns(),
            request_ids=list(scheduled_tokens_by_request),
            scheduled_tokens_by_request=scheduled_tokens_by_request,
            total_scheduled_tokens=total_scheduled_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
            offload_config=offload_config,
            forward_start_event=_record_cuda_event(),
        )
        return _current_forward.set(state)

    def end_forward(
        self,
        token: contextvars.Token[_ForwardState | None] | None,
        *,
        offload_stats: dict[str, Any] | None = None,
    ) -> None:
        if token is None or not is_enabled():
            return
        state = _current_forward.get()
        if state is None:
            _current_forward.reset(token)
            return

        aggregates = {
            "attention_compute_ms": 0.0,
            "moe_compute_ms": 0.0,
            "h2d_comm_ms": 0.0,
            "prefetch_wait_ms": 0.0,
            "copy_bytes": 0,
        }
        active_expert_ratio_sum = 0.0
        active_expert_ratio_count = 0
        layer_records = []

        for event in state.timeline_events:
            timing = _timeline_us(state.forward_start_event, event)
            if timing is None:
                continue
            duration_ms = timing["duration_us"] / 1000.0
            if event.event == "attention_compute":
                aggregates["attention_compute_ms"] += duration_ms
            elif event.event == "moe_compute":
                aggregates["moe_compute_ms"] += duration_ms
                active_ratio = event.extra.get("active_expert_ratio")
                if active_ratio is not None:
                    active_expert_ratio_sum += float(active_ratio)
                    active_expert_ratio_count += 1
            elif event.event == "h2d_comm":
                aggregates["h2d_comm_ms"] += duration_ms
                aggregates["copy_bytes"] += int(event.extra.get("copy_bytes", 0))
            elif event.event == "prefetch_wait":
                aggregates["prefetch_wait_ms"] += duration_ms

            record = {
                "type": "layer_timing",
                "forward_id": state.forward_id,
                "timestamp_ns": time.time_ns(),
                "event": event.event,
                "lane": event.lane,
                "layer_name": event.layer_name,
                "layer_id": event.layer_id,
                "start_us": timing["start_us"],
                "end_us": timing["end_us"],
                "duration_us": timing["duration_us"],
                "duration_ms": duration_ms,
                "total_scheduled_tokens": state.total_scheduled_tokens,
                "ms_per_scheduled_token": _safe_div(
                    duration_ms, state.total_scheduled_tokens
                ),
            }
            record.update(event.extra)
            layer_records.append(record)

        active_expert_ratio = _safe_div(
            active_expert_ratio_sum, active_expert_ratio_count
        )
        overlap_ratio = _overlap_ratio(
            aggregates["h2d_comm_ms"], aggregates["prefetch_wait_ms"]
        )
        forward_record = {
            "type": "forward",
            "forward_id": state.forward_id,
            "timestamp_ns": state.timestamp_ns,
            "num_reqs": len(state.request_ids),
            "request_ids": state.request_ids,
            "scheduled_tokens_by_request": state.scheduled_tokens_by_request,
            "total_scheduled_tokens": state.total_scheduled_tokens,
            "num_tokens_padded": state.num_tokens_padded,
            "cudagraph_mode": state.cudagraph_mode,
            "attention_compute_ms": aggregates["attention_compute_ms"],
            "moe_compute_ms": aggregates["moe_compute_ms"],
            "h2d_comm_ms": aggregates["h2d_comm_ms"],
            "prefetch_wait_ms": aggregates["prefetch_wait_ms"],
            "copy_bytes": aggregates["copy_bytes"],
            "attention_ms_per_scheduled_token": _safe_div(
                aggregates["attention_compute_ms"], state.total_scheduled_tokens
            ),
            "moe_ms_per_scheduled_token": _safe_div(
                aggregates["moe_compute_ms"], state.total_scheduled_tokens
            ),
            "active_expert_ratio": active_expert_ratio,
            "prefetch_overlap_ratio": overlap_ratio,
            "offload_config": state.offload_config,
            "offload_stats": offload_stats,
        }
        self._write(forward_record)
        for record in layer_records:
            self._write(record)

        self._summary["forwards"] = int(self._summary["forwards"]) + 1
        self._summary["layer_timing_records"] = int(
            self._summary["layer_timing_records"]
        ) + len(layer_records)
        self._summary["total_scheduled_tokens"] = int(
            self._summary["total_scheduled_tokens"]
        ) + state.total_scheduled_tokens
        self._summary["total_attention_ms"] = float(
            self._summary["total_attention_ms"]
        ) + aggregates["attention_compute_ms"]
        self._summary["total_moe_ms"] = float(
            self._summary["total_moe_ms"]
        ) + aggregates["moe_compute_ms"]
        self._summary["total_h2d_copy_ms"] = float(
            self._summary["total_h2d_copy_ms"]
        ) + aggregates["h2d_comm_ms"]
        self._summary["total_prefetch_wait_ms"] = float(
            self._summary["total_prefetch_wait_ms"]
        ) + aggregates["prefetch_wait_ms"]
        self._summary["total_prefetch_copy_bytes"] = int(
            self._summary["total_prefetch_copy_bytes"]
        ) + int(aggregates["copy_bytes"])
        if active_expert_ratio is not None:
            self._summary["active_expert_ratio_sum"] = float(
                self._summary["active_expert_ratio_sum"]
            ) + active_expert_ratio
            self._summary["active_expert_ratio_count"] = int(
                self._summary["active_expert_ratio_count"]
            ) + 1

        _current_forward.reset(token)

    def record_timeline_event(
        self,
        *,
        event: str,
        lane: str,
        layer_name: str | None,
        layer_id: int | None,
        start_event: Any,
        end_event: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not is_enabled():
            return
        state = _current_forward.get()
        if state is None:
            return
        state.timeline_events.append(
            _TimelineEvent(
                event=event,
                lane=lane,
                layer_name=layer_name,
                layer_id=layer_id,
                start_event=start_event,
                end_event=end_event,
                extra=extra or {},
            )
        )

    def record_scheduler_stats(
        self,
        *,
        scheduler_stats: Any,
        iteration_stats: Any | None,
        engine_idx: int,
    ) -> None:
        if not is_enabled() or scheduler_stats is None:
            return
        prefix_cache_stats = scheduler_stats.prefix_cache_stats
        num_preempted_reqs = (
            iteration_stats.num_preempted_reqs if iteration_stats is not None else 0
        )
        prefix_queries = prefix_cache_stats.queries + prefix_cache_stats.preempted_queries
        prefix_hits = prefix_cache_stats.hits + prefix_cache_stats.preempted_hits
        record = {
            "type": "scheduler",
            "timestamp_ns": time.time_ns(),
            "engine_idx": engine_idx,
            "num_running_reqs": scheduler_stats.num_running_reqs,
            "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
            "num_skipped_waiting_reqs": scheduler_stats.num_skipped_waiting_reqs,
            "kv_cache_usage": scheduler_stats.kv_cache_usage,
            "num_preempted_reqs": num_preempted_reqs,
            "prefix_cache_queries": prefix_queries,
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_hit_rate": _safe_div(prefix_hits, prefix_queries),
            "prefix_cache": {
                "requests": prefix_cache_stats.requests,
                "queries": prefix_cache_stats.queries,
                "hits": prefix_cache_stats.hits,
                "preempted_requests": prefix_cache_stats.preempted_requests,
                "preempted_queries": prefix_cache_stats.preempted_queries,
                "preempted_hits": prefix_cache_stats.preempted_hits,
            },
        }
        self._write(record)
        self._summary["scheduler_records"] = int(self._summary["scheduler_records"]) + 1
        self._summary["total_preemptions"] = int(
            self._summary["total_preemptions"]
        ) + int(num_preempted_reqs)
        self._summary["kv_cache_usage_sum"] = float(
            self._summary["kv_cache_usage_sum"]
        ) + float(scheduler_stats.kv_cache_usage)
        self._summary["kv_cache_usage_count"] = int(
            self._summary["kv_cache_usage_count"]
        ) + 1
        self._summary["prefix_cache_queries"] = int(
            self._summary["prefix_cache_queries"]
        ) + int(prefix_queries)
        self._summary["prefix_cache_hits"] = int(
            self._summary["prefix_cache_hits"]
        ) + int(prefix_hits)

    def record_request_final_stats(
        self,
        *,
        iteration_stats: Any | None,
        engine_idx: int,
    ) -> None:
        if not is_enabled() or iteration_stats is None:
            return
        for request_stats in iteration_stats.finished_requests:
            record = {
                "type": "request_final",
                "timestamp_ns": time.time_ns(),
                "engine_idx": engine_idx,
                "request_id": request_stats.request_id,
                "finish_reason": str(request_stats.finish_reason),
                "e2e_latency_s": request_stats.e2e_latency,
                "queued_time_s": request_stats.queued_time,
                "prefill_time_s": request_stats.prefill_time,
                "inference_time_s": request_stats.inference_time,
                "decode_time_s": request_stats.decode_time,
                "mean_time_per_output_token_s": request_stats.mean_time_per_output_token,
                "num_prompt_tokens": request_stats.num_prompt_tokens,
                "num_generation_tokens": request_stats.num_generation_tokens,
                "max_tokens_param": request_stats.max_tokens_param,
                "num_cached_tokens": request_stats.num_cached_tokens,
                "num_preemptions": getattr(request_stats, "num_preemptions", 0),
                "is_corrupted": request_stats.is_corrupted,
            }
            self._write(record)
            self._summary["request_final_records"] = int(
                self._summary["request_final_records"]
            ) + 1

    def get_summary(self) -> dict[str, Any]:
        return _finalize_summary(dict(self._summary), get_trace_path())

    def _write(self, record: dict[str, Any]) -> None:
        writer = self._get_writer()
        if writer is None:
            return
        with self._lock:
            json.dump(record, writer, separators=(",", ":"))
            writer.write("\n")
            writer.flush()

    def _get_writer(self):
        path = get_trace_path()
        if path is None:
            return None
        if self._writer is not None and self._writer_path == path:
            return self._writer
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = trace_path.open("a", encoding="utf-8", buffering=1)
        self._writer_path = path
        return self._writer


_profiler: ExpertKVContentionProfiler | None = None


def get_profiler() -> ExpertKVContentionProfiler:
    global _profiler
    if _profiler is None:
        _profiler = ExpertKVContentionProfiler()
    return _profiler


def start_cuda_timing() -> tuple[Any, Any] | None:
    if not has_active_forward():
        return None
    start_event = _record_cuda_event()
    if start_event is None:
        return None
    end_event = torch.cuda.Event(enable_timing=True)
    return start_event, end_event


def end_cuda_timing(timing: tuple[Any, Any] | None) -> tuple[Any, Any] | None:
    if timing is None:
        return None
    _, end_event = timing
    end_event.record(torch.cuda.current_stream())
    return timing


def summarize_trace_file(trace_path: str | None = None) -> dict[str, Any] | None:
    path = trace_path or get_trace_path()
    if not path:
        return None
    trace_file = Path(path)
    if not trace_file.exists():
        return None

    summary: dict[str, Any] = _empty_summary()
    with trace_file.open(encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = record.get("type")
            if record_type == "forward":
                _observe_forward_record(summary, record)
            elif record_type == "scheduler":
                _observe_scheduler_record(summary, record)
            elif record_type == "request_final":
                summary["request_final_records"] += 1
            elif record_type == "layer_timing":
                summary["layer_timing_records"] += 1
    return _finalize_summary(summary, str(trace_file))


def _record_cuda_event() -> Any | None:
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream())
    return event


def _timeline_us(
    forward_start_event: Any | None, event: _TimelineEvent
) -> dict[str, float] | None:
    try:
        event.end_event.synchronize()
        duration_us = event.start_event.elapsed_time(event.end_event) * 1000.0
        if forward_start_event is None:
            return {
                "start_us": 0.0,
                "end_us": duration_us,
                "duration_us": duration_us,
            }
        start_us = forward_start_event.elapsed_time(event.start_event) * 1000.0
        end_us = forward_start_event.elapsed_time(event.end_event) * 1000.0
        return {"start_us": start_us, "end_us": end_us, "duration_us": duration_us}
    except RuntimeError:
        return None


def _observe_forward_record(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["forwards"] += 1
    summary["total_scheduled_tokens"] += int(record.get("total_scheduled_tokens", 0))
    summary["total_attention_ms"] += float(record.get("attention_compute_ms", 0.0))
    summary["total_moe_ms"] += float(record.get("moe_compute_ms", 0.0))
    summary["total_h2d_copy_ms"] += float(record.get("h2d_comm_ms", 0.0))
    summary["total_prefetch_wait_ms"] += float(record.get("prefetch_wait_ms", 0.0))
    summary["total_prefetch_copy_bytes"] += int(record.get("copy_bytes", 0))
    active_ratio = record.get("active_expert_ratio")
    if active_ratio is not None:
        summary["active_expert_ratio_sum"] += float(active_ratio)
        summary["active_expert_ratio_count"] += 1


def _observe_scheduler_record(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["scheduler_records"] += 1
    summary["total_preemptions"] += int(record.get("num_preempted_reqs", 0))
    kv_cache_usage = record.get("kv_cache_usage")
    if kv_cache_usage is not None:
        summary["kv_cache_usage_sum"] += float(kv_cache_usage)
        summary["kv_cache_usage_count"] += 1
    summary["prefix_cache_queries"] += int(record.get("prefix_cache_queries", 0))
    summary["prefix_cache_hits"] += int(record.get("prefix_cache_hits", 0))


def _finalize_summary(summary: dict[str, Any], trace_path: str | None) -> dict[str, Any]:
    summary["trace_path"] = trace_path
    summary["avg_kv_cache_usage"] = _safe_div(
        summary["kv_cache_usage_sum"], summary["kv_cache_usage_count"]
    )
    summary["prefix_cache_hit_rate"] = _safe_div(
        summary["prefix_cache_hits"], summary["prefix_cache_queries"]
    )
    summary["avg_active_expert_ratio"] = _safe_div(
        summary["active_expert_ratio_sum"], summary["active_expert_ratio_count"]
    )
    summary["attention_ms_per_scheduled_token"] = _safe_div(
        summary["total_attention_ms"], summary["total_scheduled_tokens"]
    )
    summary["moe_ms_per_scheduled_token"] = _safe_div(
        summary["total_moe_ms"], summary["total_scheduled_tokens"]
    )
    summary["prefetch_overlap_ratio"] = _overlap_ratio(
        summary["total_h2d_copy_ms"], summary["total_prefetch_wait_ms"]
    )
    return summary


def _empty_summary() -> dict[str, float | int]:
    return {
        "forwards": 0,
        "scheduler_records": 0,
        "layer_timing_records": 0,
        "request_final_records": 0,
        "total_scheduled_tokens": 0,
        "total_attention_ms": 0.0,
        "total_moe_ms": 0.0,
        "total_h2d_copy_ms": 0.0,
        "total_prefetch_wait_ms": 0.0,
        "total_prefetch_copy_bytes": 0,
        "total_preemptions": 0,
        "kv_cache_usage_sum": 0.0,
        "kv_cache_usage_count": 0,
        "prefix_cache_queries": 0,
        "prefix_cache_hits": 0,
        "active_expert_ratio_sum": 0.0,
        "active_expert_ratio_count": 0,
    }


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _overlap_ratio(copy_ms: float, wait_ms: float) -> float | None:
    if copy_ms <= 0:
        return None
    return max(0.0, 1.0 - wait_ms / copy_ms)
